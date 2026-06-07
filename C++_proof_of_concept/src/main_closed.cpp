#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <Eigen/Dense>
#include "PendulumState.hpp"
#include "SpatialMath.hpp"
#include "ABISolver.hpp"
#include "SOAConstraints.hpp"

// --- Helper to compute Forward Kinematics (Global Rotations and Positions) ---
struct FKResult {
    std::vector<Eigen::Vector3d> pos;
    std::vector<Matrix6d> IR;
};

inline FKResult compute_FK(const PendulumState& state) {
    int n = state.n;
    FKResult fk;
    fk.pos.resize(n + 1, Eigen::Vector3d::Zero());
    fk.IR.resize(n + 1, Matrix6d::Identity());

    fk.IR[n] = SpatialMath::spatialrotfromquat(state.theta[n]);
    fk.pos[n] = Eigen::Vector3d::Zero(); 

    for (int i = n - 1; i >= 1; --i) {
        Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[i]);
        fk.IR[i] = fk.IR[i + 1] * pRc;
        Eigen::Matrix3d R_cum_3x3 = fk.IR[i + 1].block<3, 3>(0, 0);
        
        fk.pos[i] = fk.pos[i + 1] + R_cum_3x3 * state.l_hinge[i]; 
    }
    return fk;
}

// --- The Closed-Loop Derivative Evaluator ---
void evaluate_derivatives_closed(PendulumState& state, 
                                 std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& k_theta, 
                                 std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& k_beta,
                                 double alpha_bg, double beta_bg,
                                 double t) {
    int n = state.n;

    // 0. Joint Damping
    double c = 0.0;
    
    for (int k = 1; k <= n; ++k) {
        state.tau[k] = -c * state.beta[k];
    }

    // 1. Unconstrained Forward Dynamics
    ABISolver::compute_ATBI(state);

    // 2. Forward Kinematics
    FKResult fk = compute_FK(state);

    Eigen::Matrix3d IR1_3 = fk.IR[1].block<3, 3>(0, 0);
    Eigen::Matrix3d IRn_3 = fk.IR[n].block<3, 3>(0, 0);

    // 3. Compute Kinematic Error
    Eigen::Vector3d tip_pos = fk.pos[1] + IR1_3 * state.l_hinge[1];
    Eigen::Vector3d base_pos = fk.pos[n];
    Eigen::Vector3d Phi = -(base_pos - tip_pos);

    Eigen::Vector3d omega_1_inertial = IR1_3 * state.V[1].head<3>();
    Eigen::Matrix3d I_omega_IO = SpatialMath::skew(omega_1_inertial); 

    Eigen::Vector3d v_tip = IR1_3 * state.V[1].tail<3>() + I_omega_IO * (IR1_3 * state.l_hinge[1]);
    Eigen::Vector3d v_base = IRn_3 * state.V[n].tail<3>();
    Eigen::Vector3d Phi_dot = -(v_base - v_tip);

    Eigen::Vector3d alpha_1_inertial = IR1_3 * state.A[1].head<3>();
    Eigen::Matrix3d I_alpha_IO = SpatialMath::skew(alpha_1_inertial);

    Eigen::Vector3d a_tip = IR1_3 * state.A[1].tail<3>() 
                          + I_alpha_IO * (IR1_3 * state.l_hinge[1]) 
                          + I_omega_IO * I_omega_IO * (IR1_3 * state.l_hinge[1]);
    Eigen::Vector3d a_base = IRn_3 * state.A[n].tail<3>();
    Eigen::Vector3d Phi_ddot_free = -(a_base - a_tip);

    // 4. Baumgarte Stabilization
    Eigen::Vector3d f = Phi_ddot_free + 2.0 * alpha_bg * Phi_dot + (beta_bg * beta_bg) * Phi;

    // 5. Operational Space Inertia
    auto omega_diag = SOAConstraints::compute_omega_diag(state);
    Matrix6d omega_11 = omega_diag[1];
    Matrix6d omega_nn = omega_diag[n];
    Matrix6d omega_n1 = SOAConstraints::compute_omega_ij(n, 1, state, omega_diag);

    Matrix6d Lambda_11 = fk.IR[1] * (state.RBT[1].transpose() * omega_11 * state.RBT[1]) * fk.IR[1].transpose();
    Matrix6d Lambda_nn = fk.IR[n] * omega_nn * fk.IR[n].transpose();
    Matrix6d Lambda_n1 = fk.IR[1] * (omega_n1 * state.RBT[1]) * fk.IR[1].transpose();

    Eigen::Matrix<double, 12, 12> Lambda_block;
    Lambda_block.block<6, 6>(0, 0) = Lambda_11;
    Lambda_block.block<6, 6>(6, 6) = Lambda_nn;
    Lambda_block.block<6, 6>(0, 6) = Lambda_n1.transpose();
    Lambda_block.block<6, 6>(6, 0) = Lambda_n1;

    Eigen::Matrix<double, 3, 12> Q = Eigen::Matrix<double, 3, 12>::Zero();
    Q.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();  
    Q.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity(); 

    // 6. Solve for Constraint Forces
    Eigen::Matrix3d M_eff = Q * Lambda_block * Q.transpose();
    Eigen::Vector3d lambda = -M_eff.inverse() * f;

    // 7. Apply Corrections
    Eigen::Matrix<double, 12, 1> f_c_closed = -Q.transpose() * lambda;

    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> f_c(n + 2, Vector6d::Zero());
    f_c[1] = state.RBT[1] * fk.IR[1].transpose() * f_c_closed.head<6>();
    f_c[n] = fk.IR[n].transpose() * f_c_closed.tail<6>();

    SOAConstraints::apply_beta_dot_delta(state, f_c);

    // 8. Finalize RK4 Parameters
    for (int i = 1; i <= n; ++i) {
        k_theta[i] = SpatialMath::spherical_derivative(state.theta[i], state.beta[i]);
        k_beta[i]  = state.beta_dot[i]; 
    }
}

// --- Main Simulation Loop ---
int main(int argc, char* argv[]) {
    int n = 3; 
    double dt = 0.001;
    double t_end = 1.0;
    double alpha_bg = 50.0;
    double beta_bg = 50.0;

    if (argc >= 6) {
        n = std::atoi(argv[1]);
        dt = std::atof(argv[2]);
        t_end = std::atof(argv[3]);
        alpha_bg = std::atof(argv[4]);
        beta_bg = std::atof(argv[5]);
    } else {
        std::cout << "Tip: Run with: ./sim_closed <n> <dt> <t_end> <alpha> <beta>\n";
    }

    std::cout << "Simulating CLOSED LOOP: n=" << n << ", dt=" << dt << ", t_end=" << t_end 
              << ", alpha=" << alpha_bg << ", beta=" << beta_bg << std::endl;

    PendulumState state(n);

    // Geometry and Inertias
    for (int i = 1; i <= n; ++i) {
        double mass = 20.0;
        state.l_hinge[i] = Eigen::Vector3d(0.0, 0.0, 0.2); 
        
        Eigen::Vector3d l_com = state.l_hinge[i] / 2.0;
        double l = state.l_hinge[i].norm();
        double w = l / 50.0;
        double h = w;

        Eigen::Vector3d J_c_diag(
            (1.0 / 12.0) * mass * (h*h + l*l),
            (1.0 / 12.0) * mass * (w*w + l*l),
            (1.0 / 12.0) * mass * (w*w + h*h)
        );
        Eigen::Matrix3d J_c = J_c_diag.asDiagonal();

        Matrix6d M_c = Matrix6d::Zero();
        M_c.block<3, 3>(0, 0) = J_c;
        M_c.block<3, 3>(3, 3) = mass * Eigen::Matrix3d::Identity();

        Matrix6d RBT_com = SpatialMath::RBT(l_com);
        state.M[i] = RBT_com * M_c * RBT_com.transpose();
        state.RBT[i] = SpatialMath::RBT(state.l_hinge[i]);
    }

    // Initialize triangle perfectly using M_PI
    state.theta[n] = SpatialMath::quatfromrev(M_PI / 2.0, 'y');
    state.theta[2] = SpatialMath::quatfromrev(2.0 * M_PI / 3.0, 'y');
    state.theta[1] = SpatialMath::quatfromrev(2.0 * M_PI / 3.0, 'y');

    int steps = static_cast<int>(t_end / dt);

    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> k1_theta(n + 2), k2_theta(n + 2), k3_theta(n + 2), k4_theta(n + 2);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> k1_beta(n + 2), k2_beta(n + 2), k3_beta(n + 2), k4_beta(n + 2);

    PendulumState temp_state = state;

    std::ofstream data_file("/home/jenz/Desktop/SOAinC/pendulum_data.csv");
    if (!data_file.is_open()) return 1;
    
    data_file << "time";
    for (int i = 1; i <= n; ++i) {
        data_file << ",beta_dot" << i << "_x,beta_dot" << i << "_y,beta_dot" << i << "_z";
    }
    data_file << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // RK4 Integration Loop
    for (int step = 0; step <= steps; ++step) {
        
        double t = step * dt;
        evaluate_derivatives_closed(state, k1_theta, k1_beta, alpha_bg, beta_bg, t);

        data_file << step * dt;
        for(int i = 1; i <= n; ++i) {
            data_file << "," << state.beta_dot[i](0) << "," << state.beta_dot[i](1) << "," << state.beta_dot[i](2);
        }
        data_file << "\n";

        for(int i=1; i<=n; ++i) {
            temp_state.theta[i] = state.theta[i] + 0.5 * dt * k1_theta[i];
            SpatialMath::normalize_quaternion(temp_state.theta[i]);
            temp_state.beta[i]  = state.beta[i] + 0.5 * dt * k1_beta[i];
        }
        evaluate_derivatives_closed(temp_state, k2_theta, k2_beta, alpha_bg, beta_bg, t + 0.5 * dt);

        for(int i=1; i<=n; ++i) {
            temp_state.theta[i] = state.theta[i] + 0.5 * dt * k2_theta[i];
            SpatialMath::normalize_quaternion(temp_state.theta[i]);
            temp_state.beta[i]  = state.beta[i] + 0.5 * dt * k2_beta[i];
        }
        evaluate_derivatives_closed(temp_state, k3_theta, k3_beta, alpha_bg, beta_bg, t + 0.5 * dt);

        for(int i=1; i<=n; ++i) {
            temp_state.theta[i] = state.theta[i] + dt * k3_theta[i];
            SpatialMath::normalize_quaternion(temp_state.theta[i]);
            temp_state.beta[i]  = state.beta[i] + dt * k3_beta[i];
        }
        evaluate_derivatives_closed(temp_state, k4_theta, k4_beta, alpha_bg, beta_bg, t + dt);

        for(int i=1; i<=n; ++i) {
            state.theta[i] += (dt / 6.0) * (k1_theta[i] + 2.0*k2_theta[i] + 2.0*k3_theta[i] + k4_theta[i]);
            SpatialMath::normalize_quaternion(state.theta[i]);
            state.beta[i]  += (dt / 6.0) * (k1_beta[i]  + 2.0*k2_beta[i]  + 2.0*k3_beta[i]  + k4_beta[i]);
        }
    }

    data_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Simulation Time: " << diff.count() << " seconds" << std::endl;

    return 0;
}