#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cstdlib> // Needed for std::atoi and std::atof
#include <Eigen/Dense>
#include "PendulumState.hpp"
#include "SpatialMath.hpp"
#include "ABISolver.hpp"

// This is essentially what is the get_state_dot in python
void evaluate_derivatives(PendulumState& state, 
                          std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& k_theta, 
                          std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& k_beta,
                          double t) {
    ABISolver::compute_ATBI(state);
    for (int i = 1; i <= state.n; ++i) {
        k_theta[i] = SpatialMath::spherical_derivative(state.theta[i], state.beta[i]);
        k_beta[i]  = state.beta_dot[i]; 
    }
}

// main function.
int main(int argc, char* argv[]) {
    // --- 0. Read Command Line Arguments ---
    // Set defaults just in case you run it without arguments directly from terminal
    int n = 3; 
    double dt = 0.001;
    double t_end = 1.0;

    // Overwrite defaults if arguments are provided by the Python script
    if (argc >= 4) {
        n = std::atoi(argv[1]);
        dt = std::atof(argv[2]);
        t_end = std::atof(argv[3]);
    } else {
        std::cout << "Tip: You can run this with arguments: ./sim <n> <dt> <t_end>\n";
    }

    std::cout << "Simulating: n=" << n << ", dt=" << dt << ", t_end=" << t_end << std::endl;

    PendulumState state(n); //setting up the pendulum state struct once! when we just overwrite.

    // --- 1. Link Geometry and Inertias (Matching Python implmentation and the one in the bachelor) ---
    for (int i = 1; i <= n; ++i) {
        double mass = 20.0;
        state.l_hinge[i] = Eigen::Vector3d(0.0, 0.0, 0.2); 
        
        Eigen::Vector3d l_com = state.l_hinge[i] / 2.0;
        double l = state.l_hinge[i].norm();
        double w = l / 50.0;
        double h = w;

        // Diagonal elements of J_c
        Eigen::Vector3d J_c_diag(
            (1.0 / 12.0) * mass * (h*h + l*l),
            (1.0 / 12.0) * mass * (w*w + l*l),
            (1.0 / 12.0) * mass * (w*w + h*h)
        );
        Eigen::Matrix3d J_c = J_c_diag.asDiagonal();

        // Spatial inertia at Center of Mass (M_c)
        Matrix6d M_c = Matrix6d::Zero();
        M_c.block<3, 3>(0, 0) = J_c;
        M_c.block<3, 3>(3, 3) = mass * Eigen::Matrix3d::Identity();

        // Shift spatial inertia to the hinge frame (M) using Rigid Body Transformation
        Matrix6d RBT_com = SpatialMath::RBT(l_com);
        state.M[i] = RBT_com * M_c * RBT_com.transpose();

        // Standard RBT across the entire link
        state.RBT[i] = SpatialMath::RBT(state.l_hinge[i]);
    }
    
    // Set the initial angle of the base joint (n) 
    state.theta[n] << SpatialMath::quatfromrev(3*M_PI/4, 'y');
    

    // --- 2. Simulation Parameters ---
    int steps = static_cast<int>(t_end / dt);
    
    //preallocation of the "k" values used in rk4. These arrays are built to store the intermediate values that arise in RK4
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> k1_theta(n + 2), k2_theta(n + 2), k3_theta(n + 2), k4_theta(n + 2);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> k1_beta(n + 2), k2_beta(n + 2), k3_beta(n + 2), k4_beta(n + 2);

    //we need another state object, because when we calculate k1,k2 ... we cannot overwrite the original state
    //the original state is needed for the final step in RK4. Thus temp_state will be passed
    PendulumState temp_state = state; 

    // --- 3. Dynamic CSV Setup ---. This i dont know. Entirely vibecoded.
    std::ofstream data_file("pendulum_data.csv");
    if (!data_file.is_open()) {
        std::cerr << "Failed to open data file!" << std::endl;
        return 1;
    }
    
    data_file << "time";
    for (int i = 1; i <= n; ++i) {
        data_file << ",beta_dot" << i << "_x,beta_dot" << i << "_y,beta_dot" << i << "_z";
    }
    data_file << "\n";

    // --- 4. The RK4 Integration Loop ---
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= steps; ++step) {
        
        double t = step * dt;
        
        evaluate_derivatives(state, k1_theta, k1_beta, t);

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
        evaluate_derivatives(temp_state, k2_theta, k2_beta, t + 0.5 * dt);

        for(int i=1; i<=n; ++i) {
            temp_state.theta[i] = state.theta[i] + 0.5 * dt * k2_theta[i];
            SpatialMath::normalize_quaternion(temp_state.theta[i]);
            temp_state.beta[i]  = state.beta[i] + 0.5 * dt * k2_beta[i];
        }
        evaluate_derivatives(temp_state, k3_theta, k3_beta, t + 0.5 * dt);

        for(int i=1; i<=n; ++i) {
            temp_state.theta[i] = state.theta[i] + dt * k3_theta[i];
            SpatialMath::normalize_quaternion(temp_state.theta[i]);
            temp_state.beta[i]  = state.beta[i] + dt * k3_beta[i];
        }
        evaluate_derivatives(temp_state, k4_theta, k4_beta, t + dt);

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