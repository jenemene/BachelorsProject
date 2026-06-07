#pragma once
#include <Eigen/Dense>
#include <vector>

//make a big struct instead of a class. Its makes the translation from python a lot easier, as the code kind of resembles
//big difference is that we have to declare what datatype everythingis

// Define standard spatial types to keep the code clean. 
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3x6 = Eigen::Matrix<double, 3, 6>;

// Setting up the struct. This is almost the equivalent to the multibody class, except this doesnt hold the alrogithms itself (ATBI, RK4 etc)
// Essentially contains all variables of interest that is associated with the state of the system at time t. 
// could probably be renamed to system state lol
struct PendulumState {
    int n; // Number of bodies

    // 1. Link Properties 
    // We must use Eigen::aligned_allocator so the CPU can vectorize the math. i do not know why, will have to look into that.
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> l_hinge;
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> M;
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> RBT;
    std::vector<Matrix3x6, Eigen::aligned_allocator<Matrix3x6>> H; 

    // 2. Joint State (Gen coords and vels). For now this only handles spherical joints.
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> theta; // [x,y,z,w] Quaternions
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> beta;  // Joint velocities

    // 3. Quantities for outward sweep (Kinematic Scatter)
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> V;
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> agothic;
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> bgothic;

    // 4. ABI quantites
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> P_plus;
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> tau_bar;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> D_inv; // Storing the inverse directly
    std::vector<Eigen::Matrix<double, 6, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 3>>> G;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> nu;

    // Additional variables, unsure where to place.
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> xi_plus;
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> A;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> tau;   // Joint torques
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> beta_dot; // Output accelerations

    // Constructor to pre-allocate all arrays based on number of bodies.
    // This is kind of __init__ in python. Pendulum state is called, it preallocates the arrays containing the relevant quantities. 
    //they are arrays containing quantities for each body. So the H.resize creates and array like [H[0],H[1],H[2],...H[n],H[n+1]]
    //this is only called once to preallocate in memory. Afterwards the values are simply overwritten, such that it isnt called for each timestep t.
    PendulumState(int num_bodies) : n(num_bodies) {
        int size = n + 2; // +2 to accommodate boundary conditions at 0 and n+1

        // Initialize properties
        l_hinge.resize(size, Eigen::Vector3d::Zero());
        M.resize(size, Matrix6d::Zero());
        RBT.resize(size, Matrix6d::Identity());
        
        // H for spherical joints is a 3x6 matrix: [I_3x3, 0_3x3]
        Matrix3x6 spherical_H;
        spherical_H << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();
        H.resize(size, spherical_H);

        // Initialize states and spatial vectors to zero
        theta.resize(size, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)); // Default valid quaternion
        beta.resize(size, Eigen::Vector3d::Zero());
        tau.resize(size, Eigen::Vector3d::Zero());

        V.resize(size, Vector6d::Zero());
        agothic.resize(size, Vector6d::Zero());
        bgothic.resize(size, Vector6d::Zero());

        P_plus.resize(size, Matrix6d::Zero());
        tau_bar.resize(size, Matrix6d::Zero());
        D_inv.resize(size, Eigen::Matrix3d::Zero());
        G.resize(size, Eigen::Matrix<double, 6, 3>::Zero());
        nu.resize(size, Eigen::Vector3d::Zero());


        xi_plus.resize(size, Vector6d::Zero());
        A.resize(size, Vector6d::Zero());
        beta_dot.resize(size, Eigen::Vector3d::Zero());
    }
};