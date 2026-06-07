#pragma once
#include <Eigen/Dense>

// Define standard spatial types to keep the code clean
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

namespace SpatialMath {

    // ---------------------------------------------------------
    // 3D Vector Math
    // ---------------------------------------------------------

    // Convert a 3D vector to a 3x3 skew-symmetric matrix.
    inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S <<   0.0, -v(2),  v(1),
              v(2),   0.0, -v(0),
             -v(1),  v(0),   0.0;
        return S;
    }

    // ---------------------------------------------------------
    // Custom Quaternion Math (Convention: [x, y, z, w])
    // ---------------------------------------------------------

    // Convert a quaternion to a 3x3 rotation matrix.
    inline Eigen::Matrix3d rotfromquat(const Eigen::Vector4d& quat) {
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        
        double q0 = quat(3);                // Scalar part
        Eigen::Vector3d q = quat.head<3>(); // Vector part
        
        Eigen::Matrix3d q_tilde = skew(q);
        
        return I + 2.0 * (q0 * I + q_tilde) * q_tilde;
    }

    // Convert a quaternion to a 6x6 spatial rotation matrix.
    inline Matrix6d spatialrotfromquat(const Eigen::Vector4d& quat) {
        Matrix6d spatialR = Matrix6d::Zero();
        Eigen::Matrix3d R = rotfromquat(quat);
        
        spatialR.block<3, 3>(0, 0) = R;
        spatialR.block<3, 3>(3, 3) = R;
        return spatialR;
    }

    // Normalize a quaternion in place to prevent drift during integration.
    inline void normalize_quaternion(Eigen::Vector4d& quat) {
        quat.normalize(); 
    }

    // Map angular velocity (omega) to quaternion derivative (theta_dot) for a spherical joint
    inline Eigen::Vector4d spherical_derivative(const Eigen::Vector4d& theta, const Eigen::Vector3d& omega) {
        Eigen::Matrix4d W = Eigen::Matrix4d::Zero();
        
        W.block<3, 3>(0, 0) = -skew(omega);
        W.block<3, 1>(0, 3) = omega;
        W.block<1, 3>(3, 0) = -omega.transpose();
        // W(3,3) is already 0.0 from initialization
        
        return 0.5 * W * theta; 
    }

    // Get quartenion based on single angle and axis
    inline Eigen::Vector4d quatfromrev(double theta, char axis) {
        Eigen::Vector3d n = Eigen::Vector3d::Zero();
        
        if (axis == 'x' || axis == 'X') {
            n << 1.0, 0.0, 0.0;
        } else if (axis == 'y' || axis == 'Y') {
            n << 0.0, 1.0, 0.0;
        } else if (axis == 'z' || axis == 'Z') {
            n << 0.0, 0.0, 1.0;
        } else {
            throw std::invalid_argument("Axis must be 'x', 'y', or 'z'");
        }

        Eigen::Vector4d q;
        // Vector part (x, y, z)
        q.head<3>() = std::sin(theta / 2.0) * n; 
        // Scalar part (w)
        q(3) = std::cos(theta / 2.0);            

        return q;
    }
    // ---------------------------------------------------------
    // Spatial Math (6x6 Operators)
    // ---------------------------------------------------------

    // Rigid body transformation matrix - no rotations.
    inline Matrix6d RBT(const Eigen::Vector3d& vec) {
        Matrix6d phi = Matrix6d::Identity();
        phi.block<3, 3>(0, 3) = skew(vec);
        return phi;
    }
  
    // Convert a 6D spatial vector to a 6x6 skew-symmetric matrix (for motion).
    inline Matrix6d spatial_cross(const Vector6d& spatialvec) {
        Matrix6d S = Matrix6d::Zero();
        Eigen::Matrix3d w_tilde = skew(spatialvec.head<3>());
        Eigen::Matrix3d v_tilde = skew(spatialvec.tail<3>());

        S.block<3, 3>(0, 0) = w_tilde;
        S.block<3, 3>(3, 0) = v_tilde;
        S.block<3, 3>(3, 3) = w_tilde;
        return S;
    }

    // Convert a 6D spatial vector to a 6x6 skew-symmetric matrix (for forces).
    inline Matrix6d spatial_cross_dual(const Vector6d& X) {
        Matrix6d X_bar = Matrix6d::Zero();
        Eigen::Matrix3d w_tilde = skew(X.head<3>());
        Eigen::Matrix3d v_tilde = skew(X.tail<3>());

        X_bar.block<3, 3>(0, 0) = w_tilde;
        X_bar.block<3, 3>(0, 3) = v_tilde;
        X_bar.block<3, 3>(3, 3) = w_tilde;
        return X_bar;
    }

} // namespace SpatialMath