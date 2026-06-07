#pragma once
#include "PendulumState.hpp"
#include "SpatialMath.hpp"
#include <Eigen/Dense>
#include <vector>

namespace SOAConstraints {

    // ---------------------------------------------------------
    // 1. Beta Dot Delta (Correction Sweep)
    // Translates Python's 'beta_dot_delta'
    // ---------------------------------------------------------
    inline void apply_beta_dot_delta(PendulumState& state, const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>>& f_c) {
        int n = state.n;

        std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> xi_delta(n + 2, Vector6d::Zero());
        std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> lambda_list(n + 2, Vector6d::Zero());
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> nu_delta(n + 2, Eigen::Vector3d::Zero());

        // Gather Pass (Tip to Base)
        for (int k = 1; k <= n; ++k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k-1]);
            
            xi_delta[k] = state.RBT[k] * pRc * state.tau_bar[k-1] * xi_delta[k-1] - f_c[k];
            nu_delta[k] = state.D_inv[k] * (state.H[k] * xi_delta[k]);
        }

        // Scatter Pass (Base to Tip)
        for (int k = n; k >= 1; --k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k]);
            Matrix6d cRp = pRc.transpose();
            
            lambda_list[k] = state.tau_bar[k].transpose() * cRp * state.RBT[k].transpose() * lambda_list[k+1] 
                           + state.H[k].transpose() * nu_delta[k];

            Eigen::Vector3d b_delta = nu_delta[k] - state.G[k].transpose() * cRp * state.RBT[k].transpose() * lambda_list[k+1];
            
            // Inject the correction directly into the state's accelerations
            state.beta_dot[k] += b_delta; 
        }
    }

    // ---------------------------------------------------------
    // 2. Omega Diagonal Elements
    // Translates Python's 'get_omega_diag'
    // ---------------------------------------------------------
    inline std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> compute_omega_diag(const PendulumState& state) {
        int n = state.n;
        
        // Initialize with zeros. omega[n+1] automatically serves as the boundary condition.
        std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> omega(n + 2, Matrix6d::Zero());

        // Inward sweep (Base to Tip)
        for (int k = n; k >= 1; --k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k]);
            Matrix6d cRp = pRc.transpose();

            // Term 1: tau_bar^T * cRp * RBT^T * omega[k+1] * RBT * pRc * tau_bar
            Matrix6d term1 = state.tau_bar[k].transpose() * cRp * state.RBT[k].transpose() 
                             * omega[k+1] 
                             * state.RBT[k] * pRc * state.tau_bar[k];

            // Term 2: H^T * D_inv * H
            Matrix6d term2 = state.H[k].transpose() * state.D_inv[k] * state.H[k];

            omega[k] = term1 + term2;
        }

        return omega;
    }

    // ---------------------------------------------------------
    // 3. Omega Off-Diagonal Elements
    // Translates Python's 'get_omega_ij'
    // ---------------------------------------------------------
    inline Matrix6d compute_omega_ij(int i, int j, const PendulumState& state, 
                                     const std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>>& omega_diag) {
        // Base case: Diagonal entry
        if (i == j) {
            return omega_diag[i];
        }
        
        // Symmetry property: Omega_{i,j} = Omega_{j,i}^T
        if (i < j) {
            return compute_omega_ij(j, i, state, omega_diag).transpose();
        }

        // Propagate from body i-1 down to j
        Matrix6d current_omega = omega_diag[i];
        
        for (int k = i - 1; k >= j; --k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k]);
            Matrix6d cRp = pRc.transpose();
            
            // current_omega = cRp * current_omega * RBT * pRc * tau_bar
            current_omega = cRp * current_omega * state.RBT[k] * pRc * state.tau_bar[k];
        }
        
        return current_omega;
    }

} // namespace SOAConstraints