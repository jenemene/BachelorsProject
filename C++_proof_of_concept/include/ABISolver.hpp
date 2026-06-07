#pragma once
#include "PendulumState.hpp"
#include "SpatialMath.hpp"
#include <Eigen/Dense>

namespace ABISolver {

    inline void compute_ATBI(PendulumState& state) {
        int n = state.n;

        // --- Boundary Conditions ---
        // The "ceiling" (n+1) is rigidly fixed. It has zero spatial velocity.
        state.V[n+1] = Vector6d::Zero();
        
        // Pseudo-gravity: accelerating the base upward is equivalent to gravity pulling the chain down.
        // Assuming the Z-axis points up, so +9.81 on the linear Z component.
        state.A[n+1] << 0.0, 0.0, 0.0, 0.0, 0.0, 9.81;

        // Boundary condition for the articulated inertia sweep
        state.P_plus[0] = Matrix6d::Zero();
        state.xi_plus[0] = Vector6d::Zero();

        // --- 1. Kinematics Scatter (Base to Tip) ---
        // Outward sweep: n down to 1
        for (int k = n; k >= 1; --k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k]);
            Matrix6d cRp = pRc.transpose();

            Vector6d delta_V = state.H[k].transpose() * state.beta[k];

            state.V[k] = cRp * state.RBT[k].transpose() * state.V[k+1] + delta_V;

            state.agothic[k] = SpatialMath::spatial_cross(state.V[k]) * state.H[k].transpose() * state.beta[k];
            state.bgothic[k] = SpatialMath::spatial_cross_dual(state.V[k]) * state.M[k] * state.V[k];
        }

        // --- 2. ATBI Gather (Tip to Base) ---
        // Inward sweep: 1 up to n
        for (int k = 1; k <= n; ++k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k-1]); 
            Matrix6d cRp = pRc.transpose();

            Matrix6d P = state.RBT[k] * pRc * state.P_plus[k-1] * cRp * state.RBT[k].transpose() + state.M[k];
            
            Eigen::Matrix3d D = state.H[k] * P * state.H[k].transpose();
            state.D_inv[k] = D.inverse(); 

            state.G[k] = P * state.H[k].transpose() * state.D_inv[k]; 

            state.tau_bar[k] = Matrix6d::Identity() - state.G[k] * state.H[k];
            state.P_plus[k] = state.tau_bar[k] * P;

            Vector6d xi = state.RBT[k] * pRc * state.xi_plus[k-1] + P * state.agothic[k] + state.bgothic[k];

            Eigen::Vector3d eps = state.tau[k] - state.H[k] * xi;
            
            state.nu[k] = state.D_inv[k] * eps;
            state.xi_plus[k] = xi + state.G[k] * eps;
        }

        // --- 3. ATBI Scatter (Base to Tip) ---
        // Outward sweep: n down to 1
        for (int k = n; k >= 1; --k) {
            Matrix6d pRc = SpatialMath::spatialrotfromquat(state.theta[k]);
            Matrix6d cRp = pRc.transpose();

            Vector6d A_plus = cRp * state.RBT[k].transpose() * state.A[k+1];

            state.beta_dot[k] = state.nu[k] - state.G[k].transpose() * A_plus;

            state.A[k] = A_plus + state.H[k].transpose() * state.beta_dot[k] + state.agothic[k];
        }
    }

} // namespace ABISolver