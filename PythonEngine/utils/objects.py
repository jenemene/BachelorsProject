from matplotlib.animation import FuncAnimation
import numpy as np
from utils import soa as SOA
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time
import csv
from scipy.integrate import solve_ivp

class Joint:
    def __init__(self):
        self.nq = None #generalized coordinates
        self.nw = None #generalized velocities
        self.H = None #hingemap
        self.q_init = None # Needed to build the starting state
        self.w_init = None

    def get_derrivative(self,theta,beta):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_spatial_rotation(self,theta):
        raise NotImplementedError("This method should be implemented by subclasses.")

class SphericalJoint(Joint):
    def __init__(self):
        super().__init__()
        self.nq = 4 #quaternion
        self.nw = 3 #angular velocity as generalized velocity
        self.H = np.block([[np.eye(3), np.zeros((3,3))]])
        self.q_init = np.array([0.0, 0.0, 0.0, 1.0])
        self.w_init = np.zeros(3)

    def get_derrivative(self,theta,beta):
        return SOA.derrivmap(theta,beta,"spherical")
     
    def get_spatial_rotation(self,theta):
        return SOA.spatialrotfromquat(theta)
    
    def get_translation(self,theta):
        return np.zeros(3,)

class RevoluteJoint(Joint):
    def __init__(self,axis):
        super().__init__()
        self.nq = 1 #angle
        self.nw = 1 #angular velocity as generalized velocity
        self.axis = axis 
        self.q_init = np.array([0.0])
        self.w_init = np.array([0.0])
        
        if axis == "x": self.H = np.array([[1,0,0,0,0,0]])
        elif axis == "y": self.H = np.array([[0,1,0,0,0,0]])
        elif axis == "z": self.H = np.array([[0,0,1,0,0,0]])

    def get_derrivative(self,theta,beta):
        return beta
        
    def get_quaternion(self,theta):
        quat = SOA.quatfromrev(theta[0],self.axis)
        return quat
        
    def get_spatial_rotation(self,theta):
        quat = self.get_quaternion(theta)
        return SOA.spatialrotfromquat(quat)
    
    def get_translation(self,theta):
        return np.zeros(3,)

class FreeJoint(Joint):
    def __init__(self):
        super().__init__()
        self.nq = 7 #quaternion + position
        self.nw = 6 #angular velocity + linear velocity as generalized velocity
        self.H = np.eye(6) 
        self.q_init = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) #initially at origin with no rotation and translation
        self.w_init = np.zeros(6)

    def get_derrivative(self,theta,beta):
        theta_rot_dot = SOA.derrivmap(theta[:4],beta[:3],"spherical")
        rot = SOA.rotfromquat(theta[:4])
        theta_trans_dot = rot @ beta[3:]
        # dIl(I,n)/dt = Iv(I,n) = IRn nv(I,n). beta comes from integration of beta_dot, which is in body frame
        return np.concatenate([theta_rot_dot, theta_trans_dot])
    
    def get_spatial_rotation(self,theta):
        quat = theta[:4] #first 4 elements are quaternion
        return SOA.spatialrotfromquat(quat)
    
    def get_translation(self,theta):
        return theta[4:] #last 3 elements are translation

class Link:
    def __init__(self,mass,l_hinge,joint):
        self.m = mass
        self.l_com = l_hinge/2
        self.l_hinge = l_hinge
        self.joint = joint

        l = np.linalg.norm(l_hinge)
        w = l/50 # Width and height are 1/50th of the length. This is an arbitrary choice to give the link some thickness without dominating the inertia.
        h = w
        self.J_c = np.diag([
        1/12 * self.m * (h**2 + l**2), 
        1/12 * self.m * (w**2 + l**2), 
        1/12 * self.m * (w**2 + h**2)
        ]) #nakket fra wikipedia.

        self.M_c =  np.block([[self.J_c, np.zeros((3,3))],
                          [np.zeros((3,3)), self.m*np.eye(3)]])
    
        self.M = SOA.RBT(self.l_com)@self.M_c@SOA.RBT(self.l_com).T 

        self.RBT = SOA.RBT(l_hinge)

class MultiBodySystem:
    def __init__(self):
        self.links = []
        self.total_nq = 0
        self.total_nw = 0
        self.result = None
        self.tspan = None
        self.constraint_violation = []
        self._record_metrics = False
        self.l_from_origin = np.array([0,0,0])

    def add_link(self,link):
        self.links.insert(0, link)
        self.total_nq += link.joint.nq
        self.total_nw += link.joint.nw
    
    def get_initial_state(self):
        q0_list = [link.joint.q_init for link in self.links]
        w0_list = [link.joint.w_init for link in self.links]
        return np.concatenate(q0_list + w0_list)

    def unpack_state(self,state):
        #for unpacking state vector into a list of theta and beta
        theta_list = []
        beta_list = []

        idx_theta = 0
        idx_beta = self.total_nq
        
        for link in self.links:
            theta = state[idx_theta: idx_theta+link.joint.nq]
            
             # Normalization safety check for quaternions
            if link.joint.nq == 4:
                 theta = theta / np.linalg.norm(theta)
            elif link.joint.nq == 7:
                 theta[:4] = theta[:4] / np.linalg.norm(theta[:4])
                
            theta_list.append(theta)
            idx_theta += link.joint.nq 

            beta = state[idx_beta:idx_beta + link.joint.nw]
            beta_list.append(beta)
            idx_beta += link.joint.nw
            
        return theta_list, beta_list

    def get_state_dot(self,t,state,V_base,A_base):
        theta_list, beta_list = self.unpack_state(state)

        theta_dot_list = []

        tau_list = [np.zeros(link.joint.nw) for link in self.links]

        for i in range(len(self.links)): #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION! APPEND IS NOT EFFICIENT
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i])
            theta_dot_list.append(theta_dot)

        beta_dot_list,V,_,_,_,_ = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        state_dot = np.concatenate(theta_dot_list + beta_dot_list)
        return state_dot, V

    def get_state_dot_closed(self,t,state,V_base,A_base,BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        tau_list = [np.zeros(link.joint.nw) for link in self.links]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #ROTATIONS AND CONSTRAINT SETUPS
        link1 = self.links[0]
        linkn = self.links[-1]

        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])

        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d, -d])

        #Calculaition of LAMBDA

        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        omega_nn = omega_diag[n]
        omega_11 = omega_diag[1]
        

        omega_n1 = self.get_omega_ij(n,1,theta_list,tau_bar,omega_diag,n)

        Λ_11 = IR1 @ (link1.RBT.T @ omega_11 @ link1.RBT) @IR1.T
        Λ_nn = IRn @ (omega_nn @ IRn.T)
        Λ_n1 = IR1 @ (omega_n1 @ link1.RBT) @ IR1.T

        
    
        Λ_block = np.block([
            [Λ_11, Λ_n1.T],
            [Λ_n1, Λ_nn]
        ])

        V_tip = IR1@link1.RBT.T@V_f[1]
        v_tip  = V_tip[3:]
        v_base = IRn[:3, :3]@V_f[n][3:]

        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)

        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  -(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        Φ_dot =   v_tip - v_base
        Φ_ddot =  -(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        # Baumgarte stabilization
        α, β = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β)

        
        λ = -np.linalg.solve((Q @ Λ_block @ Q.T), f)
        

        #calculating f_c
        f_c_closed_loop_const = -Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #constraints and Q are ordered [tip, base]

        f_c[1] = link1.RBT @ IR1.T @ f_c_closed_loop_const[:6] 
        f_c[n] = IRn.T @ f_c_closed_loop_const[6:]

        if self._record_metrics:
            self.constraint_violation.append(np.linalg.norm(Φ))

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def get_state_dot_sprockets(self,t,state,V_base,A_base,BG_params,Penalty_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forced can be used to simulate damping
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]
    

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)


        #positons and rotations. Needed for constraints and penalty
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        IR_list = self.get_all_rotations_body_to_I(theta_list)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC) - WITH PENALTY!
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI_penalty(t,theta_list,beta_list,tau_list,V_base,A_base,positions,IR_list,Penalty_params=Penalty_params)

        #ROTATIONS AND CONSTRAINT SETUPS
        link1 = self.links[0]
        linkn = self.links[-1]

        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])

        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d, -d])


        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        omega_nn = omega_diag[n]
        omega_11 = omega_diag[1]
        
        omega_n1 = self.get_omega_ij(n,1,theta_list,tau_bar,omega_diag,n)

        Λ_11 = IR1 @ (link1.RBT.T @ omega_11 @ link1.RBT) @IR1.T
        Λ_nn = IRn @ (omega_nn @ IRn.T)
        Λ_n1 = IR1 @ (omega_n1 @ link1.RBT) @ IR1.T

        
    
        Λ_block = np.block([
            [Λ_11, Λ_n1.T],
            [Λ_n1, Λ_nn]
        ])

        V_tip = IR1@link1.RBT.T@V_f[1]
        v_tip  = V_tip[3:]
        v_base = IRn[:3, :3]@V_f[n][3:]


        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  -(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        Φ_dot =   v_tip - v_base
        Φ_ddot =  -(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        # Baumgarte stabilization
        alpha, beta = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, alpha, beta)


        λ = -np.linalg.solve((Q @ Λ_block @ Q.T), f)
        #calculating f_c
        f_c_closed_loop_const = -Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #constraints and Q are ordered [tip, base]

        f_c[1] = link1.RBT @ IR1.T @ f_c_closed_loop_const[:6] 
        f_c[n] = IRn.T @ f_c_closed_loop_const[6:]

        #for measuring constraint violation
        if self._record_metrics:
            self.constraint_violation.append(np.linalg.norm(Φ))

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def get_state_dot_multiple_constraints(self,t,state,V_base,A_base,BG_params):

        #to test multiple constraints a n=3 body pendulum will be implemented. The two constraints are
            #1. Closed loop constraint between 1-3
            #2. A driving constraint on k=2
                #important note: This is not efficient at all.
                #The resulting lambda matrix will be 6*n_c X 6_n_c - that is it will be 18x18. Similarly, we will have to stack Q matrices and Lambda matrices. Q will be a 6x18 matrix.
        

        #unpacking state and getting
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #compute positions of all 3 links
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        l_IO1 = positions[1]
        l_IO2 = positions[2]
        l_IOn = positions[n]
        

        #compute needed rotations
        #rotations. To keep general for now, we simply use notation that n=3. 
        linkn = self.links[-1]
        link1 = self.links[0]
        link2 = self.links[1]
        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IR2 = SOA.get_rotation_body_to_I(theta_list,self.links,n,2)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])
        

        #constraint 1 setup - Closed Loop between 1 and 3
        Q_closed = np.block([np.zeros((3,3)), np.eye(3),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),-np.eye(3)]) #how to set this up check my paper handwriting. I think it could be a good idea to draw this in as an example.

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])


        Φ_closed =  -1*(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        Φ_closed_dot = -1*(IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge))
        Φ_closed_ddot =  -1*(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        #constraint 2 setup - Driver on link 2. in x-z plane

        Q_driver = np.block([np.zeros((3,3)), np.zeros((3,3)),np.zeros((3,3)),np.eye(3),np.zeros((3,3)),np.zeros((3,3))])

        IR2_3 = IR2[:3,:3]

        omega = 2*np.pi/5
        length = np.linalg.norm(l_IO2 - l_IO1)

        l_driver = np.array([length*np.cos(omega*t),0,length*np.sin(omega*t)])
        l_driver_dot = np.array([-omega*length*np.sin(omega*t),0,omega*length*np.cos(omega*t)])
        l_driver_ddot = np.array([-omega**2*length*np.cos(omega*t),0,-omega**2*length*np.sin(omega*t)])

        Φ_driver = l_IO2  - l_driver
        Φ_driver_dot = IR2_3@V_f[2][3:] - l_driver_dot
        Φ_driver_ddot = IR2_3@A_f[2][3:]- l_driver_ddot

        #Calculating operational spatial space compliance entires. Needed are Ω(1,1), Ω(2,2), Ω(3,3), Ω(2,1), Ω(3,1), Ω(3,2). 
        #In this computation, we actually end up building the full Ω matrix using the omega function - this is not generally needed when more links are added, as the non-needed entries are then simply not calculated :)

        #calculation of Ω(1,1), Ω(2,2), Ω(3,3)
        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        Ω_11 = omega_diag[1]
        Ω_22 = omega_diag[2]
        Ω_33 = omega_diag[n]

        #calculation of off diagonal terms 
        Ω_21 = self.get_omega_ij(2, 1, theta_list, tau_bar, omega_diag,n)
        Ω_31 = self.get_omega_ij(3, 1, theta_list, tau_bar, omega_diag,n)
        Ω_32 = self.get_omega_ij(3, 2, theta_list, tau_bar, omega_diag,n) #31 gets calculated to compute 32, so this is the part that would need optimiziation

        #Building Lambda Matrix. Block entires are calculated
        #constraint 1 - closed loop
        Λ_11 = IR1 @ (link1.RBT.T @ Ω_11 @ link1.RBT) @IR1.T
        Λ_33 = IRn @ (Ω_33 @ IRn.T)
        Λ_31 = IR1 @ (Ω_31 @ link1.RBT) @ IR1.T
        Λ_13 = IR1 @ (link1.RBT.T @ Ω_31.T) @ IR1.T

        #constraint 2 - driver. Driving the base of the link here
        Λ_22 = IR2 @ (Ω_22 @ IR2.T)

        #cross couplings
        Λ_21 = IR1 @ (Ω_21 @ link1.RBT) @ IR1.T
        Λ_12 = IR1 @ (link1.RBT.T @ Ω_21.T) @ IR1.T
        Λ_32 = IR2 @ (Ω_32 @ IR2.T)
        Λ_23 = IR2 @ (Ω_32.T @ IR2.T)

        #assembling system quantities

        Λ_sys = np.block([
            [Λ_11, Λ_12, Λ_13],
            [Λ_21, Λ_22, Λ_23],
            [Λ_31, Λ_32, Λ_33]
        ])

        Q_sys = np.block([
            [Q_closed],
            [Q_driver]
        ])
        
        Φ_system = np.concatenate([Φ_closed, Φ_driver])

        Φ_dot_system = np.concatenate([Φ_closed_dot, Φ_driver_dot])
        Φ_ddot_system = np.concatenate([Φ_closed_ddot, Φ_driver_ddot])

        #Baumgarte stabilization    

        α, β = BG_params
        Φ_BG = SOA.baumgarte_stab(Φ_system, Φ_dot_system, Φ_ddot_system, α, β)

        
        #solving for lagrange multipliers
        M_eff = Q_sys @ Λ_sys @ Q_sys.T
        λ = -np.linalg.solve(M_eff, Φ_BG)
    

        #calculating f_c
        f_const = -Q_sys.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #constraints and Q are ordered [tip, base]
        f_c[1] = link1.RBT @ IR1.T @ f_const[:6]
        f_c[2] = IR2.T @ f_const[6:12]
        f_c[n] = IRn.T @ f_const[12:]

        if self._record_metrics:
            self.constraint_violation.append(np.linalg.norm(Φ_system))

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def run_ATBI(self,theta_list,beta_list,tau_list,V_base,A_base):
        n = len(self.links)

        theta = [None]*(n+2)
        beta  = [None]*(n+2)
        tau   = [None]*(n+2)
        links = [None]*(n+2) 

        for i in range(1, n+1):
            theta[i] = theta_list[i-1]
            beta[i]  = beta_list[i-1]
            tau[i]   = tau_list[i-1]
            links[i] = self.links[i-1]

        theta[0]   = np.zeros_like(theta[1])
        theta[n+1] = np.zeros_like(theta[n])
        beta[0]    = np.zeros_like(beta[1])
        beta[n+1]  = np.zeros_like(beta[n])
        tau[0]     = np.zeros_like(tau[1])
        tau[n+1]   = np.zeros_like(tau[n])

        self.l_from_origin = links[n].joint.get_translation(theta[n]) #returns [0,0,0] for anything other than FreeJoint()

        P_plus, xi_plus, nu, A, V, G, D, beta_dot, tau_bar, agothic, bgothic,pRc_cache = \
            [([None]*(n+2)) for _ in range(12)] 
            
        #boundary conditions
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = np.zeros((6,6))
            
        A[n+1] = A_base
        V[n+1] = V_base
    
        # --- ATBI scatter ---- 
        for k in range(n, 0, -1):
            if k == n: #edge case detection for free joints
                RBT = SOA.RBT(self.l_from_origin) #l(k+1,k) as we need phi(k+1,k)
            else:
                RBT = links[k+1].RBT

            pRc = links[k].joint.get_spatial_rotation(theta[k]) 
            pRc_cache[k] = pRc
            cRp = pRc.T 

            delta_V_k = links[k].joint.H.T @ beta[k]
            V[k] = cRp @ RBT.T @ V[k+1] + delta_V_k 

            agothic[k] = SOA.spatialskewtilde(V[k]) @ delta_V_k 
            #latter part commented out after talks with Jain

            bgothic[k] = SOA.spatialskewbar(V[k]) @ links[k].M @ V[k]

        # --- ATBI GATHER ---
        for k in range(1, n+1): 
            if k == 1:
                pRc = np.eye(6)
                cRp = pRc.T
            else:
                pRc = pRc_cache[k-1]
                cRp = pRc.T 

            P = links[k].RBT @ pRc @ P_plus[k-1] @ cRp @ links[k].RBT.T + links[k].M
            D[k] = links[k].joint.H @ P @ links[k].joint.H.T
            G[k] = np.linalg.solve(D[k], links[k].joint.H @ P).T 
            tau_bar[k] = np.eye(6) - G[k] @ links[k].joint.H
            P_plus[k] = tau_bar[k] @ P
            xi = links[k].RBT @ pRc @ xi_plus[k-1] + P @ agothic[k] + bgothic[k]
                    
            eps = tau[k] - links[k].joint.H @ xi
            nu[k] = np.linalg.solve(D[k], eps) 
            xi_plus[k] = xi + G[k] @ eps

        # --- 4. ATBI SCATTER ---
        for k in range(n, 0, -1):
            if k == n: #boundary condition on n. This is to model free joint if needed.
                RBT = SOA.RBT(self.l_from_origin )
            else:
                RBT = links[k+1].RBT

            pRc = pRc_cache[k]
            cRp = pRc.T 

            A_plus = cRp @ RBT.T @ A[k+1]
            beta_dot[k] = nu[k] - G[k].T @ A_plus
            A[k] = A_plus + links[k].joint.H.T @ beta_dot[k] + agothic[k]
        return beta_dot[1:n+1], V, A, tau_bar, D, G
 
   
    def run_ATBI_penalty(self,t,theta_list,beta_list,tau_list,V_base,A_base,positions,IR_list,Penalty_params):
        #nice to have = sprockets as input
        n = len(self.links)

        theta = [None]*(n+2)
        beta  = [None]*(n+2)
        tau   = [None]*(n+2)
        links = [None]*(n+2) 

        for i in range(1, n+1):
            theta[i] = theta_list[i-1]
            beta[i]  = beta_list[i-1]
            tau[i]   = tau_list[i-1]
            links[i] = self.links[i-1]

        theta[0]   = np.zeros_like(theta[1])
        theta[n+1] = np.zeros_like(theta[n])
        beta[0]    = np.zeros_like(beta[1])
        beta[n+1]  = np.zeros_like(beta[n])
        tau[0]     = np.zeros_like(tau[1])
        tau[n+1]   = np.zeros_like(tau[n])


        P_plus, xi_plus, nu, A, V, G, D, beta_dot, tau_bar, agothic, bgothic,pRc_cache = \
            [([None]*(n+2)) for _ in range(12)] 
            
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = np.zeros((6,6))
            
        A[n+1] = A_base
        V[n+1] = V_base
    
        # --- ATBI scatter (Kinematics) ---- 
        for k in range(n, 0, -1):
            if k == n:
                RBT = SOA.RBT(self.l_from_origin)
            else:
                RBT = links[k+1].RBT

            pRc = links[k].joint.get_spatial_rotation(theta[k]) 
            pRc_cache[k] = pRc
            cRp = pRc.T 

            delta_V_k = links[k].joint.H.T @ beta[k]
            V[k] = cRp @ RBT.T @ V[k+1] + delta_V_k

            agothic[k] = SOA.spatialskewtilde(V[k]) @ delta_V_k
            bgothic[k] = SOA.spatialskewbar(V[k]) @ links[k].M @ V[k]
        
        # --- PENALTY DETECTION --- #
        
        #intializing external force array. This could contain any external forces, but for now, its purely used for the forces coming from sprockets
        f_ext_body = [np.zeros(6,) for _ in range(n+2)]

        # Unpack stiffness and damping for penalty method
        k_stiffness = Penalty_params[0]
        c_damping = Penalty_params[1]

        # ---- 5. GEOMETRY ----
        sprockets = [
             {'center': np.array([-0.6, 0.0, 0.0]), 'radius': 0.3864}, # Left Sprocket
             {'center': np.array([0.6, 0.0, 0.0]), 'radius': 0.3864}  # Right Sprocket
         ]
        
        for k in range(1,n+1):
            IR_k = IR_list[k]  
            IR_k_3 = IR_k[:3, :3]
            pos = positions[k] #get current position of base of link k
            base_vel = IR_k_3 @ V[k][3:] #get current velocity

            #loop over sprockets. Right now there are two, but more can be added, thus a for loop is implemented
            for sprocket in sprockets:
                #calculating vector from sprocket center to current location aswell as distance
                vec_from_sprocket_center = pos - sprocket['center']
                distance = np.linalg.norm(vec_from_sprocket_center)

                # distance between body and spocket radius. If this is negative, then we have penetration
                d = distance - sprocket['radius']

                if d < 0: # Penetration into the sprocket (also a little cheating on the driving)

                    #geometry
                    normal_vec = vec_from_sprocket_center / distance #normal vec - this is based on where the link is at the time, and NOT where it was during penetration
                    #there is an argument for this being slightly inaccurate, but with a small enough dt the discreptency is expected to be rather small.
                    
                    tangent_vec = np.array([-normal_vec[2], 0.0, normal_vec[0]])
                    d_dot = np.dot(normal_vec, base_vel)
                    v_tangent = np.dot(tangent_vec,base_vel)
                    
                    if t>=5 and sprocket['center'][0]>0: #and statement is to ensure right sprocket is driving
                        v_target = 2.0   # Target tangential velocity in m/s
                        K_drive = 100.0  # Proportional gain (acts like the steepness of a motor's torque curve)
                        
                        F_drive_mag = K_drive * (v_target - v_tangent)
                        
                        # Prevent active braking if the chain is somehow pushed faster than target speed
                        if F_drive_mag < 0.0:
                            F_drive_mag = 0.0
                    else: 
                        F_drive_mag = 0.0
                    
                    # Calculate pure spring compliant force and damping, pushing outward
                    F_normal_mag = -k_stiffness * d - c_damping * d_dot
                    if F_normal_mag < 0:
                        F_normal_mag = 0.0

                
                    # normal force
                    F_sprocket_3_out = F_normal_mag * normal_vec
                    #driving force
                    F_sprocket_3_drive = F_drive_mag*tangent_vec
                    # Transform force to body frame
                    F_body = IR_k_3.T @ (F_sprocket_3_out + F_sprocket_3_drive)

                    # add to body k. This also in theory should handle the case that more than 1 sprocket is hit.
                    f_ext_body[k][3:] += F_body


        # --- ATBI GATHER --- Now with external forces 
        for k in range(1, n+1): 
            if k == 1:
                pRc = np.eye(6)
                cRp = pRc.T
            else:
                pRc = pRc_cache[k-1]
                cRp = pRc.T 

            P = links[k].RBT @ pRc @ P_plus[k-1] @ cRp @ links[k].RBT.T + links[k].M
            D[k] = links[k].joint.H @ P @ links[k].joint.H.T
            G[k] = np.linalg.solve(D[k], links[k].joint.H @ P).T 
            tau_bar[k] = np.eye(6) - G[k] @ links[k].joint.H
            P_plus[k] = tau_bar[k] @ P
            xi = links[k].RBT @ pRc @ xi_plus[k-1] + P @ agothic[k] + bgothic[k] - f_ext_body[k] #spatial penalty forces added to xi
                    
            eps = tau[k] - links[k].joint.H @ xi
            nu[k] = np.linalg.solve(D[k], eps) 
            xi_plus[k] = xi + G[k] @ eps

        # --- 4. ATBI SCATTER ---
        for k in range(n, 0, -1):
            if k == n: #boundary condition on n. This is to model free joint if needed.
                RBT = SOA.RBT(self.l_from_origin )
            else:
                RBT = links[k+1].RBT

            pRc = pRc_cache[k]
            cRp = pRc.T 

            A_plus = cRp @ RBT.T @ A[k+1]
            beta_dot[k] = nu[k] - G[k].T @ A_plus
            A[k] = A_plus + links[k].joint.H.T @ beta_dot[k] + agothic[k]
        return beta_dot[1:n+1], V, A, tau_bar, D, G
     
    def simulate(self, tspan, V_base, A_base, config="open", BG_params=None,Penalty_params=None):
        print(f"Simulation started ({config}-loop configuration)")
        start_time = time.perf_counter()

        # Initial configuration
        state0 = self.get_initial_state()
        dt = tspan[1] - tspan[0]
        
        nt = len(tspan)
        nq = len(state0)

        Y = np.zeros((nq, nt))
        Y[:, 0] = state0
        self.V = [None]*nt #to be able to save spatial velocities
        self.beta_dot = [None]*nt
        self.constraint_violation = [] # Clear previous violations
        # Dynamically route the derivative calculation based on config
        def ODEfun(t, state, V_base, A_base):
            if config == "closed":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for closed-loop simulation.")
                return self.get_state_dot_closed(t, state, V_base, A_base, BG_params)
            elif config == "open":
                return self.get_state_dot(t, state, V_base, A_base)
            elif config == "multiple_constraints":
                return self.get_state_dot_multiple_constraints(t, state, V_base, A_base, BG_params)
            elif config == "sprockets":
                if Penalty_params is None: # 
                    raise ValueError("penalty_params (k, c) must be provided.")
                if BG_params is None:
                    raise ValueError("BG_params must be provided for sprockets simulation.")
                return self.get_state_dot_sprockets(t, state, V_base, A_base, BG_params, Penalty_params)
            else:
                raise ValueError("Invalid config. Choose 'open', 'closed' or 'driver'.")
        
        # RK4 integration loop
        for i in range(nt-1):
            t = tspan[i]
            y = Y[:, i]

            self._record_metrics = True
            k1, V_val = ODEfun(t, y, V_base, A_base)
            self._record_metrics = False
            self.V[i] = V_val
            self.beta_dot[i] = k1[self.total_nq:]

            k2, _  = ODEfun(t + dt/2, y + dt/2 * k1, V_base, A_base)
            k3, _  = ODEfun(t + dt/2.0, y + dt/2.0 * k2, V_base, A_base)
            k4, _  = ODEfun(t + dt, y + dt * k3, V_base, A_base)

            Y[:, i+1] = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)


            
        # Calc last V entry
        self._record_metrics = True
        state_dot_last, V_last = ODEfun(tspan[-1], Y[:,-1], V_base, A_base)
        self._record_metrics = False
        self.V[-1] = V_last
        self.beta_dot[-1] = state_dot_last[self.total_nq:]

        self.result = Y
        self.tspan = tspan
        end_time = time.perf_counter()
        elapesed_time = end_time - start_time
        print(f"Simulation finished. Runtime: {elapesed_time:.2f} s")
    
    def simulate_own_RK4(self, tspan, V_base, A_base, config="open", BG_params=None):
        # print(f"Simulation started ({config}-loop configuration)")
        # start_time = time.perf_counter()
        #this is used for the order_n-val_scripts
        # Initial configuration
        state0 = self.get_initial_state()
        dt = tspan[1] - tspan[0]
        
        nt = len(tspan)
        nq = len(state0)

        Y = np.zeros((nq, nt))
        Y[:, 0] = state0
        self.V = [None]*nt #to be able to save spatial velocities
        self.beta_dot = [None]*nt

        # Dynamically route the derivative calculation based on config
        def ODEfun(t, state, V_base, A_base):
            if config == "closed":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for closed-loop simulation.")
                return self.get_state_dot_closed(t, state, V_base, A_base, BG_params)
            elif config == "open":
                return self.get_state_dot(t, state, V_base, A_base)
            else:
                raise ValueError("Invalid config. Choose 'open', 'closed' or 'driver'.")
        
        # RK4 integration loop
        for i in range(nt-1):
            t = tspan[i]
            y = Y[:, i]

            k1, V_val  = ODEfun(t, y, V_base, A_base)
            self.V[i] = V_val
            self.beta_dot[i] = k1[self.total_nq:]

            k2,_  = ODEfun(t + dt/2, y + dt/2 * k1, V_base, A_base)
            k3,_  = ODEfun(t + dt/2.0, y + dt/2.0 * k2, V_base, A_base)
            k4,_  = ODEfun(t + dt, y + dt * k3, V_base, A_base)

            Y[:, i+1] = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

            # Robust way to print every 1 second of simulation time
            # if t % 1 < dt: 
            #     print(f"t = {t:.2f} s")

        # Calc last V entry
        state_dot_last, V_last = ODEfun(tspan[-1], Y[:,-1], V_base, A_base)
        self.V[-1] = V_last
        self.beta_dot[-1] = state_dot_last[self.total_nq:]

        self.result = Y
        self.tspan = tspan
        # end_time = time.perf_counter()
        # elapesed_time = end_time - start_time
        # print(f"Simulation finished. Runtime: {elapesed_time:.2f} s")

    def simulate_solve_ivp(self, tspan, V_base, A_base, config="open", BG_params=None):
        # print(f"Simulation started ({config}-loop configuration) with solve_ivp (RK45)")
        # start_time = time.perf_counter()

        # Initial configuration
        state0 = self.get_initial_state()
        
        nt = len(tspan)

        # Dynamically route the derivative calculation based on config
        def ODEfun(t, state, V_base, A_base):
            if config == "closed":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for closed-loop simulation.")
                return self.get_state_dot_closed(t, state, V_base, A_base, BG_params)
            elif config == "open":
                return self.get_state_dot(t, state, V_base, A_base)
            else:
                raise ValueError("Invalid config. Choose 'open', 'closed' or 'driver'.")
        
        def fun(t, state):
            res, _ = ODEfun(t, state, V_base, A_base)
            if isinstance(res, tuple):
                return res[0]
            return res

        sol = solve_ivp(fun, [tspan[0], tspan[-1]], state0, method='RK45', t_eval=tspan, max_step=tspan[1]-tspan[0])
        
        if not sol.success:
            print(f"Integration stopped early: {sol.message}")
            self.tspan = sol.t
        else:
            self.tspan = tspan

        self.result = sol.y

        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print(f"Simulation finished. Runtime: {elapsed_time:.2f} s")

    def get_omega_diag(self,theta_list,tau_bar,D,n):
                #storage
        gamma = [None]*(n+2)
        omega = [None]*(n+2)
        theta = [None]*(n+2)

        #theta_list is on a 0-index basis, for convenience this is shifted. This is not effective in time, but for now is ok

        for i in range(1,n+1):
            theta[i] = theta_list[i-1]

        #boundary condition on omega
        gamma[n+1] = np.zeros((6,6))

        for k in range (n,0,-1):
            link_k = self.links[k-1] #remember, links is on a 0-index

            #rotations
            pRc = link_k.joint.get_spatial_rotation(theta[k])
            cRp = pRc.T

            #calculating diagonal entries of omega

            gamma[k] = tau_bar[k].T @ cRp @ link_k.RBT.T @ gamma[k+1] @ link_k.RBT @ pRc @ tau_bar[k] + link_k.joint.H.T @ np.linalg.solve(D[k],link_k.joint.H)

        #renaminmg for readability
        omega_diag = gamma

        return omega_diag

    def get_omega_ij(self, i, j, theta_list, tau_bar, omega_diag,n):
        #calculates off diagonal entries not the MOST efficent as this may recalculate some entires, so essentially we are making more function calls than nessecarry. 
        #THIS IS NOT ORDER N FOR ANYTHING MORE THAN A SINGULAR CONSTRAINT! - IN THAT CASE, MAKE ANOTHER FUNCTION THAT RETURNS AND ENTIRE LIST
        if i == j:
            return omega_diag[i]
        
        if i < j:
            # Omega is symmetric: Omega_{i, j} = Omega_{j, i}^T
            return self.get_omega_ij(j, i, theta_list, tau_bar, omega_diag,n).T
            
        current_omega = omega_diag[i]
        
        # Shift theta to 1-based indexing for convenience
        theta = [None]*(len(self.links)+2)
        #this 1 based indexing shift becomes a scaling issue if the function is called more than one time. If called for all off diagonal entries, it actually multiplies the complexity by O(n). 
        #as this is not the case, there is no need to optimize, because this is readable as is.
        for idx in range(1,n+1):
            theta[idx] = theta_list[idx-1] 
            
        # Propagate from body i-1 down to j
        for k in range(i-1, j-1, -1):
            link_k = self.links[k-1]
            pRc = link_k.joint.get_spatial_rotation(theta[k])
            cRp = pRc.T
            current_omega = cRp @ current_omega @ link_k.RBT @ pRc @ tau_bar[k]
        #det den roterer lever i frame j    
        return current_omega

    def beta_dot_delta(self,theta_list,tau_bar,D,f_c,G,n):
        #shifting indexing for convience (same method as in run_ATBI)
        n = len(self.links) #no of bodies

        theta = [None]*(n+2)
        links = [None]*(n+2) 
        
        for i in range(1, n+1):
            theta[i] = theta_list[i-1]
            links[i] = self.links[i-1]

        theta[0]   = np.zeros_like(theta[1])
        theta[n+1] = np.zeros_like(theta[n])
        links[0] = links[1] # For initialization, but doesn't matter, bc tau_bar is all zeros

        #storage
        xi_delta = [None]*(n+2)
        beta_dot_delta = [None] * (n+2)
        nu = [None]*(n+2)
        lambda_list = [None]*(n+2)

        #boundary conditions f xi_delta and lambda_list
        xi_delta[0] = np.zeros(6,)
        lambda_list[n+1] = np.zeros(6,)

        #gather pass
        for k in range(1,n+1):
            #rotations
            pRc = links[k-1].joint.get_spatial_rotation(theta[k-1])
            cRp = pRc.T

            xi_delta[k] = links[k].RBT @ pRc @ tau_bar[k-1] @ xi_delta[k-1] - f_c[k]
            
            nu[k] = np.linalg.solve(D[k],links[k].joint.H @ xi_delta[k])

        #scatter pass
        for k in range(n,0,-1):
            #rotations
            pRc = links[k].joint.get_spatial_rotation(theta[k])
            cRp = pRc.T
            
            lambda_list[k] = tau_bar[k].T @ cRp @ links[k].RBT.T @ lambda_list[k+1]+links[k].joint.H.T@nu[k]

            beta_dot_delta[k] = nu[k] - G[k].T@cRp@links[k].RBT.T@lambda_list[k+1]
        
        return beta_dot_delta[1:n+1] #returning on 0 based indexing so it mathes

    def plot_gen_velocities(self, savefig=False):
        n = len(self.links)
        idx_w = self.total_nq # Index where velocities start

        # Create figure for subplots - added gridspec_kw to leave space at the bottom for the legend
        fig, axes = plt.subplots(
            n, 1, 
            figsize=(8, 2 * n + 0.6), # Slightly taller figure to accommodate bottom legend
            sharex=True, 
            layout="constrained"
        )

        if n == 1:
            axes = [axes] 

        # define colors
        colors = ['#B22222', "#336933", '#000080']

        for k in range(n):
            link = self.links[k]
            nw = link.joint.nw
            beta_k = self.result[idx_w:idx_w + nw, :] 
            tspan = self.tspan 

            ax_left = axes[k]

            # --- 6-DOF FREE JOINT ---
            if nw == 6: 
                ax_right = ax_left.twinx() # Create the independent right-hand axis                

                # Angular (Left Axis, Solid Lines)
                ax_left.plot(tspan, beta_k[0, :], color=colors[0], linestyle='-', label=r'$\beta_{\omega_x}$')
                ax_left.plot(tspan, beta_k[1, :], color=colors[1], linestyle='-', label=r'$\beta_{\omega_y}$')
                ax_left.plot(tspan, beta_k[2, :], color=colors[2], linestyle='-', label=r'$\beta_{\omega_z}$')

                # Linear (Right Axis, Dashed Lines)
                
                ax_right.plot(tspan, beta_k[3, :], color=colors[0], linestyle='--', label=r'$\beta_{v_x}$')
                ax_right.plot(tspan, beta_k[4, :], color=colors[1], linestyle='--', label=r'$\beta_{v_y}$')
                ax_right.plot(tspan, beta_k[5, :], color=colors[2], linestyle='--', label=r'$\beta_{v_z}$')

                # Independent Labels
                ax_left.set_ylabel(f'Body {k+1} Ang\n[rad/s]', fontweight='bold')
                ax_right.set_ylabel(f'Body {k+1} Lin\n[m/s]', fontweight='bold', rotation=270, labelpad=15)

                # --- ALIGN ZEROS OF TWIN Y-AXES ---
                # Force Matplotlib to calculate autoscale limits first
                ax_left.relim()
                ax_left.autoscale_view()
                ax_right.relim()
                ax_right.autoscale_view()

                # Get current limits
                l_min, l_max = ax_left.get_ylim()
                r_min, r_max = ax_right.get_ylim()

                # Find the scaling factor to match zero positions
                # We balance the maximum proportion of the positive vs negative bounds
                l_ratio = max(abs(l_min), 1e-6) / max(abs(l_max), 1e-6)
                r_ratio = max(abs(r_min), 1e-6) / max(abs(r_max), 1e-6)

                if l_ratio > r_ratio:
                    # Adjust right axis limits to match left axis ratio
                    if r_max > 0:
                        r_min = -r_max * l_ratio
                    else:
                        r_max = -r_min / l_ratio
                else:
                    # Adjust left axis limits to match right axis ratio
                    if l_max > 0:
                        l_min = -l_max * r_ratio
                    else:
                        l_max = -l_min / r_ratio

                ax_left.set_ylim(l_min, l_max)
                ax_right.set_ylim(r_min, r_max)

            # --- 3-DOF SPHERICAL JOINT ---
            elif nw == 3: 
                ax_left.plot(tspan, beta_k[0, :], color=colors[0], label=r'$\beta_{\omega_x}$')
                ax_left.plot(tspan, beta_k[1, :], color=colors[1], label=r'$\beta_{\omega_y}$')
                ax_left.plot(tspan, beta_k[2, :], color=colors[2], label=r'$\beta_{\omega_z}$')
                
                ax_left.set_ylabel(f'Body {k+1}\n[rad/s]')

            # --- 1-DOF REVOLUTE JOINT ---
            elif nw == 1: 
                if self.links[k].joint.axis == "x":
                    ax_left.plot(tspan, beta_k[0, :], color=colors[0], label=r'$\beta_{\omega_x}$')
                elif self.links[k].joint.axis == "y":
                    ax_left.plot(tspan, beta_k[0, :], color=colors[1], label=r'$\beta_{\omega_y}$')
                elif self.links[k].joint.axis == "z":
                    ax_left.plot(tspan, beta_k[0, :], color=colors[2], label=r'$\beta_{\omega_z}$')

                ax_left.set_ylabel(f'Body {k+1}\n[rad/s]')

            # --- SCIENTIFIC NOTATION FORMATTING ---
            # This triggers scientific notation for numbers smaller than 10^(-2) or larger than 10^3
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3)) 
            ax_left.yaxis.set_major_formatter(formatter)

            # Update index to start at the next body's data
            idx_w += nw

            # Shared grid settings
            ax_left.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time [s]')

        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=12)

        fig.align_ylabels(axes)

        # --- GLOBAL LEGEND GENERATION ---
        # Gather all line handles and labels from every axis in the figure
        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Filter out duplicate labels (so beta_x, v_x etc. only appear once)
        by_label = dict(zip(labels, handles))

        # Place the unique legend at the bottom center of the whole figure
        fig.legend(
            by_label.values(), 
            by_label.keys(), 
            loc='outside lower center', # Places it cleanly beneath the subplots
            ncol=len(by_label),         # Forces everything onto a horizontal row
            fontsize=12, 
            frameon=True, 
            framealpha=0.9
        )

        if savefig == True:
            plt.savefig("gen_velocities.pdf")
            print("Figure saved as gen_velocities.pdf in current directory.")
                
        plt.show()

    def animation(self, config="openclosed", step=1):
        assert self.result is not None, "No simulation result found. Please run simulation before calling animation()."

        ani_tspan = self.tspan[::step]
        ani_states = self.result[:, ::step]

        # Number of bodies and time steps
        n = len(self.links)
        N = ani_states.shape[1]

        # Setting up the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Defining plotlim based on the number of bodies, link length and configuration. This is just such that a closed loop configuration isnt too zoomed out
        total_link_length = sum(np.linalg.norm(link.l_hinge) for link in self.links)
        if config == "open":
            plotlim = total_link_length + np.linalg.norm(self.links[0].l_hinge)
        elif config == "closed":
            plotlim = total_link_length/2 + np.linalg.norm(self.links[0].l_hinge)
        else:
            raise ValueError("Invalid config value. Use 'open' or 'closed'.")
        
        # Set plot limits and labels
        ax.set_xlim([-plotlim, plotlim])
        ax.set_ylim([-plotlim, plotlim])
        ax.set_zlim([-plotlim, plotlim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set(box_aspect=(1, 1, 1))

        # Initialize the line object that will be updated in the animation
        line, = ax.plot([], [], [], 'o-', lw=2)

        def compute_positions(state_k):
            theta_list ,_ = self.unpack_state(state_k)
            positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
            
            # positions is constructed as a list of arrays, where the index 0 is empty. We can conveniently insert the tip of the last link as the first element in the list,
            # so we have all positions in one array.
            R3_tip2I = SOA.get_rotation_tip_to_body_I(theta_list, self.links, n)[:3,:3]
            #R3_tip2I = R6_tip2I
            tip_pos = (positions[1] + R3_tip2I @ self.links[0].l_hinge).flatten()
            positions[0] = tip_pos
            
            return np.array(positions) # Convert list of arrays to a single 2D array of shape (n_bodies+1, 3) for easier plotting
        
        # Update function for animation
        def update(frame):
            state_k = ani_states[:, frame]
            positions = compute_positions(state_k)
            line.set_data(positions[:, 0], positions[:, 1])
            line.set_3d_properties(positions[:, 2])
            ax.set_title(f"t = {ani_tspan[frame]:.3f} s")

        # Just a robust way of calculating the interval between frames for the animation, based on the time vector. Could also do tspan[1] - tspan[0]. 
        dt = np.mean(np.diff(ani_tspan))
        interval = dt * 1000 # Convert to milliseconds for FuncAnimation
        ani = FuncAnimation(
            fig, update, frames=N, interval=interval, blit=False
        )
        
        ax.view_init(elev=0, azim=-90, roll=0)
        plt.show()
        return ani

    def plot_static_snapshots_grid(self, num_snapshots=6, config="closed"):
        assert self.result is not None, "No simulation result found. Please run simulation first."
        #for creatig nice plots with tracings. Is hardcoded for now
        total_frames = self.result.shape[1]
        snapshot_indices = np.linspace(0, total_frames - 1, num_snapshots, dtype=int)

        ncols = 3
        nrows = int(np.ceil(num_snapshots / ncols))
        
        # Adjusted figure size to 12x8 to perfectly match a 3:2 grid ratio
        fig = plt.figure(figsize=(12, 8))

        def compute_positions(state_k):
            theta_list, _ = self.unpack_state(state_k)
            positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, len(self.links))
            R3_tip2I = SOA.get_rotation_tip_to_body_I(theta_list, self.links, len(self.links))[:3, :3]
            tip_pos = (positions[1] + R3_tip2I @ self.links[0].l_hinge).flatten()
            positions[0] = tip_pos
            return np.array(positions)

        # --- AUTOMATIC BOUNDING BOX ZOOM ---
        all_x, all_z = [], []
        for f in range(total_frames):
            pos = compute_positions(self.result[:, f])
            all_x.extend(pos[:, 0])
            all_z.extend(pos[:, 2])
            
        max_range = max(max(all_x) - min(all_x), max(all_z) - min(all_z))
        mid_x = (max(all_x) + min(all_x)) / 2
        mid_z = (max(all_z) + min(all_z)) / 2
        
        # FIX 1: Increased padding from 0.51 to 0.65 to give the figures more breathing room
        padding = max_range * 0.65  
        xlims = [mid_x - padding, mid_x + padding]
        zlims = [mid_z - padding, mid_z + padding]

        # --- HARDCODED PERFECT DRIVER CIRCLE ---
        driver_center_x = 0.0
        driver_center_z = 0.0
        driver_radius = 0.2

        angles = np.linspace(0, 2 * np.pi, 100)
        circle_x = driver_center_x + driver_radius * np.cos(angles)
        circle_z = driver_center_z + driver_radius * np.sin(angles)

        # --- TRAJECTORY PATH FOR THE RED POINT ---
        body3_trajectory = np.array([compute_positions(self.result[:, f])[0] for f in range(total_frames)])

        for idx, frame_idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            
            # --- DRAW LIGHT GREY DRIVER CIRCLE ---
            ax.plot(circle_x, [0]*100, circle_z, color='#d8d8d8', linestyle='-', lw=1.2, zorder=1)

            # --- DRAW CONTINUOUS PATH TRACE ---
            if frame_idx > 0:
                past_path = body3_trajectory[:frame_idx+1]
                ax.plot(past_path[:, 0], past_path[:, 1], past_path[:, 2], 
                        color='crimson', linestyle='-', alpha=0.3, lw=1.5, zorder=2)

            current_state = self.result[:, frame_idx]
            current_pos = compute_positions(current_state)
            
            # --- DRAW MECHANISM LINKS ---
            ax.plot(current_pos[:, 0], current_pos[:, 1], current_pos[:, 2], 
                    'o-', color='C0', lw=2.5, markersize=4.5, zorder=3)
            
            ax.plot([current_pos[-1, 0], current_pos[0, 0]], 
                    [current_pos[-1, 1], current_pos[0, 1]], 
                    [current_pos[-1, 2], current_pos[0, 2]], 
                    'o-', color='C0', lw=2.5, markersize=4.5, zorder=3)

            # --- HIGHLIGHT ORIGIN NODE IN RED ---
            ax.plot([current_pos[0, 0]], [current_pos[0, 1]], [current_pos[0, 2]], 
                    'o', color='crimson', markersize=6.5, zorder=5)

            # --- VIEWPORT SETTINGS ---
            ax.set_xlim(xlims)
            ax.set_ylim([-padding, padding])
            ax.set_zlim(zlims)
            ax.view_init(elev=0, azim=-90, roll=0)
            ax.set(box_aspect=(1, 1, 1))
            
            # --- ELIMINATE 3D PERSPECTIVE ARTIFACTS ---
            ax.grid(False)
            ax.set_axis_off() 
            ax.xaxis.line.set_linewidth(0)
            ax.yaxis.line.set_linewidth(0)
            ax.zaxis.line.set_linewidth(0)

            # --- BLACK BORDERS & DARK GREY CROSSHAIRS ---
            box_x = [xlims[0], xlims[1], xlims[1], xlims[0], xlims[0]]
            box_z = [zlims[0], zlims[0], zlims[1], zlims[1], zlims[0]]
            ax.plot(box_x, [0]*5, box_z, color='#000000', lw=1.2, zorder=4) 

            ax.plot(xlims, [0, 0], [0, 0], color='#444444', linestyle='--', alpha=0.25, lw=0.8, zorder=1)
            ax.plot([0, 0], [0, 0], zlims, color='#444444', linestyle='--', alpha=0.25, lw=0.8, zorder=1)

            # --- TITLE ---
            ax.set_title(f"t = {self.tspan[frame_idx]:.2f} s", fontsize=11, y=0.88, weight='bold', color='#222222')
                
        # --- FIX 2: NEGATIVE SPACING COMPRESSION ---
        # 3D axes in Matplotlib have massive invisible margins. 
        # Using negative wspace and hspace overrides this padding to pull the squares together.
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94, wspace=-0.25, hspace=-0.15)
        plt.show()

    def plot_initial_state(self, config="openclosed"):
        # Number of bodies and time steps
        n = len(self.links)

        # Setting up the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Defining plotlim based on the number of bodies, link length and configuration
        total_link_length = sum(np.linalg.norm(link.l_hinge) for link in self.links)
        if config == "open":
            plotlim = total_link_length + np.linalg.norm(self.links[0].l_hinge)
        elif config == "closed":
            plotlim = total_link_length/2 + np.linalg.norm(self.links[0].l_hinge)
        else:
            raise ValueError("Invalid config value. Use 'open' or 'closed'.")
        
        # Set plot limits and labels
        ax.set_xlim([-plotlim, plotlim])
        ax.set_ylim([-plotlim, plotlim])
        ax.set_zlim([-plotlim, plotlim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set(box_aspect=(1, 1, 1))

        state0 = self.get_initial_state()
        theta_list ,_ = self.unpack_state(state0) #theta_list start at 0, i.e. body 1 has index 0.
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        # positions is constructed as a list of arrays, where the index 0 is empty. We can conveniently insert the tip of the last link as the first element in the list,
        # so we have all positions in one array.
        R3_tip2I = SOA.get_rotation_tip_to_body_I(theta_list, self.links, n)[:3,:3]
        tip_pos = (positions[1] + R3_tip2I @ self.links[0].l_hinge).flatten()
        positions[0] = tip_pos
        pos_array = np.array(positions)     
       
        # Plotting the "skeleton"
        ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'o-', lw=3, markersize=8)

        ax.view_init(elev=0, azim=-90, roll=0)
        plt.grid(True)
        plt.show()

        return fig, ax

    def compute_com_pos_in_inertial_frame(self, theta_list):
        n = len(self.links)
        
        positions = [None]*(n+1)
        com_positions = [None]*(n+1)

        R_cumulative = self.links[n-1].joint.get_spatial_rotation(theta_list[n-1]) #initial rotation from body n to inertial frame
        R_cumulative = R_cumulative[:3,:3]

        #BC for position of base body
        positions[n] = self.links[n-1].joint.get_translation(theta_list[n-1])
        com_positions[n] = positions[n] + R_cumulative @ self.links[n-1].l_com

        for i in range(n-1,0,-1):
            pRc = self.links[i-1].joint.get_spatial_rotation(theta_list[i-1]) # bc self.links and theta_list starts from 0
            pRc = pRc[:3,:3]

            positions[i] = positions[i+1] + R_cumulative @ self.links[i].l_hinge # self.links start from 0...
            
            R_cumulative = R_cumulative @ pRc

            com_positions[i] = positions[i] + R_cumulative @ self.links[i-1].l_com # self.links start from 0...

        return com_positions

    def calc_energies(self, z0):
        # Colab between Kap and Gemini
        """
        Calculates the kinetic, potential, and total energy of the system for all time steps.
        Saves the results as 1D numpy arrays in self.KE, self.PE, and self.TE.
        
        Args:
            z0: A scalar or list of length n, specifying the potential energy reference offset for each body.
        """
        # ADDED: Calcs TE_delta, which is change in TE compared to initial TE.

        n = len(self.links)
        
        if self.result is None:
            raise ValueError("Simulation must be run before calculating energies.")
            
        if len(z0) != n:
            raise ValueError(f"z0 must be of length {n} (one offset per link)")
        
        if hasattr(self, 'TE_delta') and self.TE_delta is not None:
            raise ValueError("calc_TE_delta has already been run.")
        
        # Adjust for indexing
        z0 = np.insert(z0, 0, 0)

        nt = len(self.tspan)
        self.KE = np.zeros(nt)
        self.PE = np.zeros(nt)
        self.TE = np.zeros(nt)
        self.TE_error = np.zeros(nt)
        
        g = 9.81
        
        for i in range(nt):
            # Initalization for each timestep
            KE_t = 0.0
            PE_t = 0.0

            # Current state
            state = self.result[:, i]
            theta_list, _ = self.unpack_state(state)            
            
            # Compute com positions of hinges in the inertial frame
            com_pos = self.compute_com_pos_in_inertial_frame(theta_list)
            
            for k in range(n, 0, -1):
                link = self.links[k-1]
                
                # Kinetic Energy for this link (0.5 * V.T * M * V)
                Vk = self.V[i][k]
                KE_t += 0.5 * (Vk.T @ link.M @ Vk)
                
                # Potential Energy for this link (m * g * h)
                zk_pot = com_pos[k][-1] + z0[k]  # z-coordinate + offset
                PE_t += link.m * g * zk_pot

            self.KE[i] = KE_t
            self.PE[i] = PE_t
            self.TE[i] = KE_t + PE_t

            if i == 0: # Initial instance
                TE_ini = KE_t + PE_t
                self.TE_error[i] = 0.0
                
            else:
                self.TE_error[i] = self.TE[i] - TE_ini
            
        print("Energies calculated!")

    def calc_TE_error(self):
        if self.result is None:
            raise ValueError("Simulation must be run before calculating energies.")
        
        n = len(self.links)
        nt = len(self.tspan)
        self.TE_error = np.zeros(nt)

        g = 9.81
        
        for i in range(nt):
            # Initalization for each timestep
            KE_rel_t = 0.0
            PE_rel_t = 0.0

            # Current state
            state = self.result[:, i]
            theta_list, _ = self.unpack_state(state)            
            
            # Compute com positions of hinges in the inertial frame
            com_pos = self.compute_com_pos_in_inertial_frame(theta_list)
            
            if i == 0: # Initial instance
                com_pos_ini = com_pos
                Vk_ini = self.V[i]

            for k in range(n, 0, -1):
                link = self.links[k-1]
                
                # Kinetic Energy for this link
                Vk = self.V[i][k]
                KE_t = 0.5 * (Vk.T @ link.M @ Vk)
                KE_ini = 0.5 * (Vk_ini[k].T @ link.M @ Vk_ini[k])
                KE_rel_t += KE_t - KE_ini

                # Relative potential Energy for this link
                zk_pot_rel = com_pos[k][-1] - com_pos_ini[k][-1]
                PE_rel_t += link.m * g * zk_pot_rel

            self.TE_error[i] = KE_rel_t + PE_rel_t

        print("TE_error calculated!")

    def return_TE_error_mean(self):
        if self.TE_error is None:
            raise ValueError("TE_error has not been calculated yet.")
        return np.mean(self.TE_error)

    def CSV_creator(self, path, filename, *attr_names):

        # Made mainly by Gemini
        """
        Merges an arbitrary number of attributes (lists/arrays stored in self) 
        into a CSV file, where each attribute represents a column. 
        Raises a ValueError if the attributes do not have the same number of rows.
        """
        
        if not attr_names:
            print("No attribute names provided.")
            return
            
        extracted_lists = []
        for name in attr_names:
            if not hasattr(self, name):
                raise AttributeError(f"The system does not have an attribute named '{name}'.")
            
            attr_data = getattr(self, name)
            
            # Handle lists that might contain mixed elements (e.g., None and arrays)
            if isinstance(attr_data, list):
                processed_data = []
                for row in attr_data:
                    if isinstance(row, (list, tuple, np.ndarray)):
                        flat_row = []
                        for item in row:
                            if item is None:
                                continue
                            elif isinstance(item, (int, float, str, np.number)):
                                flat_row.append(item)
                            else:
                                flat_row.extend(np.asarray(item).flatten().tolist())
                        processed_data.append(flat_row)
                    else:
                        processed_data.append(row)
                arr = np.asarray(processed_data)
            else:
                arr = np.asarray(attr_data)
                
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            extracted_lists.append(arr)
            
        expected_length = extracted_lists[0].shape[0]
        
        for i in range(len(extracted_lists)):
            if extracted_lists[i].shape[0] != expected_length:
                # Automatically transpose wide arrays (like self.result) to match expected row count
                if extracted_lists[i].ndim == 2 and extracted_lists[i].shape[1] == expected_length:
                    extracted_lists[i] = extracted_lists[i].T
                else:
                    raise ValueError(f"Attribute '{attr_names[i]}' has {extracted_lists[i].shape[0]} rows, expected {expected_length}.")
                
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Combine arrays horizontally to support multiple columns per array
        combined_data = np.hstack(extracted_lists)
        
        # Combine path and filename
        path_filename = path + "/" + filename
        
        with open(path_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(combined_data)
            
        print(f"Data successfully saved as {filename} in {path}.")

    def get_all_rotations_body_to_I(self, theta_list):
        """
        Computes the spatial rotation matrix from each body's frame to the inertial frame 
        in a single O(n) sweep. Returns a 1-indexed list of 6x6 spatial rotation matrices.
        """
        n = len(self.links)
        IR_list = [None] * (n + 1)
        
        # Start at the base (body n)
        # Note: self.links and theta_list are 0-indexed, so body n is at index n-1
        IR_cumulative = self.links[n-1].joint.get_spatial_rotation(theta_list[n-1])
        IR_list[n] = IR_cumulative
        
        # Sweep down the chain from n-1 to the tip (1)
        for k in range(n-1, 0, -1):
            pRc = self.links[k-1].joint.get_spatial_rotation(theta_list[k-1])
            IR_cumulative = IR_cumulative @ pRc  # Multiply parent's rotation by child's relative rotation
            IR_list[k] = IR_cumulative
            
        return IR_list

    def plot_attribute(self, attr_name):
        """
        A simple plot of a system attribute against time.

        Args:
            attr_name (str): The name of the attribute to plot.
        """
        if not hasattr(self, attr_name):
            print(f"Error: Attribute '{attr_name}' not found in the system.")
            return

        y_data = getattr(self, attr_name)
        plt.figure()
        plt.plot(self.tspan, y_data)
        plt.xlabel("Time [s]")

        # --- SCIENTIFIC NOTATION FORMATTING --- 
        # This triggers scientific notation for numbers smaller than 10^(-2) or larger than 10^3
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(formatter)
        
        plt.show()

    def calc_and_plot_penetration(self):
        """
        Calculates and plots the maximum penetration depth of any joint into the sprockets over time.
        """
        if self.result is None:
            raise ValueError("Simulation must be run before calculating penetration.")
            
        n = len(self.links)
        nt = len(self.tspan)
        max_penetrations = np.zeros(nt)
        
        for i in range(nt):
            t = self.tspan[i]
            state = self.result[:, i]
            theta_list, _ = self.unpack_state(state)
            positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
            
            #hardcoded sprocket geometry. could be made into an input if wanted
            sprockets = (
                 (np.array([-0.6, 0.0, 0.0]) , 0.3864), # Left Sprocket
                 (np.array([0.6, 0.0, 0.0]), 0.3864)   # Right Sprocket
             )

            max_pen = 0.0
            for k in range(1, n+1):
                pos = positions[k]
                for center, radius in sprockets:
                    dist = np.linalg.norm(pos - center)
                    pen = radius - dist
                    if pen > max_pen:
                        max_pen = pen
            max_penetrations[i] = max_pen
            
        self.penetration = max_penetrations
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")
        

        ax.plot(self.tspan, max_penetrations, color='red', label='Max Penetration')
        
        ax.set_xlabel("Time [s]", fontsize=14)
        ax.set_ylabel("Penetration Depth [m]", fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Force scientific notation for the y-axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Shared Legend style
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=1, fontsize=14, frameon=True, framealpha=0.9, labelspacing=1.2)
        
        plt.show()