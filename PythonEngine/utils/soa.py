import numpy as np

def rotfromquat(quat):
    #Convert a quaternion to a rotation matrix.

    #Args: quat: Unit quaternion as a np array of shape (4,)

    #Returns: np array: A 3x3 rotation matrix a np array of shape (3,3)

    assert quat.shape == (4,), "Input quaternion must be of shape (4,)"
    
    I = np.eye(3)
    q0 = quat[3]
    q = quat[:3]
    q_tilde = skewfromvec(q)
    R = I + 2*(q0*I + q_tilde)@q_tilde
    return R

def skewfromvec(vec):
    #Convert a 3D vector to a skew-symmetric matrix.

    #Args: vec: A 3D vector as a np array of shape (3,)

    #Returns: np array: A 3x3 skew-symmetric matrix as a np array of shape (3,3)
    
    assert vec.shape == (3,), "Input vector must be of shape (3,)"

    S = np.array([[0, -vec[2], vec[1]],
                      [vec[2], 0, -vec[0]],
                      [-vec[1], vec[0], 0]])
    return S

def RBT(vec):
    #Rigid body transformation matrix - no rotations

    # Args: 3D vector as a np array of shape (3,)

    # Returns: 6x6 rigid body transformation matrix as a np array of shape (6,6)  
    
    assert vec.shape == (3,), "Input vector must be of shape (3,)"

    I = np.eye(3)
    Z = np.zeros((3,3))
    l_tilde = skewfromvec(vec)

    phi = np.block([[I,l_tilde],[Z,I]])
    return phi

def spatialskewbar(X):
    #Convert a 6D spatial-vector into a 6x6 skew-symmetric matrix as defined in 1.25 in ABI book

    #Args: vec: A 6D vector as a np array of shape (6,)

    #Returns: np array: A 6x6 skew-symmetric matrix as a np array of shape (6,6)

    assert X.shape == (6,), "Input spatial vector must be of shape (6,)"

    X_bar = np.block([[skewfromvec(X[:3]),skewfromvec(X[3:])],
             [np.zeros((3,3)),skewfromvec(X[:3])]])
    return X_bar
  
def spatialskewtilde(spatialvec):
    #Convert a 6D spatial vector to a 6x6 skew-symmetric matrix.

    #Args: spatialvec: A 6D spatial vector as a np array of shape (6,)

    #Returns: np array: A 6x6 skew-symmetric matrix as a np array of shape (6,6)

    assert spatialvec.shape == (6,), "Input spatial vector must be of shape (6,)"

    w = spatialvec[:3]
    v = spatialvec[3:]

    w_tilde = skewfromvec(w)
    v_tilde = skewfromvec(v)

    S = np.block([[w_tilde, np.zeros((3,3))],
                  [v_tilde, w_tilde]])
    return S
    
def spatialrotfromquat(quat):
    #Convert a quaternion to a 6x6 spatial rotation matrix.

    #Args: quat: Unit quaternion as a np array of shape (4,)

    #Returns: np array: A 6x6 spatial rotation matrix as a np array of shape (6,6)

    assert quat.shape == (4,), "Input quaternion must be of shape (4,)"
    
    R = rotfromquat(quat)

    spatialR = np.block([[R, np.zeros((3,3))],
                         [np.zeros((3,3)), R]])
    return spatialR

def derrivmap(theta,omega,type="type of joint"):
    #arguments:
    #theta: scalar or np array of shape (N,). Generalized coordinate(s)
    #omega: angular velocity of k wrt to k+1 (this is usually just the generalized velocity depending on definition chosen)
    #type: string indicating the type of hinge, either "revolute" or "spherical)

    #returns:
    #derrivative of generalized coordiantes (theta_dot) - flattened, i.e of shape (N,)

    omega = omega.reshape(3,1)
    
    if type == "revolute":
        derriv = omega
    elif type == "spherical":
        derriv = 0.5*np.block([[-skewfromvec(omega.flatten()), omega],
                               [-omega.T, 0]]) @ theta.reshape(4,1)

    else:
        raise ValueError("Type must be either 'revolute' or 'spherical'")
    
    return derriv.flatten()

def quatfromrev(theta,axis="axis of orientation"):
    #computes the quartenion represenation of the relative rotation of two bodies for a revolute joint
    #arguments:
    #theta: scalar: Generalized coordinate for that revolute joint
    #axis: Rotation axis as a string, either "x", "y" or "z"

    #returns:
    #quat: Unit quaternion as a np array of shape (4,)

    if axis == "x":
        n = np.array([1,0,0])
    elif axis == "y":
        n = np.array([0,1,0])
    elif axis == "z":
        n = np.array([0,0,1])
    else:
        raise ValueError("Axis must be either 'x', 'y' or 'z'")
    
    q_vec = np.sin(theta/2)*n
    q_scalar = np.cos(theta/2)

    q = np.concatenate((q_vec, np.array([q_scalar])))

    return q

def normalize_quaternions(q):
    # ADD copy=True to protect your RK4 state!
    q_safe = np.array(q, copy=True) 
    
    q_reshaped = q_safe.reshape(-1, 4)
    norms = np.linalg.norm(q_reshaped, axis=1, keepdims=True)
    q_reshaped /= norms
    return q_reshaped.reshape(-1)
 
###----------------- BENEATH HERE IS LEGACY CODE ----------------- ### 

def get_rotation_tip_to_body_I(theta_list, links, n):   
    # Initialize total rotation as Identity (Body 1 in Body 1 frame)
    R_total = np.eye(3) 

    # Loop from 1 up to n-1
    # We use theta[k] which describes orientation of k relative to k+1 (parent)
    # Chain: R_n1 = R_n,n-1 @ ... @ R_3,2 @ R_2,1
    
    for k in range(0, n): 
        # Calculate rotation R_{k+1, k} (Parent-from-Child)
        pRc = links[k].joint.get_spatial_rotation(theta_list[k])[:3, :3]
        
        # Accumulate: New_Total = Current_Link_Rotation @ Old_Total
        # This builds the chain: R_{k+1, 1} = R_{k+1, k} @ R_{k, 1}
        R_total = pRc @ R_total

    R = np.block([[R_total, np.zeros((3,3))],
                    [np.zeros((3,3)),R_total]])

    return R

def compute_pos_in_inertial_frame(theta_list, links, n):

    positions = [None]*(n+1) # allow for index to match body

    #BC for position of base body
    positions[n] = links[n-1].joint.get_translation(theta_list[n-1])

    R_cumulative = links[n-1].joint.get_spatial_rotation(theta_list[n-1]) #initial rotation from body n to inertial frame
    R_cumulative = R_cumulative[:3, :3]

    for i in range(n-1,0,-1):        
        pRc = links[i-1].joint.get_spatial_rotation(theta_list[i-1])
        pRc = pRc[:3, :3]

        #positions variable follows body index, but links does not, i.e. body 1 is index 0 in links.
        #hence to get the position of body k, you take the position of body k+1 and add the vector from k+1 to k, l(k+1,k).
        #but that vector is defined in link k+1. BUT since index for links starts from 0, it should be i and not i+1
        positions[i] = positions[i+1] + R_cumulative @ links[i].l_hinge

        R_cumulative = R_cumulative @ pRc

    return positions

def compute_com_pos_in_inertial_frame(theta_vec, l_vec, n):
    
    theta = [None]*(n+1)

    #unpacking interior 
    for i in range(1, n+1):
        idxq = 4*(i-1)
        theta[i] = theta_vec[idxq:idxq+4]

    positions = [None]*(n+1)
    com_positions = [None]*(n+1)

    R_cumulative = rotfromquat(theta[n]) #initial rotation from body n to inertial frame

    #BC for position of base body
    positions[n] = np.zeros(3)
    com_positions[n] = R_cumulative @ l_vec*0.5

    for i in range(n-1,0,-1):
        pRc = rotfromquat(theta[i])

        positions[i] = positions[i+1] + R_cumulative @ l_vec
        
        R_cumulative = R_cumulative @ pRc

        com_positions[i] = positions[i] + R_cumulative @ l_vec*0.5

    return com_positions

def compute_pos_in_body_frame(theta_vec, l_vec, n):
    # Args:
    # theta_vec: Flattened state vector of quaternions
    # l_vec: Vector from O_k to O+_k-1 in k frame (same for all links)
    # n: number of bodies (where body 1 is tip, body n is connected to base)
    # Returns:
    # positions: List of 3D position vectors of each body frame in body frame (where position of body k is position of O_k in k frame)

    theta = [None]*(n+1)

    #unpacking interior 
    for i in range(1, n+1):
        idxq = 4*(i-1)
        theta[i] = theta_vec[idxq:idxq+4]

    positions = [None]*(n+1)

    #BC for positions
    positions[n] = np.zeros(3)

    for i in range(n-1,0,-1):        
        cRp = rotfromquat(theta[i]).T

        positions[i] = cRp @ (positions[i+1] + l_vec)

    return positions

def baumgarte_stab(Φ, Φ_dot, Φ_ddot, alpha, beta):

    return Φ_ddot + (2*alpha * Φ_dot) + (beta**2 * Φ)

def FE_int(odefun,initial_cond,time_vec,n,link,RBT):
    dt = time_vec[1] - time_vec[0] # calculation of timestep
    N = len(time_vec) #amount of time steps
    m = len(initial_cond) #size of result vectors for each timestep

    Y = np.zeros((m,N)) #intializing storage array

    Y[:,0] = initial_cond #intialzing initial cond

    for i in range(N-1):
        Y[:,i+1] = Y[:,i] + dt * odefun(time_vec[i],Y[:,i],n,link,RBT)

    return Y

def RK4_int(odefun, initial_cond, time_vec, n,link):
    time_vec = np.asarray(time_vec)
    y0 = np.asarray(initial_cond).reshape(-1)

    dt = time_vec[1] - time_vec[0]
    N  = len(time_vec)
    m  = len(y0)

    Y = np.zeros((m, N))
    Y[:, 0] = y0

    # initial V - spatial vel

    for i in range(N - 1):
        t = time_vec[i]
        y = Y[:, i]

        k1 = odefun(t,y,n,link)
        k2 = odefun(t + dt/2.0,y + dt/2.0 * k1,n,link)
        k3 = odefun(t + dt/2.0,y + dt/2.0 * k2,n,link)
        k4 = odefun(t + dt,y + dt * k3,n,link)

        Y[:, i+1] = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
 
    return Y

def RK4_int_with_V(odefun, initial_cond, time_vec, n,link):
    time_vec = np.asarray(time_vec)
    y0 = np.asarray(initial_cond).reshape(-1)

    dt = time_vec[1] - time_vec[0]
    N  = len(time_vec)
    m  = len(y0)

    Y = np.zeros((m, N))
    Y[:, 0] = y0

    #for storing spatial velcoities
    V_storage = [None]*N

    # initial V - spatial vel



    for i in range(N - 1):
        t = time_vec[i]
        y = Y[:, i]

        k1,V_val = odefun(t,y,n,link)
        k2,_ = odefun(t + dt/2.0,y + dt/2.0 * k1,n,link)
        k3,_ = odefun(t + dt/2.0,y + dt/2.0 * k2,n,link)
        k4,_ = odefun(t + dt,y + dt * k3,n,link)

        Y[:, i+1] = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V_storage[i] = V_val

    #filling in last timestep 
    _,V_storage[N-1] = odefun(time_vec[N-1], Y[:, N-1], n, link)

    return Y, V_storage
    
def RK4_int_with_V_BG(odefun, initial_cond, time_vec, n, link, BG_params):
    time_vec = np.asarray(time_vec)
    y0 = np.asarray(initial_cond).reshape(-1)

    dt = time_vec[1] - time_vec[0]
    N  = len(time_vec)
    m  = len(y0)

    Y = np.zeros((m, N))
    Y[:, 0] = y0

    #for storing spatial velcoities
    V_storage = [None]*N

    # initial V - spatial vel

    for i in range(N - 1):
        t = time_vec[i]
        y = Y[:, i]

        k1,V_val = odefun(t, y, n, link, BG_params)
        k2,_ = odefun(t + dt/2.0, y + dt/2.0 * k1, n, link, BG_params)
        k3,_ = odefun(t + dt/2.0, y + dt/2.0 * k2, n, link, BG_params)
        k4,_ = odefun(t + dt, y + dt * k3, n, link, BG_params)

        Y[:, i+1] = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V_storage[i] = V_val

    #filling in last timestep 
    _,V_storage[N-1] = odefun(time_vec[N-1], Y[:, N-1], n, link, BG_params)

    return Y, V_storage

class SimpleLink:
    def __init__(self,m,l_hinge):

        #adding attribtues to object
        self.m = m
        self.l_com = l_hinge/2
        self.l_hinge = l_hinge

        #calculating geometry (right now width and heigh of link is just 1/10 of length)
        l = np.linalg.norm(l_hinge)
        w = l/50
        h = w
        self.J_c = np.diag([1/12*m*(h**2 + w**2), 1/12*m*(l**2 + h**2), 1/12*m*(l**2 + w**2)])

        #spatial inertia at COM
        self.M_c =  np.block([[self.J_c, np.zeros((3,3))],
                          [np.zeros((3,3)), m*np.eye(3)]])
    
        self.M = RBT(self.l_com)@self.M_c@RBT(self.l_com).T #spatial inertia at body frame (located at hinge)

        #rigidbody transform across link
        self.RBT = RBT(l_hinge)
    
    def set_hingemap(self,type="hingetype"):
        if type == "spherical":
            self.H =np.block([[np.eye(3), np.zeros((3,3))]])
        else:
            print("right now i have only specified for spherical joints")


def get_rotation_body_to_I(theta_list, links, n, body_index):
    #kan nok godt slåes sammen med den ovenfor :)
    """
    Computes the rotation from a specific body to the inertial frame I.
    body_index: 1-based index (1 is the tip, n is the base connected to I).
    """
    R_total = np.eye(3) 

    # Chain from body_index up to n (Inertial frame)
    for k in range(body_index - 1, n): 
        pRc = links[k].joint.get_spatial_rotation(theta_list[k])[:3, :3]
        R_total = pRc @ R_total

    R = np.block([[R_total, np.zeros((3,3))],
                  [np.zeros((3,3)),R_total]])

    return R

       
def ATBI(state,tau_vec,n,link):
        #inputs
        #state: np.array on form [theta_dot, beta]
        #tau_vec: generalized forces as np.array
        #l_hinge: vector from O_k to O+_k-1 in k frame (this doesnt matter as they are identical in our case)
        #m: mass of length. Ensure that you dont have a very long and slender link with a small mass to avoid very stiff elements
        #type: hinge-type for all links. Right now its purely spherical that is implemented
        #n: no_bodies
        #link: Instantiate a link using the SimpleLink class and pass it
        #outputs beta_dot

        #unpacking state
        theta_vec = state[:4*n]
        beta_vec  = state[4*n:]

        theta = [None]*(n+2)
        beta  = [None]*(n+2)
        tau   = [None]*(n+2)

        # boundary conditions - det kan diskuteres om man behøver i begge ender for dem alle, det gør man vidst nok ikke
        theta[0]   = np.zeros(4)
        theta[n+1] = np.zeros(4)

        beta[0]    = np.zeros(3)
        beta[n+1]  = np.zeros(3)

        tau[0]     = np.zeros(3)
        tau[n+1]   = np.zeros(3)

        #unpacking interior 
        for i in range(1, n+1):

            idxq = 4*(i-1)
            idxw = 3*(i-1)

            theta[i] = theta_vec[idxq:idxq+4]
            beta[i]  = beta_vec[idxw:idxw+3]
            tau[i]   = tau_vec[3*(i-1):3*i]

         #if damping is to be implemented, then add a -b*beta[i] component in the for loop, or just do a simple tau[n] = -b*beta[n] if you only wish to damp body attatched to ground   

        for i in range(1, n+1):
            # ... unpacking idx ...
            
            # Calculate damping torque (viscous friction)
            b = 0 # Damping coefficient
            damping_tau = -b * beta[i]
            
            # Add it to any other external torques (currently zero)
            tau[i] = tau_vec[3*(i-1):3*i] + damping_tau

        #storage
        P_plus = [None]*(n+2)
        xi_plus = [None]*(n+2)
        nu = [None]*(n+2)
        A = [None]*(n+2)
        V = [None]*(n+2)
        G = [None]*(n+2)
        D = [None]*(n+2)
        beta_dot = [None]*(n+2)
        tau_bar = [None]*(n+2)
        agothic = [None]*(n+2)
        bgothic = [None]*(n+2)

        #boundary conditions on spatial operator quantities
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = P_plus[0]
        A[n+1] = np.array([0, 0, 0, 0, 0, 9.81]) # Psudo gravity in the last frame, which is the inertial frame
        V[n+1] = np.zeros((6,))

        #kinematics scatter

        for k in range(n,0,-1):
            #rotation matrices
            pRc = spatialrotfromquat(theta[k]) 
            cRp = pRc.T #from parent to child -> this is the direction we are going right now

            #hinge contribtuion
            delta_V = link.H.T @ beta[k]

            #spatial velocity
            V[k] = cRp @ link.RBT.T @ V[k+1] + delta_V

            #coriolois acc
            agothic[k] = spatialskewtilde(V[k]) @ link.H.T @ beta[k]

            #gyroscopic term
            bgothic[k] = spatialskewbar(V[k]) @ link.M @ V[k]

        #ATBI gather 
        for k in range(1,n+1): #n+1 as python does not include end index

            #rotations
            pRc = spatialrotfromquat(theta[k-1]) #using k-1 as orientation is defined as k+1_q_k and we need k_q_k-1
            cRp = pRc.T 

            P = link.RBT @ pRc @ P_plus[k-1] @ cRp@link.RBT.T + link.M
            D[k] = link.H @ P @ link.H.T
            G[k] = np.linalg.solve(D[k], link.H @ P).T #P @ link.H.T @ np.linalg.inv(D)
            tau_bar[k] = np.eye(6) - G[k] @ link.H
            P_plus[k] = tau_bar[k] @ P
            xi = link.RBT @ pRc @ xi_plus[k-1] + P @ agothic[k] + bgothic[k]
            eps = tau[k] - link.H@xi
            nu[k] = np.linalg.solve(D[k], eps) #= np.linalg.inv(D)@eps
            xi_plus[k] = xi + G[k]@eps

        #ATBI scatter
        for k in range(n,0,-1):
            #rotations
            pRc = spatialrotfromquat(theta[k])
            cRp = pRc.T 

            A_plus = cRp@ link.RBT.T @A[k+1]
            beta_dot[k] = nu[k] - G[k].T @ A_plus
            A[k] = A_plus + link.H.T @ beta_dot[k] + agothic[k]

        return A, V, beta_dot, tau_bar, D, G
    
def omega(theta_vec,link,tau_bar,D,n):
    #unpacking generalized coordinates
    theta = [None]*(n+2)
    theta[0] = np.zeros(4)
    theta[n+1] = np.zeros(4)

    #unpacking interior 
    for i in range(1, n+1):
        idxq = 4*(i-1)
        theta[i] = theta_vec[idxq:idxq+4]

    #space allocation
    gamma = [None]*(n+2)
    omega = [None]*(n+2)
    
    #boundary condition
    gamma[n+1] = np.zeros((6,6))
    
    for k in range (n,0,-1):
    #calculating diagonal entries of omega
        pRc = spatialrotfromquat(theta[k]) #rotations
        cRp = pRc.T

        ##### ---------- ÆNDRET LINJER MED NYE ROTATIONER, GAMLE LINJE ER OVER DENNE --------------------------
        gamma[k] = tau_bar[k].T @ cRp @ link.RBT.T @ gamma[k+1] @ link.RBT @ pRc @ tau_bar[k] + link.H.T @ np.linalg.solve(D[k],link.H)
        ##### -------------------------------------------------------------------------------------------------  

    #assigning these
    omega[n] = gamma[n]

    #calculating off diagonal entries (and inserting the one on the dignoal)
        
    #de to loops kan nok godt kombineres. 
    for k in range (n-1,0,-1):
        pRc = spatialrotfromquat(theta[k]) #rotations
        cRp = pRc.T
        #OLD: psi = link.RBT @ tau_bar[k]
        #OLD: omega[k] = cRp @ omega[k+1] @ pRc @ psi

        # New?:
        omega[k] = cRp @ omega[k+1] @ link.RBT @ pRc @tau_bar[k]

    omega_nn = gamma[n]
    omega_n1 = omega[1]
    omega_1n = omega_n1.T
    omega_11 = gamma[1]

    return omega_nn, omega_n1, omega_1n,omega_11

def beta_dot_delta(theta_vec,tau_bar,link,n,D,f_c,G):

    #unpacking generalized coordinates
    theta = [None]*(n+2)
    theta[0] = np.zeros(4)
    theta[n+1] = np.zeros(4)

    #unpacking interior 
    for i in range(1, n+1):
        idxq = 4*(i-1)
        theta[i] = theta_vec[idxq:idxq+4]


    #f_c comes with RBT already applied where nessecary

    xi_delta = [None]*(n+2)
    beta_dot_delta = [None] * (n+2)
    nu = [None]*(n+2)
    lambda_list = [None]*(n+2) #NOT TO BE CONFUSED WITH LAGRANGE MULTIPLIERS; THIS IS JUST THE NOTATION FROM THE BOOK


    #boundary cond on xi_delta and lambda_list
    xi_delta[0] = np.zeros(6,)
    lambda_list[n+1] = np.zeros(6,)

    for k in range (1,n+1):
        pRc = spatialrotfromquat(theta[k-1]) #using k-1 as orientation is defined as k+1_q_k and we need k_q_k-1
        cRp = pRc.T 
        
        xi_delta[k] = link.RBT@pRc@tau_bar[k-1]@xi_delta[k-1] - f_c[k] #f_c er allerede rykket ud, derfor RBT er udeladt her
        nu[k] = np.linalg.solve(D[k],link.H@xi_delta[k]) #skulle være ok den her linje

    for k in range(n,0,-1):
        pRc = spatialrotfromquat(theta[k]) 
        cRp = pRc.T      

        lambda_list[k] = tau_bar[k].T @ cRp @ link.RBT.T @ lambda_list[k+1]+ link.H.T@nu[k]


        beta_dot_delta[k] = nu[k] - G[k].T@cRp@link.RBT.T@lambda_list[k+1]

    return beta_dot_delta