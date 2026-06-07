import numpy as np
from utils import objects as ob
from utils import soa as SOA

# Initialize multibody system
robot = ob.MultiBodySystem()

#chain genereation
N_s = 6  # Number of straight links per top/bottom edge
N_c = 6  # Number of curved links per left/right edge
N = 2 * N_s + 2 * N_c  # Total = 24 links
L = 0.2  # Length of each link

# We need exactly 23 relative angles for the 23 spherical joints to close the loop:
# To form a symmetric semi-circle with discrete links, the transitions from straight to curved is 15 degrees. The inner joints forming half the 12-gon are 30 degrees
# 
rel_angles = (
    [0.0] * (N_s - 1) +               # Top straight (5 joints of 0°)
    [np.pi / (2 * N_c)] +             # Transition into right curve (15°)
    [np.pi / N_c] * (N_c - 1) +       # Right curve internal joints (5 joints of 30°)
    [np.pi / (2 * N_c)] +             # Transition out of right curve (15°)
    [0.0] * (N_s - 1) +               # Bottom straight (5 joints of 0°)
    [np.pi / (2 * N_c)] +             # Transition into left curve (15°)
    [np.pi / N_c] * (N_c - 1)         # Left curve internal joints (5 joints of 30°)
)

# Base initial coordinate
# Shift X left by half the straight section to center it at X=0
start_x = - (N_s * L) / 2

# Sprockets are at X= +/- 0.6, Z=0 with radius 0.3864.
# We spawn the top straight at Z = 0.3864 to give it exact clearance.
start_z = 0.3864
pos_base = np.array([start_x, 0.0, start_z])

# Rotate the base 90 degrees around Y so it lays perfectly horizontal
quat_base = SOA.quatfromrev(np.pi/2, "y")

# For loop to build the initial config
for i in range(N, 0, -1):
    if i == N:
        # BASE JOINT (FreeJoint)
        joint = ob.FreeJoint()
        joint.q_init = np.concatenate([quat_base, pos_base])
    else:
        # INTERNAL JOINTS (Spherical)
        joint = ob.SphericalJoint()
        # Map our angle array to the joints
        angle_idx = (N - 1) - i 
        joint.q_init = SOA.quatfromrev(rel_angles[angle_idx], "y")
        
    # Create the link 
    link = ob.Link(mass=20.0, l_hinge=np.array([0, 0, L]), joint=joint)
    
    # Add to system
    robot.add_link(link)

# --- 4. SIMULATION PARAMETERS ---
tspan = np.arange(0, 15.001, 0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 # Gravity in Z


# initial state plot
robot.plot_initial_state("closed")



# k = 5e7 
# c = 4000
robot.simulate(
    tspan, V_base, A_base, 
    config="sprockets", 
    BG_params=[0.1, 800], 
    Penalty_params=[5e7, 4000]
)
robot.calc_and_plot_penetration()

path = "PythonEngine/results"
file_name = "constraint_violation_sprockets"
robot.CSV_creator(path, file_name, "tspan", "constraint_violation")

robot.animation(config="closed", step=30)
