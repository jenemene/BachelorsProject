#imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint3 = ob.FreeJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#intialziing such that pendulum is hanging to the right
joint3.q_init = np.hstack([SOA.quatfromrev(np.pi/2, "y"),np.array([0.0,0.0,0.0])])
joint2.q_init = SOA.quatfromrev(2*np.pi/3, "y")
joint1.q_init = SOA.quatfromrev(2*np.pi/3, "y")

#defining link
link3 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 5
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

robot.plot_initial_state("closed")

robot.simulate(tspan,V_base,A_base,"multiple_constraints",BG_params=[100,200])

robot.plot_static_snapshots_grid()

# robot.plot_gen_velocities(savefig=True)

# robot.animation(config="closed",step=5)
