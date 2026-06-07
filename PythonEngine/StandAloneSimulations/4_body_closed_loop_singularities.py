#imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils import objects as ob
from utils import soa as SOA
import matplotlib.pyplot as plt
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints

joint4 = ob.SphericalJoint()
joint3 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right

joint4.q_init = SOA.quatfromrev(np.pi, "y")
joint3.q_init = SOA.quatfromrev(-np.pi/2, "y")
joint2.q_init = SOA.quatfromrev(-np.pi/2, "y")
joint1.q_init = SOA.quatfromrev(-np.pi/2, "y")


joint4.w_init = np.array([0,1,0])

#defining link
mass = 20
link4 = ob.Link(mass=mass, l_hinge=np.array([0, 0, 0.2]), joint=joint4)
link3 = ob.Link(mass=mass, l_hinge=np.array([0, 0, 0.2]), joint=joint3)
link2 = ob.Link(mass=mass, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=mass, l_hinge=np.array([0, 0, 0.2]), joint=joint1)


#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link4)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 5
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 0*9.81 #simulating gravity in z


robot.plot_initial_state("closed")
robot.simulate(tspan,V_base,A_base,"closed",BG_params=[10,40])
#plt.plot(tspan,robot.constraint_violation)
#plt.grid()
#plt.show()

robot.animation(config="closed",step=5)