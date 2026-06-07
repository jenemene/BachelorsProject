#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints

joint3_fixed = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right
joint3_fixed.q_init = SOA.quatfromrev(0.5*np.pi, "y")
joint2.q_init = SOA.quatfromrev(2*np.pi/3, "y")
joint1.q_init = SOA.quatfromrev(2*np.pi/3, "y")

#defining link
link3_fixed= ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3_fixed)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#if free joint is wanted
joint3 = ob.FreeJoint()
joint3.q_init = np.hstack([SOA.quatfromrev(0.5*np.pi, "y"),np.array([0,0,0])])
joint3.w_init = np.array([0,1,0,0,0,0])

link3 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 10
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

robot.plot_initial_state("closed")
robot.simulate(tspan,V_base,A_base,"closed",BG_params=[200,500])
robot.calc_TE_error()
robot.plot_attribute("TE_error")
robot.plot_gen_velocities(savefig=False)
robot.animation(config="closed",step=5)

# path = "Arbejdspakke2/results"
# file_name = "3_closed_TE_error_BG_01_500"
# robot.CSV_creator(path, file_name, "tspan", "TE_error")

# file_name = "3_closed_gen_acc_t100"
# robot.CSV_creator(path, file_name, "tspan", "beta_dot")

