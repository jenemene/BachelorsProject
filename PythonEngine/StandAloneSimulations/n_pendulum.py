#imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
dp = ob.MultiBodySystem()

jointFree = ob.FreeJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#intialziing such that pendulum is hanging to the right
joint2.q_init = SOA.quatfromrev(3*np.pi/4, "y")
joint1.q_init = SOA.quatfromrev(0, "y")

jointFree.w_init = np.array([1,0,0,0,0,0])
# joint2.w_init = np.array([1,0,0])

#defining link
linkFree = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=jointFree)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
dp.add_link(link2)
dp.add_link(link1)
dp.add_link(link1)
dp.add_link(link1)
dp.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 10
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

# dp.plot_initial_state("open")

dp.simulate(tspan,V_base,A_base,"open")
dp.calc_TE_error()
dp.plot_attribute("TE_error")
dp.plot_gen_velocities(savefig=False)
dp.animation(config="open",step=5)

# should be specified [body 1, body 2, ..., body n]
# z0 = np.array([0.9,0.7,0.5,0.3,0.1])
# z0 = np.array([0.3,0.1])
# dp.calc_energies(z0)

# path = "Arbejdspakke2/results"
# file_name = "5p_TE_delta"
# dp.CSV_creator(path, file_name, "tspan", "TE_delta")

# file_name = "dp_gen_acc"
# dp.CSV_creator(path, file_name, "tspan", "beta_dot")

