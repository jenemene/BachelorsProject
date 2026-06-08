import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import csv
from utils import objects as ob
from utils import soa as SOA

# Parameters for simulation
dt = 0.005
end_time = 10
tspan = np.arange(0, end_time + dt/2, dt)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81  # Simulating gravity in z

# Setup for validation
n_start = 1
n_end = 20
step_length = 1
repeats = 10

n_bodies_list = [1] + list(range(n_start, n_end+1, step_length))


# Initialize CSV file and write the header
csv_filename = "PythonEngine/order_n_val_results/solver_benchmark_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Num_Bodies", "Run_Index", "Solver_RK4_Time", "Solver_RK45_Time", "TE_delta_rk4"])

    # Loop through each configuration of body counts
    for n_bodies in n_bodies_list:
        print(f"Benchmarking system with {n_bodies} bodies...")
        
        # 1. Rebuild the MultiBodySystem for the current number of bodies
        pend = ob.MultiBodySystem()
        
        for i in range(n_bodies):
            joint = ob.SphericalJoint()
            # First link hangs to the right, subsequent links hang straight down
            if i == 0:
                joint.q_init = SOA.quatfromrev(3 * np.pi / 4, "y")
            else:
                joint.q_init = SOA.quatfromrev(0, "y")
                
            link = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint)
            pend.add_link(link)
        
        # 2. Run the benchmark repeats
        for r in range(repeats):
            # Benchmark custom RK4 solver
            start_rk4 = time.perf_counter()
            pend.simulate_own_RK4(tspan, V_base, A_base, "open")
            end_rk4 = time.perf_counter()
            t_rk4 = end_rk4 - start_rk4
            pend.calc_TE_error()
            TE_delta_rk4 = pend.return_TE_error_mean()
            
            # Benchmark solve_ivp RK45 solver
            start_rk45 = time.perf_counter()
            pend.simulate_solve_ivp(tspan, V_base, A_base, "open")
            end_rk45 = time.perf_counter()
            t_rk45 = end_rk45 - start_rk45
            
            # 3. Save the results row by row to prevent data loss if it takes long
            writer.writerow([n_bodies, r + 1, t_rk4, t_rk45, TE_delta_rk4])
            file.flush() # Forces writing to disk immediately

print(f"Benchmark complete! Data saved to {csv_filename}")