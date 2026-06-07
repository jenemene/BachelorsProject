#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import subprocess

# --- Helper to ask for numeric/float input with a default fallback ---
def ask_param(prompt, default_val, cast_type):
    user_input = input(f"{prompt} (default {default_val}): ").strip()
    if not user_input:
        return default_val
    try:
        return cast_type(user_input)
    except ValueError:
        print(f"  -> Invalid input. Using default: {default_val}")
        return default_val

# --- Helper to ask for a specific choice ---
def ask_choice(prompt, default_choice):
    user_input = input(f"{prompt} (default {default_choice}): ").strip()
    if not user_input:
        return default_choice
    if user_input in ['1', '2']:
        return user_input
    print(f"  -> Invalid selection. Using default: {default_choice}")
    return default_choice

print("\n=== Multibody Pendulum Simulator ===")
# Ask the user for simulation type and base parameters
sim_choice = ask_choice("Select engine (1: Open-Loop, 2: Closed-Loop)", '1')
n_bodies = ask_param("Enter number of bodies (n)", 3, int)
time_step = ask_param("Enter time step (dt)", 0.001, float)
end_time = ask_param("Enter simulation duration in seconds (t_end)", 2.0, float)

# 1. Route to the correct C++ Physics Engine and ask for specific params
if sim_choice == '2':
    cpp_executable = "/home/jenz/Desktop/SOAinC/build/sim_closed"
    sim_title = "Closed-Loop"
    print("\n--- Closed-Loop Tuning ---")
    alpha_bg = ask_param("Enter Baumgarte alpha (damping)", 50.0, float)
    beta_bg = ask_param("Enter Baumgarte beta (stiffness)", 50.0, float)
    print("====================================\n")
    
    args ake= [cpp_executable, str(n_bodies), str(time_step), str(end_time), str(alpha_bg), str(beta_bg)]
    print(f"Crunching numbers in C++ {sim_title} Engine (n={n_bodies}, dt={time_step}, t_end={end_time}, alpha={alpha_bg}, beta={beta_bg})...")

else:
    cpp_executable = "/home/jenz/Desktop/SOAinC/build/sim_open"
    sim_title = "Open-Loop"
    print("====================================\n")
    
    args = [cpp_executable, str(n_bodies), str(time_step), str(end_time)]
    print(f"Crunching numbers in C++ {sim_title} Engine (n={n_bodies}, dt={time_step}, t_end={end_time})...")

csv_file = "/home/jenz/Desktop/SOAinC/pendulum_data.csv"

if not os.path.exists(cpp_executable):
    print(f"Error: Could not find C++ executable at {cpp_executable}. Did you build it?")
    sys.exit(1)

# Run the engine with the dynamic argument list
subprocess.run(args, check=True) 


# 2. Load and Plot the Data
print("Loading and plotting data...")
if not os.path.exists(csv_file):
    print("Error: CSV file was not generated.")
    sys.exit(1)

df = pd.read_csv(csv_file)

n = (len(df.columns) - 1) // 3

fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
if n == 1:
    axes = [axes]

# Dynamically update the plot title based on the engine chosen
fig.suptitle(f'{n}-Body Pendulum Generalized Accelerations ({sim_title})')

for i in range(1, n + 1):
    ax = axes[i-1]
    
    ax.plot(df['time'], df[f'beta_dot{i}_x'], label=r'$\dot{\beta}_x$')
    ax.plot(df['time'], df[f'beta_dot{i}_y'], label=r'$\dot{\beta}_y$')
    ax.plot(df['time'], df[f'beta_dot{i}_z'], label=r'$\dot{\beta}_z$')
    
    if i == 1:
        label = 'Tip (J1)'
    elif i == n:
        label = f'Base (J{n})'
    else:
        label = f'Joint {i}'
        
    ax.set_ylabel(f'{label}\nrad/s²')
    ax.legend(loc='upper right')
    ax.grid(True)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()