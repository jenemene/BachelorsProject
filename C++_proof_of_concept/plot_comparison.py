#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#coded with help from Gemini Pro 3.1 VS-code extension.

script_dir = os.path.dirname(os.path.abspath(__file__))
# --- File Paths ---
cpp_csv = "pendulum_data.csv"
adams_csv = os.path.join(script_dir, "closed_3_body_adams_results(in).csv")
savefig = True

# --- 1. Load C++ Engine Data ---
if not os.path.exists(cpp_csv):
    print(f"Error: Could not find C++ results at {cpp_csv}")
    sys.exit(1)
    
df_cpp = pd.read_csv(cpp_csv)

# --- 2. Load ADAMS Data ---
if not os.path.exists(adams_csv):
    print(f"Error: Could not find ADAMS results at {adams_csv}")
    sys.exit(1)

# The ADAMS file has 6 rows of header/metadata before the actual numbers begin.
# We skip them and assign custom column names. 'sep=r"\s+"' handles the variable whitespace spacing.
df_adams = pd.read_csv(
    adams_csv, 
    sep=r'\s+', 
    skiprows=6, 
    names=['Time', 'J1_deg', 'J2_deg', 'J3_deg']
)

# Convert ADAMS (deg/s²) to match C++ Engine (rad/s²)
df_adams['J1_rad'] = -np.radians(df_adams['J1_deg'])
df_adams['J2_rad'] = -np.radians(df_adams['J2_deg'])
df_adams['J3_rad'] = -np.radians(df_adams['J3_deg'])

plot_configs = [
    (0, 'beta_dot1_y', 'J1_rad', 'Body 1'),
    (1, 'beta_dot2_y', 'J2_rad', 'Body 2'),
    (2, 'beta_dot3_y', 'J3_rad', 'Body 3')
]

# --- 3. Compute Error ---
if len(df_cpp) != len(df_adams):
    print(f"Error: Data length mismatch! C++ engine has {len(df_cpp)} rows, ADAMS has {len(df_adams)} rows.")
    sys.exit(1)

for ax_idx, cpp_col, adams_col, label in plot_configs:
    df_cpp[f'error_{cpp_col}'] = (df_cpp[cpp_col] - df_adams[adams_col])

if not np.allclose(df_cpp['time'], df_adams['Time'], atol=1e-5):
    print("Warning: Timestep values do not match between C++ and Adams data exactly.")

# --- 4. Create the Comparison Plots ---
n_bodies = len(plot_configs)
fig, axes = plt.subplots(n_bodies, 1, figsize=(8, 2 * n_bodies + 0.6), layout="constrained")
if n_bodies == 1:
    axes = [axes]

for ax_idx, cpp_col, adams_col, label in plot_configs:
    ax = axes[ax_idx]
    
    # Plot Data
    ax.plot(df_cpp['time'], df_cpp[cpp_col], label='C++', linewidth=2)
    ax.plot(df_adams['Time'], df_adams[adams_col], label='Adams', linestyle='--', linewidth=2)
    ax.plot(df_cpp['time'], df_cpp[f'error_{cpp_col}'], label='Error (C++ - Adams)', color='red', linestyle='--', linewidth=1.5)
    
    # Formatting
    ax.set_ylabel(f'{label} $\\dot{{\\beta}}_y$\n[rad/s$^2$]', fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if ax_idx in [0, 1]:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

axes[-1].set_xlabel('Time [s]', fontsize=14, labelpad=10)
fig.align_ylabels(axes)

# --- GLOBAL LEGEND GENERATION ---
handles, labels = [], []
for ax in fig.axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Filter out identical label duplicates (results in exactly 3 keys: C++, Adams, Error)
by_label = dict(zip(labels, handles))

# Place unique legend at the bottom center of the whole figure window
fig.legend(
    by_label.values(), 
    by_label.keys(), 
    loc='outside lower center', 
    ncol=len(by_label), 
    fontsize=12, 
    frameon=True, 
    framealpha=0.9
)

if savefig == True:
    plt.savefig("adams_comp_c++.pdf")
    print("Figure saved as adams_comp.pdf in current directory.")

plt.show()