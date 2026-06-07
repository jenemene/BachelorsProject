import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plot_all_energies(filename):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data = np.loadtxt(file_path, delimiter=',')
    tspan, PE, KE, TE = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    plt.figure(figsize=(10, 6))

    # Plot each component
    plt.plot(tspan, KE, label='Kinetic Energy (KE)')
    plt.plot(tspan, PE, label='Potential Energy (PE)')
    plt.plot(tspan, TE, label='Total Energy (TE)', linestyle='--', color='black')

    # Formatting
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.show()

def plot_TE(filename):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data = np.loadtxt(file_path, delimiter=',')
    tspan, PE, KE, TE = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    TE_Delta = TE - TE[0] # Normalize total energy to start at 0 for easier comparison of drift over time

    plt.figure(figsize=(10, 6))

    plt.plot(tspan, TE_Delta, color='black')

    # Formatting
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta$ Total Energy [J]")
    plt.grid(True, alpha=0.5)

    # Force scientific notation for the y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

def compare_TE(filename1,filename2):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(script_dir, filename1)
    file_path2 = os.path.join(script_dir, filename2)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data1 = np.loadtxt(file_path1, delimiter=',')
    tspan1, TE1 = data1[:, 0], data1[:, 3]
    dt1 = tspan1[1] - tspan1[0]

    data2 = np.loadtxt(file_path2, delimiter=',')
    tspan2, TE2 = data2[:, 0], data2[:, 3]
    dt2 = tspan2[1] - tspan2[0]

    TE_Delta1 = TE1 - TE1[0] # Normalize total energy to start at 0
    TE_Delta2 = TE2 - TE2[0]

    plt.figure(figsize=(10, 6))

    plt.plot(tspan1, TE_Delta1, color='black', label=fr'$\Delta t = {dt1:.2e}$')
    plt.plot(tspan2, TE_Delta2, color='red', linestyle='--', label=fr'$\Delta t = {dt2:.2e}$')

    # Formatting
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta$ Total Energy [J]")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Force scientific notation for the y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

def adams_comp_pend(filename, adams_file_path=None, plot_diff=False):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, A)
    data = np.loadtxt(file_path, delimiter=',')
    tspan = data[:, 0]

    # Determine number of bodies dynamically (1 column for time, 3 per body)
    n_bodies = (data.shape[1] - 1) // 3

    fig, axes = plt.subplots(n_bodies, 1, figsize=(10, 4 * n_bodies), sharex=True)
    if n_bodies == 1:
        axes = [axes]
        
    # Load Adams Data if provided
    adams_data = []
    if adams_file_path:
        if os.path.exists(adams_file_path):
            blocks = []
            current_block = []
            with open(adams_file_path, 'r') as f:
                for line in f:
                    clean_line = line.strip().replace(';', ' ')
                    if not clean_line:
                        continue
                    
                    if clean_line[0].isalpha():
                        if current_block:
                            blocks.append(current_block)
                            current_block = []
                        continue
                        
                    try:
                        # Attempt 1: Space-separated, with commas replaced by dots (handles Euro decimals)
                        parts = [float(x) for x in clean_line.replace(',', '.').split()]
                        if parts:
                            current_block.append(parts)
                    except ValueError:
                        try:
                            # Attempt 2: Comma-separated (Standard CSV)
                            parts = [float(x) for x in clean_line.split(',')]
                            if parts:
                                current_block.append(parts)
                        except ValueError:
                            if current_block:
                                blocks.append(current_block)
                                current_block = []
            if current_block:
                blocks.append(current_block)
            if blocks:
                min_rows = min(len(b) for b in blocks)
                truncated_blocks = [np.array(b)[:min_rows] for b in blocks]
                adams_data = np.hstack(truncated_blocks)
        else:
            print(f"Warning: Adams file not found at {adams_file_path}")

    axes_labels = ['x', 'y', 'z']

    if len(adams_data) > 0:
        t_adams = adams_data[:, 0]
        if len(tspan) != len(t_adams):
            raise ValueError(f"Timestep lengths do not match: SOA ({len(tspan)}) vs Adams ({len(t_adams)})")
        if not np.allclose(tspan, t_adams, atol=1e-5):
            raise ValueError("Timestep values do not match between SOA and Adams data.")

    for b in range(n_bodies):
        ax = axes[b]
        
        # Plot SOA data
        active_axis = 'unknown'
        max_acc = 1e-10
        soa_acc = None
        for i, axis in enumerate(axes_labels):
            col_idx = 1 + 3 * b + i
            if col_idx < data.shape[1] and np.any(np.abs(data[:, col_idx]) > 1e-10):  # Check for non-zero entries (with a small tolerance)
                ax.plot(tspan, data[:, col_idx], label=f'SOA: Body {b+1} Ang Acc {axis}', linewidth=2)
                
                max_val = np.max(np.abs(data[:, col_idx]))
                if max_val > max_acc:
                    max_acc = max_val
                    active_axis = axis
                    soa_acc = data[:, col_idx]

        # Plot Adams data
        if len(adams_data) > 0:
            t_adams = adams_data[:, 0]
            col_idx = b + 1
            if col_idx < adams_data.shape[1]:
                if np.any(np.abs(adams_data[:, col_idx]) > 1e-10):
                    adams_acc = -adams_data[:, col_idx] * (np.pi / 180.0) # Convert deg/s^2 to rad/s^2
                    label_suffix = f" {active_axis}" if active_axis != 'unknown' else ""
                    ax.plot(t_adams, adams_acc, label=f'Adams: Body {b+1} Ang Acc{label_suffix}', linestyle='--', linewidth=2)
                    
                    if plot_diff and soa_acc is not None:
                        delta = soa_acc - adams_acc
                        ax.plot(tspan, delta, label=r'$\Delta$ (SOA - Adams)', color='red', linestyle='--', linewidth=1.5)

        ax.set_ylabel(r'Ang Acc [rad/s$^2$]')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.show()

def adams_delta_pend(filename, adams_file_path=None):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, A)
    data = np.loadtxt(file_path, delimiter=',')
    tspan = data[:, 0]

    # Determine number of bodies dynamically (1 column for time, 3 per body)
    n_bodies = (data.shape[1] - 1) // 3

    fig, axes = plt.subplots(n_bodies, 1, figsize=(10, 4 * n_bodies), sharex=True)
    if n_bodies == 1:
        axes = [axes]
        
    # Load Adams Data if provided
    adams_data = []
    if adams_file_path:
        if os.path.exists(adams_file_path):
            blocks = []
            current_block = []
            with open(adams_file_path, 'r') as f:
                for line in f:
                    clean_line = line.strip().replace(';', ' ')
                    if not clean_line:
                        continue
                    
                    if clean_line[0].isalpha():
                        if current_block:
                            blocks.append(current_block)
                            current_block = []
                        continue
                        
                    try:
                        # Attempt 1: Space-separated, with commas replaced by dots (handles Euro decimals)
                        parts = [float(x) for x in clean_line.replace(',', '.').split()]
                        if parts:
                            current_block.append(parts)
                    except ValueError:
                        try:
                            # Attempt 2: Comma-separated (Standard CSV)
                            parts = [float(x) for x in clean_line.split(',')]
                            if parts:
                                current_block.append(parts)
                        except ValueError:
                            if current_block:
                                blocks.append(current_block)
                                current_block = []
            if current_block:
                blocks.append(current_block)
            if blocks:
                min_rows = min(len(b) for b in blocks)
                truncated_blocks = [np.array(b)[:min_rows] for b in blocks]
                adams_data = np.hstack(truncated_blocks)
        else:
            print(f"Warning: Adams file not found at {adams_file_path}")

    axes_labels = ['x', 'y', 'z']

    if len(adams_data) > 0:
        t_adams = adams_data[:, 0]
        if len(tspan) != len(t_adams):
            raise ValueError(f"Timestep lengths do not match: SOA ({len(tspan)}) vs Adams ({len(t_adams)})")
        if not np.allclose(tspan, t_adams, atol=1e-5):
            raise ValueError("Timestep values do not match between SOA and Adams data.")

    for b in range(n_bodies):
        ax = axes[b]
        
        soa_acc = None
        active_axis = 'unknown'
        max_acc = 1e-10
        
        # Find the active SOA axis for this body
        for i, axis in enumerate(axes_labels):
            col_idx = 1 + 3 * b + i
            if col_idx < data.shape[1]:
                max_val = np.max(np.abs(data[:, col_idx]))
                if max_val > max_acc:
                    max_acc = max_val
                    soa_acc = data[:, col_idx]
                    active_axis = axis

        # Plot delta if we have both data sources
        if soa_acc is not None and len(adams_data) > 0:
            adams_col_idx = b + 1
            if adams_col_idx < adams_data.shape[1]:
                if np.any(np.abs(adams_data[:, adams_col_idx]) > 1e-10):
                    adams_acc = -adams_data[:, adams_col_idx] * (np.pi / 180.0) # Convert deg/s^2 to rad/s^2
                    
                    # Compute error delta directly since we checked shape/values above
                    delta = soa_acc - adams_acc
                    
                    label_suffix = f" {active_axis}" if active_axis != 'unknown' else ""
                    ax.plot(tspan, delta, label=f'SOA - Adams: Body {b+1} Ang Acc{label_suffix}', color='red', linewidth=1.5)

        ax.set_ylabel(r'$\Delta$ Ang Acc [rad/s$^2$]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.show()

def plot_constraint_violation(filenames, bg_params_list, title="Constraint Violation vs. Baumgarte Parameters",savefig=False, plot_log=True):
    """
    Plots constraint violation from multiple CSV files on a single graph.

    Args:
        filenames (list of str): List of CSV filenames.
        bg_params_list (list of list/tuple): List of [alpha, beta] pairs corresponding to each file.
        title (str): The title for the plot.
        savefig (bool): Whether to save the figure to a PDF.
        plot_log (bool): Whether to include a second subplot with a logarithmic y-axis.
    """
    if len(filenames) != len(bg_params_list):
        raise ValueError("The number of filenames must match the number of Baumgarte parameter sets.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if plot_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")
        ax2 = None

    num_files = len(filenames)
    for i, (filename, bg_params) in enumerate(zip(filenames, bg_params_list)):
        file_path = os.path.join(script_dir, filename)
        
        try:
            # Assuming CSV has two columns: tspan, violation
            data = np.loadtxt(file_path, delimiter=',')
            tspan, violation = data[:, 0], data[:, 1]
            
            # Find the index where time is >= 1.0 second to trim the plot
            start_index = np.argmax(tspan >= 1.0)
            
            alpha, beta = bg_params
            
            # Make the last line dotted for emphasis
            linestyle = '--' if i == num_files - 1 else '-'
            # Plot only from the start_index onwards
            ax1.plot(tspan[start_index:], violation[start_index:], label=fr'$A={alpha}, B={beta}$', linestyle=linestyle)
            if ax2:
                ax2.plot(tspan[start_index:], violation[start_index:], label=fr'$A={alpha}, B={beta}$', linestyle=linestyle)

        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not process {filename}. Error: {e}")

    # Formatting Linear Plot
    ax1.set_xlabel("Time [s]", fontsize=14)
    ax1.set_ylabel("Constraint Violation ||Φ|| [m]", fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Force scientific notation for the linear y-axis
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Formatting Logarithmic Plot
    if ax2:
        ax2.set_xlabel("Time [s]", fontsize=14)
        ax2.set_ylabel("Constraint Violation ||Φ|| [m]", fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.tick_params(axis='both', which='major', labelsize=12)

    # Shared Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=num_files, fontsize=14, frameon=True, framealpha=0.9, labelspacing=1.2)

    if savefig == True:
        out_path = os.path.join(script_dir, "constraint_violation.pdf")
        plt.savefig(out_path)
        print(f"Figure saved as constraint_violation.pdf in {script_dir}")

    plt.show()


def plot_TE_error(filename):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data = np.loadtxt(file_path, delimiter=',')
    tspan, TE_error = data[:, 0], data[:, 1]

    plt.figure(figsize=(10, 6))

    plt.plot(tspan, TE_error, color='black')

    # Formatting
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel(r"Relative Total Energy [J]", fontsize=14)
    plt.grid(True, alpha=0.5)

    # Force scientific notation for the y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()
# 1. List the CSV files you want to compare
violation_files = [
    "constraint_violation_sprockets.csv"
]
# 2. List the corresponding Baumgarte parameters for each file
bg_sets = [
    [0.1,800]
]

# 3. Call the plotting function

plot_constraint_violation(violation_files, bg_sets, savefig=True, plot_log=False)
