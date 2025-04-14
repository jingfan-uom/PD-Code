import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def temperature(R_all, Z_all, T, total_time, nsteps,dt):
    T_min = np.min(T)
    T_max = np.max(T)

    plt.figure(figsize=(6, 5))
    levels = np.arange(270, 380, 5)  # 到 380，右边闭区间建议设成 385

    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.0, .1])
    plt.ylim([0.0, 0.1])
    cbar = plt.colorbar(ctf)
    cbar.set_label(f"Temperature (K)\nMin: {T_min:.2f} K, Max: {T_max:.2f} K")
    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.show()

def plot_z_profile(T_record, z_all, r_all, save_times_hours):
    """
    Plot temperature evolution over time at a fixed z slice (z = 0.4 m).

    Parameters:
    -----------
    T_record : list of 2D arrays
        Temperature snapshots at different times.
    z_all : 1D array
        z coordinates.
    r_all : 1D array
        r coordinates.
    save_times_hours : list of floats
        Simulation times (in hours) corresponding to T_record.
    """
    z_target = 0.4
    z_index = np.argmin(np.abs(z_all - z_target))

    plt.figure(figsize=(8, 5))
    for T, t_hour in zip(T_record, save_times_hours):
        plt.plot(r_all, T[z_index, :], label=f"{t_hour} h")

    plt.xlabel("r (m)")
    plt.ylabel("Temperature at z = 0.4 m (K)")
    plt.title("z = 0.4 m Cross-sectional Temperature Evolution")
    plt.xlim(0.2, 1.0)
    plt.ylim(200, 500)

    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(25))
    plt.tick_params(axis='y', which='minor', length=4, color='gray')
    plt.tick_params(axis='y', which='major', length=7)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_1d_temperature(r_all, T, time_seconds, save_dir=None):
    """
    Plot 1D temperature profile (r-direction only), with optional saving.

    Parameters:
    -----------
    r_all : 1D array
        Radial coordinate (including ghost nodes if any).
    T : 1D or 2D array
        Temperature field. If 2D, it will be flattened.
    time_seconds : float
        Simulation time in seconds (for labeling).
    save_dir : str, optional
        If provided, saves the plot to the specified directory.
    """
    T = T.flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(r_all, T, 'r-', linewidth=2)
    plt.xlabel("r (m)")
    plt.ylabel("Temperature (K)")
    plt.xlim(0.05, 1.45)
    plt.ylim(300, 500)
    plt.title(f"1D Temperature Distribution at t = {time_seconds/3600:.2f} h")
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"T_1D_{int(time_seconds)}s.png"
        plt.savefig(os.path.join(save_dir, fname))
        print(f"[plot] Saved 1D temperature plot to {os.path.join(save_dir, fname)}")
    else:
        plt.show()

    plt.close()

