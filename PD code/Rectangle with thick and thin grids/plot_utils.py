
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import MultipleLocator

def temperature(R_all, Z_all, T, total_time, nsteps,dt):
    T_min = np.min(T)
    T_max = np.max(T)
    print("Z_all range:", np.min(Z_all), np.max(Z_all))
    plt.figure(figsize=(6, 5))
    levels = np.arange(270 ,380, 5)  
    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.00, 0.05])
    plt.ylim([0., 0.1])
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
    plt.xlim(0., .5)
    plt.ylim(1, 400)

    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(25))
    plt.tick_params(axis='y', which='minor', length=4, color='gray')
    plt.tick_params(axis='y', which='major', length=7)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def temperature_combined(
    Rmat_coarse, Zmat_coarse, T_coarse,
    Rmat_fine, Zmat_fine, T_fine,
    total_time, time1, nsteps,
    levels,
    r_start_coarse, Lr_coarse, dr_coarse, dz_coarse, ghost_nodes_r_coarse,
    r_start_fine, Lr_fine, dr_fine, dz_fine,
    ghost_nodes_r_fine):
    """
    Combine the temperature fields of coarse and fine grids and plot them in one figure (physical domain only).
    The physical region mask is calculated automatically — no need to pass it manually.
    """

    # --- 1. Compute physical region boundaries (including ghost layers) ---
    r_phys_min_fine = np.min(Rmat_fine + ghost_nodes_r_fine * dr_fine)
    r_phys_max_fine = np.max(Rmat_fine)
    z_phys_min_fine = np.min(Zmat_fine)
    z_phys_max_fine = np.max(Zmat_fine)

    r_phys_min_coarse = np.min(Rmat_coarse)
    r_phys_max_coarse = np.max(Rmat_coarse - ghost_nodes_r_coarse * dr_coarse)
    z_phys_min_coarse = np.min(Zmat_coarse)
    z_phys_max_coarse = np.max(Zmat_coarse)

    # --- 2. Automatically generate masks for physical regions ---
    fine_mask_phys = (Rmat_fine >= r_phys_min_fine) & (Rmat_fine <= r_phys_max_fine) & \
                     (Zmat_fine >= z_phys_min_fine) & (Zmat_fine <= z_phys_max_fine)

    coarse_mask_phys = (Rmat_coarse >= r_phys_min_coarse) & (Rmat_coarse <= r_phys_max_coarse) & \
                       (Zmat_coarse >= z_phys_min_coarse) & (Zmat_coarse <= z_phys_max_coarse)

    # --- 3. Combine coordinates and temperatures from physical regions ---
    R_f = Rmat_fine[fine_mask_phys].flatten()
    Z_f = Zmat_fine[fine_mask_phys].flatten()
    T_f = T_fine[fine_mask_phys].flatten()

    R_c = Rmat_coarse[coarse_mask_phys].flatten()
    Z_c = Zmat_coarse[coarse_mask_phys].flatten()
    T_c = T_coarse[coarse_mask_phys].flatten()

    R_all = np.concatenate([R_f, R_c])
    Z_all = np.concatenate([Z_f, Z_c])
    T_all = np.concatenate([T_f, T_c])

    # --- 4. Interpolate onto a regular grid ---
    r_min = min(r_phys_min_fine, r_phys_min_coarse)
    r_max = max(r_phys_max_fine, r_phys_max_coarse)
    z_min = min(z_phys_min_fine, z_phys_min_coarse)
    z_max = max(z_phys_max_fine, z_phys_max_coarse)

    r_lin = np.linspace(r_min, r_max, 300)
    z_lin = np.linspace(z_min, z_max, 300)
    R_grid, Z_grid = np.meshgrid(r_lin, z_lin)

    T_grid = griddata(
        points=np.stack((R_all, Z_all), axis=-1),
        values=T_all,
        xi=(R_grid, Z_grid),
        method='linear'
    )

    T_min = np.nanmin(T_grid)
    T_max = np.nanmax(T_grid)

    # --- 5. Plotting ---
    # Multi-line colorbar label shows temperature, time, and grid info
    fig, ax = plt.subplots(figsize=(7, 5))
    ctf = ax.contourf(R_grid, Z_grid, T_grid, levels=levels, cmap='jet')
    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 0.1])

    cbar = fig.colorbar(ctf, ax=ax)
    cbar.set_label(
        f"Temperature (K)\n"
        f"Computation time: {time1:.1f} s\n"
        f"Δr_fine = {dr_fine:.4f}, Δz_fine = {dz_fine:.4f}\n"
        f"Δr_coarse = {dr_coarse:.4f}, Δz_coarse = {dz_coarse:.4f}\n"
        f"Min: {T_min:.2f} K, Max: {T_max:.2f} K"
    )

    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.tight_layout()
    plt.show()

