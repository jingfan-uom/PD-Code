import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def temperature(R_all, Z_all, T, total_time, nsteps, dr, dz, time, mask):
    import numpy as np
    import matplotlib.pyplot as plt

    if mask is None:
        mask = np.ones_like(T, dtype=bool)
    # 只在非ghost点里找最大最小
    valid_field = np.where(mask, T, np.nan)
    T_min = np.nanmin(valid_field)
    T_max = np.nanmax(valid_field)

    plt.figure(figsize=(6, 5))
    levels = np.arange(274, 275, 0.2)
    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.0, 0.01])
    plt.ylim([0.0, 0.01])
    cbar = plt.colorbar(ctf)

    cbar.set_label(
        f"Temperature (K)\n"
        f"Computation time: {time:.1f} s\n"
        f"Δr = {dr:.4f}, Δz = {dz:.4f}\n"
        f"Min: {T_min:.2f} K\nMax: {T_max:.2f} K",
        rotation=270, labelpad=30, va='bottom'
    )

    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.tight_layout()
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


import numpy as np
import matplotlib.pyplot as plt


def plot_displacement_field(Rmat, Zmat,r_start, Ur, Uz, mask=None, title_prefix="Displacement", save=False, time=0, dr=0, dz=0):
    import numpy as np
    import matplotlib.pyplot as plt

    U_mag = np.sqrt(Ur**2 + Uz**2)
    if mask is None:
        mask = np.ones_like(Ur, dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = 'viridis'
    fields = [Ur, Uz, U_mag]
    titles = [f"{title_prefix} - Ur", f"{title_prefix} - Uz", f"{title_prefix} - |U|"]

    for ax, field, title in zip(axes, fields, titles):
        im = ax.contourf(Rmat, Zmat, field, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("r (m)")
        ax.set_ylabel("z (m)")
        ax.set_xlim(r_start, r_start+0.1)
        ax.set_ylim(0, 0.1)
        cbar = fig.colorbar(im, ax=ax)

        # 只在非ghost点里找最大最小
        valid_field = np.where(mask, field, np.nan)
        vmin, vmax = np.nanmin(valid_field), np.nanmax(valid_field)

        # 竖排写到 colorbar label
        cbar.set_label(
            f"{title}\n"
            f"Min: {vmin:.2e}, Max: {vmax:.2e}"
        )

    plt.tight_layout()
    if save:
        plt.savefig(f"{title_prefix}_field.png", dpi=300)
    plt.show()


