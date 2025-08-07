import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata

def temperature_contour(T, coords, mask_circle, total_time, nsteps, dr, r):


    coords_valid = coords[mask_circle]
    T_valid = T[mask_circle]


    Nr_plot = Nz_plot = int(r / dr)*10
    r_vals = np.linspace(0, r, Nr_plot)
    z_vals = np.linspace(0, 2 * r, Nz_plot)
    R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

    # 插值
    T_grid = griddata(coords_valid, T_valid, (R_grid, Z_grid), method='linear')

    # 温度范围
    T_min = np.nanmin(T_valid)
    T_max = np.nanmax(T_valid)
    levels = np.linspace(np.nanmin(T_valid -1), np.nanmax(T_valid), num=9)

    # 绘图
    plt.figure(figsize=(6, 5))
    ctf = plt.contourf(R_grid, Z_grid, T_grid, levels=levels, cmap='jet')
    plt.xlim([0.0, r])
    plt.ylim([0.0, 2 * r])
    cbar = plt.colorbar(ctf)

    cbar.set_label(
        f"Temperature (K)\n"
        f"Computation time: {nsteps:.0f} steps in {total_time:.1f} s\n"
        f"Min: {T_min:.2f} K\nMax: {T_max:.2f} K",
        rotation=270, labelpad=30, va='bottom'
    )

    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.tight_layout()
    plt.show()




def plot_displacement_field(Rmat, Zmat, Ur, Uz, mask, Lr, Lz, title_prefix="Displacement", save=False):
    """
    Plot Ur, Uz, and |U| fields using contourf, only considering the non-ghost (masked) area for min/max.
    """

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
        ax.set_xlim(0, Lr)
        ax.set_ylim(0, Lz)
        cbar = fig.colorbar(im, ax=ax)

        # Only search for the minimum and maximum within non-ghost points
        valid_field = np.where(mask, field, np.nan)
        vmin, vmax = np.nanmin(valid_field), np.nanmax(valid_field)

        # Display as a vertical text on the colorbar label
        cbar.set_label(
            f"{title}\n"
            f"Min: {vmin:.2e}, Max: {vmax:.2e}"
        )

    plt.tight_layout()
    if save:
        plt.savefig(f"{title_prefix}_field.png", dpi=300)
    plt.show()



def plot_mu_field(Rmat, Zmat, mu, mask, Lr, Lz, title_prefix, save, filename):
    """
    Draw contour lines for mu, counting only the masked area.
    """
    if mask is None:
        mask = np.ones_like(mu, dtype=bool)
    masked_mu = np.where(mask, mu, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = np.arange(0, 1.01, 0.05)  # Finer color gradation
    cmap = plt.get_cmap('turbo')  # Use 'jet' or 'turbo' for a colormap closer to the target effect

    im = ax.contourf(Rmat, Zmat, masked_mu, levels=levels, cmap=cmap, alpha=0.95)
    im.set_clim(0, 1)

    ax.set_title(f"{title_prefix} ($\mu$)", fontsize=13)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(0, Lr)
    ax.set_ylim(0, Lz)

    cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(0, 1, 11))
    cbar.set_label(f"{title_prefix} ($\mu$)", fontsize=12)

    plt.tight_layout()
    if save:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.savefig("mu_field.png")
    plt.close()
    plt.show()
