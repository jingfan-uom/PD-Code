import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def plot_temperature_contour_in_circle(phys_coords_list,dr, T_phys, radius, title='Temperature Contour', cmap='jet',
                                       levels=20):
    """
    Plot contour lines of the temperature field only within the semicircular region (r, z).
    - radius: Radius of the circle (units should be consistent with the coordinate units)
    """
    # 1. Merge the physical point coordinates of all regions
    all_coords = np.vstack([arr[:, :2] for arr in phys_coords_list])  # shape: (N, 2)

    # 2. Merge all temperatures
    if isinstance(T_phys, dict):
        all_temps = np.concatenate([T_phys[i] for i in range(len(phys_coords_list))])
    elif isinstance(T_phys, list):
        all_temps = np.concatenate(T_phys)
    else:
        raise TypeError("T_phys 必须是 list 或 dict 类型")

    # 3. Create a rule grid
    r_vals = all_coords[:, 0]
    z_vals = all_coords[:, 1]

    r_lin = np.linspace(np.min(r_vals), np.max(r_vals), 300)
    z_lin = np.linspace(np.min(z_vals), np.max(z_vals), 300)
    r_grid, z_grid = np.meshgrid(r_lin, z_lin)

    # 4. Interpolation
    T_grid = griddata(all_coords, all_temps, (r_grid, z_grid), method='linear')

    # 5. Mask processing (only retain the semicircular area)
    mask_circle = (r_grid ** 2 + (z_grid - radius) ** 2 <= radius ** 2 ) & \
                  (r_grid > 0) & (z_grid > 0) & (z_grid < 2 * radius)

    T_grid[~mask_circle] = np.nan  # Set the area outside the circle to NaN and do not draw it.

    # 6. Set contour line levels
    vmin = np.nanmin(T_grid)
    vmax = np.nanmax(T_grid)
    level_values = np.linspace(vmin, vmax, levels)

    # 7. Drawing
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(r_grid, z_grid, T_grid, levels=level_values, cmap=cmap)

    cbar = plt.colorbar(contour)
    cbar.set_label(
        f"Temperature (K)\nMin: {vmin:.2f} K\nMax: {vmax:.2f} K",
        rotation=270, labelpad=30, va='bottom'
    )

    plt.xlabel('r (m)')
    plt.ylabel('z (m)')
    plt.title(title)
    plt.xlim([dr/2, radius])
    plt.ylim([0, 2 * radius])
    plt.tight_layout()
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

def plot_displacement_contours_in_circle(
    phys_coords_list_m,
    U_phys,
    radius,
    titles=('Ur Contour', 'Uz Contour', '|U| Contour'),
    cmaps=('jet', 'jet', 'jet'),
    levels=20,
    grid_n=800
):
    """
    Plot contour maps of Ur, Uz, and |U| within a semicircular domain.

    Parameters
    ----------
    phys_coords_list_m : list of arrays
        Each array is (Ni, >=2). First two columns are (r, z) of physical points per region.
    U_phys : dict or list
        If dict: U_phys[i] = {"Ur": array, "Uz": array, "Umag": optional}
        If list: list of tuples (Ur_i, Uz_i) or (Ur_i, Uz_i, Umag_i) following region order.
    radius : float
        Radius of the semicircle (same unit as coords).
    titles : 3-tuple of str
        Titles for the three subplots.
    cmaps : 3-tuple of str
        Colormaps for the three subplots.
    levels : int
        Number of contour levels.
    grid_n : int
        Resolution of interpolation grid per axis (larger = finer).
    """

    # 1) Merge physical coordinates of all regions
    all_coords = np.vstack([arr[:, :2] for arr in phys_coords_list_m])  # (N, 2)
    r_vals = all_coords[:, 0]
    z_vals = all_coords[:, 1]

    # 2) Merge Ur, Uz, Umag from U_phys
    if isinstance(U_phys, dict):
        Ur_all = np.concatenate([U_phys[i]["Ur"] for i in range(len(phys_coords_list_m))])
        Uz_all = np.concatenate([U_phys[i]["Uz"] for i in range(len(phys_coords_list_m))])
        if all(("Umag" in U_phys[i]) for i in range(len(phys_coords_list_m))):
            Umag_all = np.concatenate([U_phys[i]["Umag"] for i in range(len(phys_coords_list_m))])
        else:
            Umag_all = np.sqrt(Ur_all**2 + Uz_all**2)
    elif isinstance(U_phys, list):
        # Each item: (Ur_i, Uz_i) or (Ur_i, Uz_i, Umag_i)
        Ur_all = np.concatenate([item[0] for item in U_phys])
        Uz_all = np.concatenate([item[1] for item in U_phys])
        if len(U_phys[0]) >= 3:
            Umag_all = np.concatenate([item[2] for item in U_phys])
        else:
            Umag_all = np.sqrt(Ur_all**2 + Uz_all**2)
    else:
        raise TypeError("U_phys must be a dict or list")

    # 3) Build regular grid
    r_lin = np.linspace(np.min(r_vals), np.max(r_vals), grid_n)
    z_lin = np.linspace(np.min(z_vals), np.max(z_vals), grid_n)
    r_grid, z_grid = np.meshgrid(r_lin, z_lin)

    # 4) Interpolate Ur, Uz, Umag onto the grid
    Ur_grid = griddata(all_coords, Ur_all, (r_grid, z_grid), method='linear')
    Uz_grid = griddata(all_coords, Uz_all, (r_grid, z_grid), method='linear')
    Umag_grid = griddata(all_coords, Umag_all, (r_grid, z_grid), method='linear')

    # 5) Mask: keep only semicircle (center at (0, radius), radius=radius)
    mask_circle = (
        (r_grid**2 + (z_grid - radius)**2 <= radius**2) &
        (r_grid >= 0) & (z_grid >= 0) & (z_grid <= 2 * radius)
    )

    for G in (Ur_grid, Uz_grid, Umag_grid):
        G[~mask_circle] = np.nan

    # 6) Levels for each field
    def level_array(G):
        vmin = np.nanmin(G)
        vmax = np.nanmax(G)
        return np.linspace(vmin, vmax, levels), vmin, vmax

    levels_Ur, vmin_Ur, vmax_Ur = level_array(Ur_grid)
    levels_Uz, vmin_Uz, vmax_Uz = level_array(Uz_grid)
    levels_Um, vmin_Um, vmax_Um = level_array(Umag_grid)

    # 7) Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    grids = [Ur_grid, Uz_grid, Umag_grid]
    levels_list = [levels_Ur, levels_Uz, levels_Um]
    vmins = [vmin_Ur, vmin_Uz, vmin_Um]
    vmaxs = [vmax_Ur, vmax_Uz, vmax_Um]

    for ax, G, lv, ttl, cmap, vmin, vmax in zip(axes, grids, levels_list, titles, cmaps, vmins, vmaxs):
        cf = ax.contourf(r_grid, z_grid, G, levels=lv, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(f"Min: {vmin:.3e}\nMax: {vmax:.3e}", rotation=270, labelpad=10, va='bottom')
        ax.set_title(ttl)
        ax.set_xlabel('r (m)')
        ax.set_ylabel('z (m)')
        ax.set_xlim([0, radius])
        ax.set_ylim([0, 2 * radius])

    plt.show()

def temperature_contour(T, coords,  total_time, nsteps, dr, r):

    Nr_plot = Nz_plot = int(r / dr) * 50
    r_vals = np.linspace(0, r, Nr_plot)
    z_vals = np.linspace(0, 2 * r, Nz_plot)
    R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

    # 插值
    T_grid = griddata(coords, T, (R_grid, Z_grid), method='linear')

    # 温度范围
    T_min = np.nanmin(T)
    T_max = np.nanmax(T)
    levels = np.linspace(np.nanmin(280), np.nanmax(400), num=9)

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
