import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata

def temperature(R_all, Z_all, T, total_time, nsteps, dr, dz, time, mask,Lr,Lz):
    import numpy as np
    import matplotlib.pyplot as plt

    if mask is None:
        mask = np.ones_like(T, dtype=bool)
    # 只在非ghost点里找最大最小
    valid_field = np.where(mask, T, np.nan)
    T_min = np.nanmin(valid_field)
    T_max = np.nanmax(valid_field)

    plt.figure(figsize=(6, 5))
    levels = np.arange(300, 1101, 100)
    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.0, Lr])
    plt.ylim([0.0, Lz])
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

def temperature_combined_four_regions(
    R1mat, Z1mat, T1, mask1,
    R2mat, Z2mat, T2, mask2,
    R3mat, Z3mat, T3, mask3,
    R4mat, Z4mat, T4, mask4,
    Lr1,Lr4,Lz1,Lz2,
    total_time, nsteps, levels):
    """
    将四个区域的温度场绘制在一张图上（自动排除 ghost 区域）
    """

    def extract(Rmat, Zmat, T, mask):
        return Rmat[mask].flatten(), Zmat[mask].flatten(), T[mask].flatten()

    R1, Z1, T1 = extract(R1mat, Z1mat, T1, mask1)
    R2, Z2, T2 = extract(R2mat, Z2mat, T2, mask2)
    R3, Z3, T3 = extract(R3mat, Z3mat, T3, mask3)
    R4, Z4, T4 = extract(R4mat, Z4mat, T4, mask4)

    # --- 3. 合并所有区域数据 ---
    R_all = np.concatenate([R1, R2, R3, R4])
    Z_all = np.concatenate([Z1, Z2, Z3, Z4])
    T_all = np.concatenate([T1, T2, T3, T4])

    # --- 4. 构造插值网格 ---
    r_min = np.min(R_all)
    r_max = np.max(R_all)
    z_min = np.min(Z_all)
    z_max = np.max(Z_all)

    r_lin = np.linspace(r_min, r_max, 200)
    z_lin = np.linspace(z_min, z_max, 200)
    R_grid, Z_grid = np.meshgrid(r_lin, z_lin)

    T_grid = griddata(
        points=np.stack((R_all, Z_all), axis=-1),
        values=T_all,
        xi=(R_grid, Z_grid),
        method='linear'
    )

    T_min = np.nanmin(T_grid)
    T_max = np.nanmax(T_grid)

    # --- 5. 绘图 ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ctf = ax.contourf(R_grid, Z_grid, T_grid, levels=levels, cmap='jet')
    ax.set_xlim([1e-6, Lr1 + Lr4])
    ax.set_ylim([1e-6, Lz1 + Lz2])

    cbar = fig.colorbar(ctf, ax=ax)
    cbar.set_label(
        f"Temperature (K)\n"
        f"Min: {T_min:.2f} K, Max: {T_max:.2f} K"
    )

    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.tight_layout()
    plt.show()

def displacement_combined_four_regions(
    R1mat, Z1mat, Ur1, Uz1, mask1,
    R2mat, Z2mat, Ur2, Uz2, mask2,
    R3mat, Z3mat, Ur3, Uz3, mask3,
    R4mat, Z4mat, Ur4, Uz4, mask4,
    Lr1, Lr4, Lz1, Lz2,
    total_time, nsteps, levels):
    """
    将四个区域的位移场（Ur, Uz, U）分别插值后绘图，组合在一张图中，并显示 min/max。
    """

    # 提取坐标和位移数据
    R1, Z1 = R1mat[mask1].flatten(), Z1mat[mask1].flatten()
    R2, Z2 = R2mat[mask2].flatten(), Z2mat[mask2].flatten()
    R3, Z3 = R3mat[mask3].flatten(), Z3mat[mask3].flatten()
    R4, Z4 = R4mat[mask4].flatten(), Z4mat[mask4].flatten()

    Ur1_masked = Ur1[mask1.flatten()]
    Uz1_masked = Uz1[mask1.flatten()]
    Ur2_masked = Ur2[mask2.flatten()]
    Uz2_masked = Uz2[mask2.flatten()]
    Ur3_masked = Ur3[mask3.flatten()]
    Uz3_masked = Uz3[mask3.flatten()]
    Ur4_masked = Ur4[mask4.flatten()]
    Uz4_masked = Uz4[mask4.flatten()]

    # 合并
    R_all = np.concatenate([R1, R2, R3, R4])
    Z_all = np.concatenate([Z1, Z2, Z3, Z4])
    Ur_all = np.concatenate([Ur1_masked, Ur2_masked, Ur3_masked, Ur4_masked])
    Uz_all = np.concatenate([Uz1_masked, Uz2_masked, Uz3_masked, Uz4_masked])
    U_all = np.sqrt(Ur_all**2 + Uz_all**2)

    # 网格
    r_min, r_max = np.min(R_all), np.max(R_all)
    z_min, z_max = np.min(Z_all), np.max(Z_all)
    r_lin = np.linspace(r_min, r_max, 300)
    z_lin = np.linspace(z_min, z_max, 300)
    R_grid, Z_grid = np.meshgrid(r_lin, z_lin)

    points = np.stack((R_all, Z_all), axis=-1)
    Ur_grid = griddata(points, Ur_all, (R_grid, Z_grid), method='linear')
    Uz_grid = griddata(points, Uz_all, (R_grid, Z_grid), method='linear')
    U_grid  = griddata(points, U_all,  (R_grid, Z_grid), method='linear')

    # 图像范围
    r_display_max = Lr1 + Lr4
    z_display_max = Lz1 + Lz2

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def draw_subplot(ax, data_grid, title_base, unit_label):
        min_val = np.nanmin(data_grid)
        max_val = np.nanmax(data_grid)
        title = f"{title_base}\nMin: {min_val:.2e}, Max: {max_val:.2e}"
        contour = ax.contourf(R_grid, Z_grid, data_grid, levels=levels, cmap='jet')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(unit_label)
        ax.set_xlim([1e-6, r_display_max])
        ax.set_ylim([1e-6, z_display_max])
        ax.set_xlabel("r (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(title)

    draw_subplot(axes[0], Ur_grid, f"Radial displacement $U_r$\nafter {total_time:.2f}s ({nsteps} steps)", "Ur (m)")
    draw_subplot(axes[1], Uz_grid, f"Axial displacement $U_z$", "Uz (m)")
    draw_subplot(axes[2], U_grid,  f"Total displacement $|U|$", "|U| (m)")

    plt.tight_layout()
    plt.show()
