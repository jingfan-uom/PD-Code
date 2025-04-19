import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def temperature(R_all, Z_all, T, total_time, nsteps,dt):
    T_min = np.min(T)
    T_max = np.max(T)

    plt.figure(figsize=(6, 5))
    levels = np.arange(300, 400, 5)  # 到 380，右边闭区间建议设成 385

    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.0, 0.4])
    plt.ylim([0.0, 0.4])
    cbar = plt.colorbar(ctf)
    cbar.set_label(f"Temperature (K)\nMin: {T_min:.2f} K, Max: {T_max:.2f} K")
    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.show()

def temperature_fine(R_all, Z_all, T, total_time, nsteps,dt):
    T_min = np.min(T)
    T_max = np.max(T)

    plt.figure(figsize=(6, 5))
    levels = np.arange(300, 405, 5)  # 到 380，右边闭区间建议设成 385

    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.4, 0.6])
    plt.ylim([0.0, 0.4])
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
    plt.ylim(200, 400)

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


def plot_combined_temperature_contour(
        Rmat_fine: np.ndarray,
        Zmat_fine: np.ndarray,
        T_fine: np.ndarray,
        Rmat_coarse: np.ndarray,
        Zmat_coarse: np.ndarray,
        T_coarse: np.ndarray,
        nsteps: int,
        total_time: float,
        levels=np.arange(270, 385, 5)
):
    """
    将 fine 和 coarse 网格的温度合并后插值为规则网格，并绘制等高填色图。

    参数：
    - Rmat_fine, Zmat_fine: 细网格的坐标矩阵
    - T_fine: 细网格温度矩阵
    - Rmat_coarse, Zmat_coarse: 粗网格的坐标矩阵
    - T_coarse: 粗网格温度矩阵
    - nsteps: 当前模拟的步数
    - total_time: 当前模拟的累计时间（单位：秒）
    - levels: 等高线温度层级
    """

    # 合并所有坐标和温度为一维
    R_all = np.concatenate([Rmat_fine.flatten(), Rmat_coarse.flatten()])
    Z_all = np.concatenate([Zmat_fine.flatten(), Zmat_coarse.flatten()])
    T_all = np.concatenate([T_fine.flatten(), T_coarse.flatten()])

    # 创建规则网格以便插值
    r_lin = np.linspace(np.min(R_all), np.max(R_all), 300)
    z_lin = np.linspace(np.min(Z_all), np.max(Z_all), 300)
    R_grid, Z_grid = np.meshgrid(r_lin, z_lin)

    # 插值到规则网格
    T_grid = griddata(
        points=np.stack((R_all, Z_all), axis=-1),
        values=T_all,
        xi=(R_grid, Z_grid),
        method='linear'
    )

    # 提取有效温度范围（排除插值空洞）
    T_min = np.nanmin(T_grid)
    T_max = np.nanmax(T_grid)

    # 绘图
    plt.figure(figsize=(6, 5))
    ctf = plt.contourf(R_grid, Z_grid, T_grid, levels=levels, cmap='jet')
    plt.xlim([np.min(R_all), np.max(R_all)])
    plt.ylim([np.min(Z_all), np.max(Z_all)])
    cbar = plt.colorbar(ctf)
    cbar.set_label(f"Temperature (K)\nMin: {T_min:.2f} K, Max: {T_max:.2f} K")
    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.tight_layout()
    plt.show()
