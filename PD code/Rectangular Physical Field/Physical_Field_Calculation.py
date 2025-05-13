
import numpy as np


def compute_direction_matrix(X, Y, horizon_mask):
    """
    基于小变形假设，计算方向矩阵。

    输入：
    - X, Y: 坐标网格 (Ny, Nx)
    - horizon_mask: 掩码矩阵 (N, N)，表示粒子对之间是否在作用范围内

    输出：
    - dir_x: r方向的单位向量（余弦），形状为 (N, N)
    - dir_z: z方向的单位向量（正弦），形状为 (N, N)
    """

    x = X.flatten()  # (N,)
    y = Y.flatten()  # (N,)

    dx = x[None, :] - x[:, None]  # (N, N)
    dz = y[None, :] - y[:, None]  # (N, N)

    distance = np.sqrt(dx ** 2 + dz ** 2)
    distance[~horizon_mask] = 1.0  # 避免除以0

    dir_x = np.where(horizon_mask, dx / distance, 0.0)
    dir_z = np.where(horizon_mask, dz / distance, 0.0)

    return dir_x, dir_z

def compute_s_matrix(X, Y, Ux, Uz,horizon_mask):
    """
    计算伸长率矩阵 s_matrix (N, N)，使用二维网格输入并带 horizon_mask。

    参数:
        X, Y: 原始网格坐标 (Ny, Nx)
        Ux, Uz: 对应点位移 (Ny, Nx)
        horizon_mask: shape = (N, N)，布尔矩阵

    返回:
        s_matrix: shape = (N, N)，伸长率矩阵
    """
    # 当前位移后的坐标

    x_def = (Ux + X).flatten()
    y_def = (Uz + Y).flatten()
    # 原始位置
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # 初始长度
    dx0 = x_flat[None, :] - x_flat[:, None]  # xj - xi
    dz0 = y_flat[None, :] - y_flat[:, None]  # zj - zi
    L0 = np.sqrt(dx0 ** 2 + dz0 ** 2)

    # 当前长度
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # 避免除以 0
    L0[L0 == 0] = 1.0

    # 伸长率计算
    s_matrix = np.zeros_like(L0)
    s_matrix[horizon_mask] = (L1[horizon_mask] - L0[horizon_mask]) / L0[horizon_mask]

    return s_matrix


def compute_delta_temperature(T_grid, horizon_mask, T_prev_avg):
    """
    计算平均温度矩阵 T_avg (N, N)，并可选地返回与上一时刻平均温度的差值。

    参数:
        T_grid: 温度场二维数组 (Ny, Nx)
        horizon_mask: bool 数组，shape = (N, N)
        T_prev_avg: 可选，上一时刻的平均温度矩阵，shape = (N, N)

    """

    T_flat = T_grid.flatten()  # (N,)
    T_i = T_flat[:, np.newaxis]  # shape (N, 1)
    T_j = T_flat[np.newaxis, :]  # shape (1, N)
    T_avg = 0.5 * (T_i + T_j)  # shape (N, N)

    # 非作用域点置零
    T_avg[~horizon_mask] = 0.0
    T_prev_avg[~horizon_mask] = 0.0
    T_delta = T_avg - T_prev_avg

    return T_delta

def compute_velocity_third_step(Vr_half, Vz_half, Ar_next, Az_next, dt):
    """
    实现公式14中的第三步：根据下一时刻加速度，更新速度到时间 n+1。

    参数：
        Vr_half, Vz_half: 中间时刻 (n+1/2) 的速度 (r, z方向)
        Ar_next, Az_next: 下一时刻的加速度 (r, z方向)
        dt: 时间步长

    返回：
        Vr_new, Vz_new: 完整更新后的速度 (r, z方向)
    """
    Vr_new = Vr_half + 0.5 * dt * Ar_next
    Vz_new = Vz_half + 0.5 * dt * Az_next
    return Vr_new, Vz_new