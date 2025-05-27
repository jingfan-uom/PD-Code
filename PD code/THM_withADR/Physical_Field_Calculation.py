
from scipy.spatial import cKDTree
import numpy as np

def compute_direction_matrix(x, y, ux, uz, horizon_mask): 
    """
    Compute updated direction matrix based on current relative positions: (x' + u') - (x + u)

    Inputs:
    - x, y: coordinate vectors (N,)
    - ux, uz: displacement vectors (N,)
    - horizon_mask: (N, N) interaction mask

    Outputs:
    - dir_x, dir_z: direction unit vectors (N, N)
    """
    # Current relative positions: (x' + u') - (x + u)
    dx_eff = (x[None, :] + ux[None, :]) - (x[:, None] + ux[:, None])
    dz_eff = (y[None, :] + uz[None, :]) - (y[:, None] + uz[:, None])

    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)

    # 只对 horizon_mask == True 的元素计算，其他点直接设为 0
    dir_x = np.zeros_like(dx_eff)
    dir_z = np.zeros_like(dz_eff)

    dir_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    dir_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    return dir_x, dir_z



def compute_s_matrix(x_flat, y_flat, Ux, Uz, horizon_mask, distance_matrix):
    """
    Compute the elongation matrix s_matrix (N, N) using 2D grid input and horizon_mask.

    Parameters:
        X, Y: original mesh coordinates
        Ux, Uz: displacement fields at corresponding points
        horizon_mask: boolean array of shape
        distance_matrix :Initial lengths

    Returns:
        s_matrix: elongation matrix of shape (N, N)
    """
    # Deformed coordinates
    x_def = (Ux + x_flat)
    y_def = (Uz + y_flat)
    # Original coordinates

    # Deformed lengths
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Elongation computation
    s_matrix = np.zeros_like(distance_matrix)
    s_matrix[horizon_mask] = (L1[horizon_mask] - distance_matrix[horizon_mask]) / distance_matrix[horizon_mask]

    return s_matrix


def compute_delta_temperature(T_grid,  Tpre_avg):
    """
    Compute the average temperature matrix T_avg (N, N), and optionally return the delta
    compared to the previous time step average temperature.

    Parameters:
        T_grid: 2D temperature field array (Ny, Nx)

        T_prev_avg: previous time step average temperature matrix, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """
    T_i = T_grid [:, np.newaxis]  # shape (N, 1)
    T_j = T_grid [np.newaxis, :]  # shape (1, N)
    Tcurr_avg = 0.5 * (T_i + T_j) - Tpre_avg# shape (N, N)

    return Tcurr_avg



import numpy as np
from scipy.spatial import cKDTree

def shrink_Tth_by_matching_coords( Rmat_m, Zmat_m, Rmat_th, Zmat_th):
    """
    删除 Tth 中与 Rmat_m/Zmat_m 坐标不对应的点，仅保留对应点。

    参数：
        Tth        - 一维数组，展开的 T_th（长度 M）
        Rmat_m     - 目标区域 r 坐标（2D）
        Zmat_m     - 目标区域 z 坐标（2D）
        Rmat_th    - Tth 的 r 坐标（2D）
        Zmat_th    - Tth 的 z 坐标（2D）

    返回：
        Tth_shrunk - 删除不相关坐标后的 Tth 子数组（一维，长度 = T_m.size）
    """
    # 所有原始坐标（参考网格）
    points_th = np.column_stack((Rmat_th.ravel(), Zmat_th.ravel()))
    # 目标坐标（只保留这些）
    points_m = np.column_stack((Rmat_m.ravel(), Zmat_m.ravel()))

    # 构建 KDTree，找到目标区域的索引
    tree = cKDTree(points_th)
    _, indices = tree.query(points_m)
    return indices


def filter_array_by_indices_keep_only(Tarr, indices):

    keep_mask = np.zeros_like(Tarr, dtype=bool)
    keep_mask[indices] = True

    # 删除不需要的索引
    Tth_shrunk = Tarr[keep_mask]

    return Tth_shrunk
