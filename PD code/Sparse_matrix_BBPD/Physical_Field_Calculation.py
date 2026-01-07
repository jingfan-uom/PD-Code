from scipy.spatial import cKDTree
import numpy as np

def compute_direction_matrix(x, y, ux, uz, indptr, indices, eps=1e-15):
    """
    CSR 版本：只对有效邻居边 (i->j) 计算方向单位向量

    Inputs:
    - x, y: (N,)
    - ux, uz: (N,)
    - indptr: (N+1,)
    - indices: (nnz,)
    - eps: 防止除零

    Outputs:
    - dir_x_flat, dir_z_flat: (nnz,) 与 indices 对齐
      对应边 p：i 是该边所在行，j = indices[p]
      dir = ( (xj+uxj)-(xi+uxi), (yj+uzj)-(yi+uzi) ) / dist_eff
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ux = np.asarray(ux)
    uz = np.asarray(uz)

    N = len(indptr) - 1

    # 生成每条边对应的行号 i（长度 nnz）
    edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))

    # 变形后相对位移（按边）
    dx_eff = (x[indices] + ux[indices]) - (x[edge_i] + ux[edge_i])
    dz_eff = (y[indices] + uz[indices]) - (y[edge_i] + uz[edge_i])

    dist_eff = np.sqrt(dx_eff * dx_eff + dz_eff * dz_eff)

    inv = 1.0 / np.maximum(dist_eff, eps)
    dir_x_flat = dx_eff * inv
    dir_z_flat = dz_eff * inv

    return dir_x_flat, dir_z_flat


def compute_s_matrix(x_flat, y_flat, Ux, Uz, indptr, indices, dist, eps=1e-15):

    N = len(indptr) - 1
    # 每条边属于哪一行 i（长度 nnz）
    edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
    # 变形后坐标
    x_def = x_flat + Ux
    y_def = y_flat + Uz
    # 变形后边向量（按边）
    dx1 = x_def[indices] - x_def[edge_i]
    dz1 = y_def[indices] - y_def[edge_i]
    # 变形后键长 L1（按边）
    L1 = np.sqrt(dx1 * dx1 + dz1 * dz1)
    # 初始键长 L0（按边）
    L0 = dist
    # 相对伸长率（按边）
    s_flat = (L1 - L0) / np.maximum(L0, eps)

    return s_flat



def compute_delta_temperature(T_grid, Tpre_avg):
    """
    Compute the average temperature matrix and the difference from the previous step.

    Parameters:
        T_grid: current temperature field (1D or flattened)
        Tpre_avg: average temperature matrix from previous time step, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """
    T_i = T_grid[:, np.newaxis]  # shape (N, 1)
    T_j = T_grid[np.newaxis, :]  # shape (1, N)
    Tcurr_avg = 0.5 * (T_i + T_j) - Tpre_avg  # shape (N, N)

    return Tcurr_avg



def shrink_Tth_by_matching_coords(Rmat_m, Zmat_m, Rmat_th, Zmat_th):
    """
    Remove Tth values whose coordinates do not match with Rmat_m / Zmat_m;
    only retain matching coordinate points.

    Parameters:
        Rmat_m     - target region r coordinates (2D)
        Zmat_m     - target region z coordinates (2D)
        Rmat_th    - Tth's corresponding r coordinates (2D)
        Zmat_th    - Tth's corresponding z coordinates (2D)

    Returns:
        indices    - indices of matched coordinates for filtering Tth
    """
    # All original coordinates (reference grid)
    points_th = np.column_stack((Rmat_th.ravel(), Zmat_th.ravel()))
    # Target coordinates (to keep)
    points_m = np.column_stack((Rmat_m.ravel(), Zmat_m.ravel()))

    # Use KDTree to find index mapping from target to reference
    tree = cKDTree(points_th)
    _, indices = tree.query(points_m)
    return indices


def filter_array_by_indices_keep_only(Tarr, indices):
    """
    Keep only elements in Tarr whose indices are in the given list.

    Parameters:
        Tarr: 1D array to filter
        indices: indices to retain

    Returns:
        Filtered 1D array with only selected elements
    """
    keep_mask = np.zeros_like(Tarr, dtype=bool)
    keep_mask[indices] = True

    # Filter using mask
    Tth_shrunk = Tarr[keep_mask]

    return Tth_shrunk

