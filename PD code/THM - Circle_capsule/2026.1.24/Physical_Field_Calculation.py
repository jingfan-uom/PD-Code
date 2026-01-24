from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

def compute_direction_edges_csr_numba(coords, edge_i, edge_j, dist, eps=1e-30):
    Nnz = dist.size
    dir_r = np.zeros(Nnz, dtype=np.float64)
    dir_z = np.zeros(Nnz, dtype=np.float64)

    x = coords[:, 0]
    z = coords[:, 1]

    for p in prange(Nnz):
        i = edge_i[p]
        j = edge_j[p]
        dij = dist[p]
        if dij > eps and i != j:
            dir_r[p] = (x[j] - x[i]) / dij
            dir_z[p] = (z[j] - z[i]) / dij
        else:
            dir_r[p] = 0.0
            dir_z[p] = 0.0

    return dir_r, dir_z



def compute_Tensor_product(x, y, horizon_mask):

    # Current relative positions after deformation: x '- x
    dx_eff = x[None, :] - x[:, None]
    dz_eff = y[None, :] - y[:, None]
    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)
    # Only compute values where horizon_mask is True; set others to zero
    n_x = np.zeros_like(dx_eff)
    n_z = np.zeros_like(dz_eff)

    n_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    n_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    N = n_x.shape[0]
    Cxx = np.zeros((N, N))
    Cxz = np.zeros((N, N))
    Czx = np.zeros((N, N))
    Czz = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # 只对有效（horizon_mask为True）的键计算
            if horizon_mask[i, j]:
                n_vec = np.array([n_x[i, j], n_z[i, j]])  # 组成二维单位向量
                n_outer = np.outer(n_vec, n_vec)  # 2x2张量积

                Cxx[i, j] = n_outer[0, 0]
                Cxz[i, j] = n_outer[0, 1]
                Czx[i, j] = n_outer[1, 0]
                Czz[i, j] = n_outer[1, 1]

    return Cxx, Cxz, Czx, Czz

def compute_s_matrix(coords, Ux, Uz, horizon_mask):
    """
    Compute elongation matrix s_matrix (N, N) using vectorized matrix operations.

    Parameters:
        coords: (N, 2) array of original coordinates (x, y)
        Ux, Uz: displacement arrays (N,)
        horizon_mask: boolean array of shape (N, N), True if bond (i, j) is valid

    Returns:
        s_matrix: elongation matrix (N, N)
    """
    # Original coordinates
    x_flat = coords[:, 0]
    y_flat = coords[:, 1]

    # Deformed coordinates
    x_def = x_flat + Ux
    y_def = y_flat + Uz

    # Initial lengths L0
    dx0 = x_flat[None, :] - x_flat[:, None]
    dz0 = y_flat[None, :] - y_flat[:, None]
    L0 = np.sqrt(dx0 ** 2 + dz0 ** 2)

    # Deformed lengths L1
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Elongation computation (vectorized)
    s_matrix = np.zeros_like(L0)
    mask = horizon_mask & (L0 > 0)
    s_matrix[mask] = (L1[mask] - L0[mask]) / L0[mask]

    return s_matrix

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
    Tcurr_avg = 0.5 * (T_i + T_j)- Tpre_avg  # shape (N, N)

    return Tcurr_avg


def shrink_Tth_by_matching_coords(coords_m, coords_t):

    coords_m = np.asarray(coords_m)
    coords_t = np.asarray(coords_t)

    # If there are three columns, take the first two columns to match
    if coords_m.shape[1] >= 3:
        coords_m = coords_m[:, :2]
    if coords_t.shape[1] >= 3:
        coords_t = coords_t[:, :2]

    # Nearest Neighbor Matching
    tree = cKDTree(coords_t)
    _, indices = tree.query(coords_m, k=1)

    return indices.astype(np.int64)



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



def update_mu_by_failure(mu, Relative_elongation, s0):

    failure_mask = (mu == 1) & (Relative_elongation >= s0)
    mu_new = mu.copy()
    mu_new[failure_mask] = 0

    return mu_new


def find_inner_surface_layer(coords_phys_m, r, dshell, dr):

    coords = np.asarray(coords_phys_m)

    r_arr = coords[:, 0]
    z_arr = coords[:, 1]

    rc, zc = 0.0, float(r)
    dist = np.sqrt((r_arr - rc)**2 + (z_arr - zc)**2)

    r_inner = float(r) - float(dshell)

    mask = (dist >= r_inner) & (dist <= r_inner + 3 * dr/4 )
    idx_layer = np.where(mask)[0]

    # 计算单位向量（从点指向圆心）
    vr = - rc + r_arr[idx_layer]
    vz = - zc + z_arr[idx_layer]
    norm = np.hypot(vr, vz)
    unit = np.zeros((idx_layer.size, 2), dtype=coords.dtype)
    nz = norm > 1e-12
    unit[nz, 0] = vr[nz] / norm[nz]
    unit[nz, 1] = vz[nz] / norm[nz]

    return {
        "indices": idx_layer,
        "unit_to_center": unit
    }


def compute_melt_and_thermal_expansion(
    T,                    # (N,) 温度场 (K)
    mask_core,            # (N,) bool，核心/PCM区域掩码
    rho_s, rho_l,         # 固/液密度 (kg/m^3)
    Ts, Tl,               # 固/液相线温度 (K)
    cell_volume,          # 每个点的“体积/面积” (与维度一致, 标量)
    beta_s,           # 固相体膨胀系数 (1/K)，若不考虑热膨胀可置0
    beta_l,           # 液相体膨胀系数 (1/K)
    T_ref,           # 热膨胀参考温度 (K)，默认取 Ts
):

    T = np.asarray(T, dtype=float)
    core = np.asarray(mask_core, dtype=bool)
    T_core = T[core]

    dT_phase = Tl - Ts
    # --- 熔化分数 alpha ∈ [0,1]：0=固，1=液，线性糊区 ---
    alpha = (T_core - Ts) / dT_phase
    alpha = np.clip(alpha, 0.0, 1.0)

    # 等效“已熔化体积/面积” = ∑alpha * cell_volume
    V_melt_equiv = float(np.sum(alpha) * cell_volume)

    # --- 相变导致的体积(面积)跳变 ---
    # 由质量守恒: ΔV_phase = V_s(ρ_s/ρ_l - 1)，这里 V_s 用等效已熔化体积近似
    kappa = (rho_s / rho_l) - 1.0
    deltaV_phase = V_melt_equiv * kappa

    # --- 热膨胀导致的体积(面积)变化 ---
    dT_vec = T_core - T_ref

    # 相依 β: 固相用 beta_s, 液相用 beta_l, 糊区线性混合
    beta_eff = beta_s * (1.0 - alpha) + beta_l * alpha
    # 线性小膨胀近似: ΔV/V ≈ β ΔT
    deltaV_thermal = float(np.sum(beta_eff * dT_vec) * cell_volume)

    return float(deltaV_phase), float(deltaV_thermal)


import math

def void_z(f_void: float, r: float, theta_tol: float = 1e-12) -> float:

    if f_void <= 0.0:
        return 2.0 * r  # 没有空腔
    if f_void >= 1.0:
        return 0.0      # 空腔占满半圆

    target = 2.0 * math.pi * f_void   # 右端常数
    def F(theta: float) -> float:
        return theta - math.sin(theta) - target

    lo, hi = 0.0, 2.0 * math.pi
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if (hi - lo) < theta_tol:
            theta = mid
            break
        if F(lo) * F(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    else:
        theta = 0.5 * (lo + hi)

    z0 = r * (1.0 + math.cos(0.5 * theta))
    return z0

def find_region_and_index(phys_coords_list_t, target=(0.0, 20e-6)):
    """
    在每个区域的物理点列表中找到与 target 最近的点。
    phys_coords_list_t[i] 形如 [r, z, region_id]。
    Returns: (region_id, index_in_region)
    """
    best_region, best_idx, best_d2 = None, None, np.inf
    tgt = np.asarray(target, dtype=float)

    for region_id, arr in enumerate(phys_coords_list_t):
        if arr.size == 0:
            continue
        coords = arr[:, :2]          # 只取 r,z
        d2 = np.sum((coords - tgt)**2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] < best_d2:
            best_region, best_idx, best_d2 = region_id, idx, float(d2[idx])

    if best_region is None:
        raise ValueError("phys_coords_list_t 为空或无点。")

    return best_region, best_idx


def map_Tth_to_mech(coords_m, coords_th, T_th, tol):
    pts_m  = np.ascontiguousarray(coords_m[:, :2])
    pts_th = np.ascontiguousarray(coords_th[:, :2])

    tree = cKDTree(pts_th)
    dist, idx = tree.query(pts_m, k=1)

    # 强制容差，否则会“乱配”
    bad = dist > tol
    if np.any(bad):
        raise ValueError(
            f"KDTree mapping failed: {bad.sum()} / {len(dist)} points exceed tol. "
            f"max_dist={dist.max():.3e}, tol={tol:.3e}"
        )

    # 关键：按 mech 顺序直接重排
    return T_th[idx]
