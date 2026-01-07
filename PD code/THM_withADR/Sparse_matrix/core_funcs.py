import numpy as np

# Compute the shape factor matrix for all pairs (i, j) within the horizon
# This weight is used in nonlocal thermal conductivity calculations
def compute_shape_factor_matrix(Rmat, true_indices):
    r_flat = Rmat.flatten()
    shape_factor_matrix = np.zeros((len(r_flat), len(r_flat)))
    for i, j in zip(*true_indices):
        shape_factor_matrix[i, j] = 1 #(2 * r_flat[j]) / (r_flat[i] + r_flat[j])
    return shape_factor_matrix


# Return thermal conductivity field (assumed constant here)

def get_thermal_conductivity(Tarr, mask_core_th, k_s, k_l, T_solidus, T_liquidus, kshell, delta):

    base_factor = 4 / (np.pi * delta ** 2)
    lam = np.zeros_like(Tarr)
    # 固态区
    mask_solid = mask_core_th & (Tarr < T_solidus)
    lam[mask_solid] = k_s

    # 液态区
    mask_liquid = mask_core_th & (Tarr > T_liquidus)
    lam[mask_liquid] = k_l

    # 相变区
    mask_transition = mask_core_th & (Tarr >= T_solidus) & (Tarr <= T_liquidus)
    alpha = (Tarr[mask_transition] - T_solidus) / (T_liquidus - T_solidus)
    lam[mask_transition] = (1 - alpha) * k_s + alpha * k_l

    # 壳层区
    lam[~mask_core_th] = kshell

    return lam * base_factor


# Build a matrix of pairwise thermal conductivities for each i-j pair within the horizon
def compute_thermal_conductivity_matrix(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, r_flat, true_indices,mask_core_th,kshell):

    lambda_flat = get_thermal_conductivity(Tarr, mask_core_th, k_s, k_l, T_solidus, T_liquidus,kshell, delta)
    lambda_flat = lambda_flat.ravel()
    N = len(r_flat)
    thermal_conductivity_matrix = np.zeros((N, N), dtype=float)

    for i, j in zip(*true_indices):
        thermal_conductivity_matrix[i, j] = (
            2 * lambda_flat[i] / (lambda_flat[i] + lambda_flat[j]) * lambda_flat[j])
    return thermal_conductivity_matrix

# Convert temperature to enthalpy
def get_enthalpy(Tarr, rho, cs, cl, L, T_solidus, T_liquidus):
    H = np.zeros_like(Tarr)

    # =============== 三段处理 ===============
    mask_solid = Tarr < T_solidus
    mask_phase = (Tarr >= T_solidus) & (Tarr <= T_liquidus)
    mask_liquid = Tarr > T_liquidus

    # (1) 固态
    H[mask_solid] = rho[mask_solid] * cs * Tarr[mask_solid]

    # (2) 相变区间

    dT = (T_liquidus - T_solidus)
    alpha = (Tarr[mask_phase] - T_solidus) / dT
    # 显热 + 部分潜热: Cs * T_solidus + alpha * L
    H[mask_phase] = rho[mask_phase] * (
            cs * T_solidus + alpha * L
    )

    # (3) 液态
    H[mask_liquid] = rho[mask_liquid] * (
            cs * T_solidus + L
            + cl * (Tarr[mask_liquid] - T_liquidus)
    )

    return H

# Convert enthalpy back to temperature
def get_temperature(Harr, mask_core_th, rho_s, rho_l, Cs, Cl, L, T_solidus, T_liquidus,
                    rho_shell, Cshell):

    T = np.zeros_like(Harr)
    # For core region (mask_core_th == True)
    mask_core = mask_core_th
    mask_shell = ~mask_core
    # Three regions in the core
    # All operations below are only for core region!

    H_solid_max = rho_s * Cs * T_solidus
    H_liquid_min = H_solid_max + (rho_s + rho_l)/2 * L

    # 1. Solid (core region)
    mask_solid = mask_core & (Harr <= H_solid_max)
    T[mask_solid] = Harr[mask_solid] / (rho_s * Cs)

    # 2. Phase change (core region)
    mask_phase = mask_core & (Harr > H_solid_max) & (Harr <= H_liquid_min)
    dT = T_liquidus - T_solidus
    alpha = (Harr[mask_phase] - H_solid_max) / (rho_s * L)
    T[mask_phase] = T_solidus + alpha * dT

    # 3. Liquid (core region)
    mask_liquid = mask_core & (Harr > H_liquid_min)
    denom = rho_l * Cl
    H_adj = Harr[mask_liquid] - (rho_s * Cs * T_solidus + rho_s * L)
    T[mask_liquid] = H_adj / denom + T_liquidus

    # For shell region (simply use linear relationship)
    T[mask_shell] = Harr[mask_shell] / (rho_shell * Cshell)

    return T


def get_density(T_field, mask_core, rho_s, rho_l, rho_shell, Ts, Tl, Rmat):

    # Initialize all points as shell density
    rho_field = np.full(T_field.shape, rho_shell, dtype=float)

    # Points in the core region
    mask = mask_core.ravel()
    # 1. Solid phase (core region, T < Ts)
    mask_solid = mask & (T_field < Ts)
    rho_field[mask_solid] = rho_s
    # 2. Liquid phase (core region, T >= Tl)
    mask_liquid = mask & (T_field >= Tl)
    rho_field[mask_liquid] = rho_l
    # 3. Between solidus and liquidus (core region, Ts <= T < Tl) -- linear interpolation
    mask_interp = mask & (T_field >= Ts) & (T_field < Tl)
    if np.any(mask_interp):
        T = T_field[mask_interp]
        rho_interp = (rho_l - rho_s) / (Tl - Ts) * (T - Ts) + rho_s
        rho_field[mask_interp] = rho_interp


    rho_avg = (rho_field[None, :] + rho_field[:, None]) / 2
    rho_field = rho_field.reshape(Rmat.shape)

    return rho_field, rho_avg


# Construct the global conductivity matrix K for nonlocal heat transfer
def build_K_matrix(Tarr, compute_thermal_conductivity_matrix, factor_mat,
                   partial_area_matrix, shape_factor_matrix,
                   distance_matrix, horizon_mask, true_indices, r_flat, k_s, k_l, T_solidus, T_liquidus, delta, dt,mask_core_th,kshell):
    N = len(r_flat)
    cond_mat = compute_thermal_conductivity_matrix(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, r_flat, true_indices,mask_core_th, kshell)
    K1 = np.zeros((N, N))

    # Off-diagonal terms (i ≠ j)
    K1[horizon_mask] = (
        factor_mat[horizon_mask] *
        partial_area_matrix[horizon_mask] *
        shape_factor_matrix[horizon_mask] *
        cond_mat[horizon_mask] /
        (distance_matrix[horizon_mask] ** 2)*dt
    )

    # Diagonal terms to ensure row sums to zero

    DiagTerm = np.zeros((N, N))
    DiagTerm[horizon_mask] = factor_mat[horizon_mask] * (
        partial_area_matrix[horizon_mask] *
        shape_factor_matrix[horizon_mask] *
        cond_mat[horizon_mask] * (-1.0) /
        (distance_matrix[horizon_mask] ** 2)*dt
    )
    row_sum = DiagTerm.sum(axis=1)
    np.fill_diagonal(K1, row_sum)
    return K1


def compute_dt_cr_th_solid_with_dist(
    rho_solid, c_solid, k_solid,
    partial_area_flat,   # (nnz,)
    indptr, dist,         # CSR: indptr (N+1,), dist (nnz,)
    delta,
    i=None,               # 可选：默认用 N//2
    eps=1e-15
):
    """
    CSR 版本：计算某个粒子 i 的临界热时间步长 Δt_cr^TH
    只遍历 i 的邻居段，不再需要 horizon_mask 和 N×N distance_matrix / area_matrix。

    partial_area_flat[p] 与 dist[p] 对齐：同一条边 (i -> indices[p]) 的面积/距离
    这里其实不需要 indices（因为只用距离和面积做求和）。
    """
    N = len(indptr) - 1
    if i is None:
        i = N // 2

    a, b = indptr[i], indptr[i + 1]
    if a == b:
        return np.inf  # 没有邻居，分母为0

    dij = dist[a:b]
    areas = partial_area_flat[a:b]

    # 有效导热系数（与你原式一致）
    k_eff = k_solid * 4.0 / (np.pi * delta ** 2)

    # 防止除零：排除 dij=0（正常 remove_self=True 就不会发生，但保险起见）
    safe = dij > eps
    if not np.any(safe):
        return np.inf

    denominator = np.sum((k_eff / (dij[safe] ** 2)) * areas[safe])

    numerator = rho_solid * c_solid
    return numerator / denominator if denominator != 0.0 else np.inf


def calc_expansion_pressure(T, mask_core, V0, ps, pl, Ts, Tl, dV, K_shell):

    # 仅统计核心区

    T_core = T[mask_core]
    # 完全液态
    mask_liquid = T_core >= Tl
    deltaV_liquid = np.sum(mask_liquid) * ((ps - pl) / pl * dV)

    # 相变区间
    mask_phase = (T_core >= Ts) & (T_core < Tl)
    T_phase = T_core[mask_phase]
    # 插值因子
    alpha = (T_phase - Ts) / (Tl - Ts)
    deltaV_phase = np.sum(((ps - pl) / pl) * alpha * dV)

    # 总膨胀体积
    deltaV = deltaV_liquid + deltaV_phase

    # 相对体积变化率
    eps_v = deltaV / V0

    # 膨胀压力
    P = K_shell * eps_v

    return P


import numpy as np

def find_shell_interface(mask_core):
    """
    查找core-shell交界面在shell侧的点，包括最外圈一圈。
    输入:
        mask_core: 2D bool array, True为core，False为shell
    输出:
        shell_edge: 2D bool array，True为shell侧交界点
    """
    rows, cols = mask_core.shape
    # 用False填充外围
    padded = np.pad(mask_core, pad_width=1, mode='constant', constant_values=False)
    shell_edge = np.zeros_like(mask_core, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if not mask_core[i, j]:  # shell区
                # 3x3邻域，现在i,j都要+1对应padded的索引
                local = padded[i:i+3, j:j+3]
                if np.any(local):  # 邻域内有core
                    shell_edge[i, j] = True
    return shell_edge

