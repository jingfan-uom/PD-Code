import numpy as np

# Compute the shape factor matrix for all pairs (i, j) within the horizon
# This weight is used in nonlocal thermal conductivity calculations
def compute_shape_factor_matrix(Rmat, true_indices):
    r_flat = Rmat.flatten()
    shape_factor_matrix = np.zeros((len(r_flat), len(r_flat)))
    for i, j in zip(*true_indices):
        shape_factor_matrix[i, j] = (2 * r_flat[j]) / (r_flat[i] + r_flat[j])
    return shape_factor_matrix


# Return thermal conductivity field (assumed constant here)
def get_thermal_conductivity(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, dz):
    """
    Tarr: 温度场（可为 1D 或 2D）
    k_f: 固相导热率（低温侧）
    k_m: 液相导热率（高温侧）
    T_solidus, T_liquidus: 相变起止温度；若相等则为单点相变
    delta: 视界大小，用于非局部归一化系数
    返回: 热导率场 lam，与 Tarr 形状相同
    """

    if dz == 0:
        base_factor = 1.0 / delta
    else:
        pi = np.pi
        base_factor = 4.0 / (pi * delta ** 2)

    lam = np.zeros_like(Tarr)

    # 情况①：单点相变（刀切）
    if np.allclose(T_solidus, T_liquidus):
        T_f = T_solidus

        lam[Tarr < T_f] = k_s
        lam[Tarr >= T_f] = k_l

    # 情况②：有相变区间（平滑过渡）
    else:
        # 固态区
        lam[Tarr < T_solidus] = k_s
        # 液态区
        lam[Tarr > T_liquidus] = k_l
        # 相变区间 - 线性插值
        in_transition = (Tarr >= T_solidus) & (Tarr <= T_liquidus)
        alpha = (Tarr[in_transition] - T_solidus) / (T_liquidus - T_solidus)
        lam[in_transition] = (1 - alpha) * k_s + alpha * k_l

    return lam * base_factor




# Build a matrix of pairwise thermal conductivities for each i-j pair within the horizon
def compute_thermal_conductivity_matrix(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, r_flat, true_indices,dz):

    lambda_flat = get_thermal_conductivity(Tarr.flatten(), k_s, k_l, T_solidus, T_liquidus, delta,dz)
    N = len(r_flat)
    thermal_conductivity_matrix = np.zeros((N, N), dtype=float)

    for i, j in zip(*true_indices):
        thermal_conductivity_matrix[i, j] = (
            2 * lambda_flat[i] / (lambda_flat[i] + lambda_flat[j]) * lambda_flat[j])
    return thermal_conductivity_matrix

# Convert temperature to enthalpy
def get_enthalpy(Tarr, rho, cs, cl, L, T_solidus, T_liquidus):
    """
    Tarr: 温度数组 (N,) 或 (Nr, Nz)
    rho, Cs, Cl, L: 同形状的材料属性数组
    T_solidus, T_liquidus: 标量或可广播数组
    当 T_solidus == T_liquidus 时，即刀切式相变
    返回: 总焓 H
    """
    H = np.zeros_like(Tarr)


    # 如果有区间: T_solidus < T_liquidus
    if not np.allclose(T_solidus, T_liquidus):
        # =============== 三段处理 ===============
        mask_solid = Tarr < T_solidus
        mask_phase = (Tarr >= T_solidus) & (Tarr <= T_liquidus)
        mask_liquid = Tarr > T_liquidus

        # (1) 固态
        H[mask_solid] = rho * cs * Tarr[mask_solid]

        # (2) 相变区间
        # 注意：当 (T_liquidus - T_solidus) 很小也可能导致数值不稳定
        # 建议实际中加上 epsilon 处理
        dT = (T_liquidus - T_solidus)
        alpha = (Tarr[mask_phase] - T_solidus) / dT
        # 显热 + 部分潜热: Cs * T_solidus + alpha * L
        H[mask_phase] = rho * (
            cs * T_solidus + alpha * L
        )

        # (3) 液态
        H[mask_liquid] = rho * (
            cs * T_solidus + L
            + cl * (Tarr[mask_liquid] - T_liquidus)
        )

    else:
        # =============== 单点相变: 刀切式 ===============
        T_m = T_solidus  # == T_liquidus

        mask_solid = Tarr < T_m
        mask_phasechange =  Tarr == T_m
        mask_liquid = Tarr > T_m

        # (1) 固态: H = rho * Cs * T
        H[mask_solid] = rho * cs * Tarr[mask_solid]
        # (2) 一旦 >= T_m, 说明材料全部经历相变 => Cs*T_m + L + Cl*(T - T_m)
        H[mask_phasechange] = rho * (cs * T_m + L)
        H[mask_liquid] = rho * (cs * T_m + L + cl * (Tarr[mask_liquid] - T_m))

    return H



# Convert enthalpy back to temperature
def get_temperature(Harr, rho, Cs, Cl, L, T_solidus, T_liquidus):
    """
    Harr: 焓数组 (N,) 或 (Nr, Nz)
    当 T_solidus == T_liquidus => 单点相变
    """
    T = np.zeros_like(Harr)

    # 判断是否是单点相变
    if not np.allclose(T_solidus, T_liquidus):
        # =============== 三段处理 ===============
        # 1) 固态区焓阈值
        H_solid_max = rho * Cs * T_solidus
        # 2) 相变区上限: H_solid_max + rho*L
        H_liquid_min = H_solid_max + rho * L

        mask_solid = Harr <= H_solid_max
        mask_phase = (Harr > H_solid_max) & (Harr <= H_liquid_min)
        mask_liquid = Harr > H_liquid_min

        # 固态: T = H / (ρ Cs)
        T[mask_solid] = Harr[mask_solid] / (rho * Cs)

        # 相变: α = [H - H_solid_max]/(ρ L)
        # T = T_solidus + α*(T_liquidus - T_solidus)
        dT = (T_liquidus - T_solidus)
        alpha = (Harr[mask_phase] - H_solid_max) / (rho * L)
        T[mask_phase] = T_solidus + alpha * dT

        # 液态: T = ...
        # (H - [Cs*T_solidus + L]) / (rho*Cl) + T_liquidus
        # = { Harr - [H_solid_max + rho*L] } / (rho Cl) + T_liquidus
        # 其中 H_solid_max = rho*Cs*T_solidus
        #     [H_solid_max + rho*L] = rho*(Cs*T_solidus + L)
        denom = rho * Cl
        H_adj = Harr[mask_liquid] - (H_solid_max + rho*L)
        T[mask_liquid] = H_adj/denom + T_liquidus

    else:
        # =============== 单点相变: 刀切式 ===============
        T_m = T_solidus

        # 在单点相变下，存在两段:
        # (1) Solid: H <= rho*Cs*T_m
        H_solid_max = rho * Cs * T_m

        mask_solid = Harr < H_solid_max
        mask_phase = (Harr <= (H_solid_max + L * rho)) & (Harr >= H_solid_max)
        mask_liquid = Harr > H_solid_max + L * rho

        # (1) 固态
        T[mask_solid] = Harr[mask_solid] / (rho * Cs)

        # (2) 液态: Once H>H_solid_max => T> T_m
        dT = (T_liquidus - T_solidus)
        T[mask_phase] = T_m
        T[mask_liquid] = (Harr[mask_liquid] - rho * ( Cs * T_m + L))/(rho*Cl) + T_m

    return T



# Construct the global conductivity matrix K for nonlocal heat transfer
def build_K_matrix(Tarr, compute_thermal_conductivity_matrix, factor_mat,
                   partial_area_matrix, shape_factor_matrix,
                   distance_matrix, horizon_mask, true_indices, k_s, k_l, T_s,T_l, delta, r_flat,dt,dz):
    N = len(r_flat)
    cond_mat = compute_thermal_conductivity_matrix(Tarr, k_s, k_l, T_s,T_l, delta, r_flat, true_indices,dz)
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

