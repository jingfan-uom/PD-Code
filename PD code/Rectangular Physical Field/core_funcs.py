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
def get_thermal_conductivity(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, dz):
    """
    Compute the thermal conductivity field based on the temperature field.

    Parameters:
    - Tarr: temperature array
    - k_s: thermal conductivity in the solid phase
    - k_l: thermal conductivity in the liquid phase
    - T_solidus: solidus temperature (below which material is fully solid)
    - T_liquidus: liquidus temperature (above which material is fully liquid)
    - delta: horizon radius (for nonlocal models)
    - dz: cell size in the z-direction (used to distinguish 1D vs 2D)

    Returns:
    - Thermal conductivity array scaled by the base factor
    """
    if dz == 0:
        base_factor = 1.0 / delta
    else:
        pi = np.pi
        base_factor = 4.0 / (pi * delta ** 2)

    lam = np.zeros_like(Tarr)

    # Solid phase
    lam[Tarr < T_solidus] = k_s

    # Liquid phase
    lam[Tarr > T_liquidus] = k_l

    # Phase change region – linear interpolation
    in_transition = (Tarr >= T_solidus) & (Tarr <= T_liquidus)
    alpha = (Tarr[in_transition] - T_solidus) / (T_liquidus - T_solidus)
    lam[in_transition] = (1 - alpha) * k_s + alpha * k_l

    return lam * base_factor



# Build a matrix of pairwise thermal conductivities for each i-j pair within the horizon
def compute_thermal_conductivity_matrix(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, dz, r_flat, true_indices):
    if Tarr.ndim == 1:
        lambda_flat = get_thermal_conductivity(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, dz)
    else:
        lambda_flat = get_thermal_conductivity(Tarr.flatten(), k_s, k_l, T_solidus, T_liquidus, delta, dz)
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
    H[mask_solid] = rho * cs * Tarr[mask_solid]

    # (2) 相变区间

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

    return H


# Convert enthalpy back to temperature
def get_temperature(Harr, rho, Cs, Cl, L, T_solidus, T_liquidus):
    T = np.zeros_like(Harr)

    # 1) 固态区焓阈值
    H_solid_max = rho * Cs * T_solidus
    # 2) 相变区上限: H_solid_max + rho*L
    H_liquid_min = H_solid_max + rho * L

    mask_solid = Harr <= H_solid_max
    mask_phase = (Harr > H_solid_max) & (Harr <= H_liquid_min)
    mask_liquid = Harr > H_liquid_min

    T[mask_solid] = Harr[mask_solid] / (rho * Cs)

    dT = (T_liquidus - T_solidus)
    alpha = (Harr[mask_phase] - H_solid_max) / (rho * L)
    T[mask_phase] = T_solidus + alpha * dT

    denom = rho * Cl
    H_adj = Harr[mask_liquid] - (H_solid_max + rho * L)
    T[mask_liquid] = H_adj / denom + T_liquidus

    return T

# Construct the global conductivity matrix K for nonlocal heat transfer
def build_K_matrix(Tarr, compute_thermal_conductivity_matrix, factor_mat,
                   partial_area_matrix, shape_factor_matrix,
                   distance_matrix, horizon_mask, true_indices, r_flat, k_s, k_l, T_solidus, T_liquidus, delta, dz,dt):
    N = len(r_flat)
    cond_mat = compute_thermal_conductivity_matrix(Tarr, k_s, k_l, T_solidus, T_liquidus, delta, dz, r_flat, true_indices)
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

