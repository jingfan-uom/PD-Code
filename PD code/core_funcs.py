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
def get_thermal_conductivity(Tarr, k_mat, delta):
    pi = np.pi

    return np.full_like(Tarr, k_mat*4/pi / delta/delta, dtype=float)


# Build a matrix of pairwise thermal conductivities for each i-j pair within the horizon
def compute_thermal_conductivity_matrix(Tarr, k_mat, delta, r_flat, true_indices):
    if Tarr.ndim == 1:
        lambda_flat = get_thermal_conductivity(Tarr, k_mat, delta)
    else:
        lambda_flat = get_thermal_conductivity(Tarr.flatten(), k_mat, delta)
    N = len(r_flat)
    thermal_conductivity_matrix = np.zeros((N, N), dtype=float)

    for i, j in zip(*true_indices):
        thermal_conductivity_matrix[i, j] = (
            2 * lambda_flat[i] / (lambda_flat[i] + lambda_flat[j]) * lambda_flat[j])
    return thermal_conductivity_matrix

# Convert temperature to enthalpy
def get_enthalpy(Tarr, rho, Cp):
    return rho * Cp * Tarr

# Convert enthalpy back to temperature
def get_temperature(Harr, rho, Cp):
    return Harr / (rho * Cp)

# Construct the global conductivity matrix K for nonlocal heat transfer
def build_K_matrix(Tarr, compute_thermal_conductivity_matrix, factor_mat,
                   partial_area_matrix, shape_factor_matrix,
                   distance_matrix, horizon_mask, true_indices, r_flat, k_mat, delta,dt):
    N = len(r_flat)
    cond_mat = compute_thermal_conductivity_matrix(Tarr, k_mat, delta, r_flat, true_indices)
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

