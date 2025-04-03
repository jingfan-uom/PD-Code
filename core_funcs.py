import numpy as np

# Compute the shape factor matrix for all pairs (i, j) within the horizon
# This weight is used in nonlocal thermal conductivity calculations
def compute_shape_factor_matrix(Rmat, true_indices):
    r_flat = Rmat.flatten()
    shape_factor_matrix = np.zeros((len(r_flat), len(r_flat)))
    for i, j in zip(*true_indices):
        shape_factor_matrix[i, j] = (2 * r_flat[j]) / (r_flat[i] + r_flat[j])
    return shape_factor_matrix

# Estimate the overlapping area between a rectangular cell and a circular horizon using sub-grid sampling
def partial_area_of_cell_in_circle(x_cell_center, z_cell_center, dx, dz,
                                   x_circle_center, z_circle_center, delta):
    sub = 10  # Subdivisions per dimension (total sub*sub sampling points)
    step_x = dx / sub
    step_z = dz / sub
    x0 = x_cell_center - 0.5 * dx  # Bottom-left x corner of cell
    z0 = z_cell_center - 0.5 * dz  # Bottom-left z corner of cell
    count_in = 0  # Count how many sample points are inside the circle
    for ix in range(sub):
        for iz in range(sub):
            x_samp = x0 + (ix + 0.5) * step_x
            z_samp = z0 + (iz + 0.5) * step_z
            dist2 = (x_samp - x_circle_center) ** 2 + (z_samp - z_circle_center) ** 2
            if dist2 <= delta ** 2 + 1e-6:
                count_in += 1
    return (count_in / (sub * sub)) * dx * dz  # Fractional area within the circle

#   Construct the area overlap matrix between every point and its neighbors within the horizon
def compute_partial_area_matrix(x_flat, z_flat, dx, dz, delta, distance_matrix):
    N = len(x_flat)
    area_mat = np.zeros((N, N), dtype=float)
    for i in range(N):
        cx, cz = x_flat[i], z_flat[i]  # Center of the circle (i)
        for j in range(N):
            dist = distance_matrix[i, j]
            if dist > delta + 1e-6:
                area_mat[i, j] = 0.0  # Definitely outside
            elif dist <= (delta - 0.5 * dx + 1e-6):
                area_mat[i, j] = dx * dz  # Entire cell inside circle
            else:
                xj, zj = x_flat[j], z_flat[j]
                area_mat[i, j] = partial_area_of_cell_in_circle(xj, zj, dx, dz, cx, cz, delta)
    return area_mat

# Return thermal conductivity field (assumed constant here)
def get_thermal_conductivity(Tarr, k_mat, delta):
    pi = np.pi
    return np.full_like(Tarr, k_mat * 4 / pi / delta / delta)

# Build a matrix of pairwise thermal conductivities for each i-j pair within the horizon
def compute_thermal_conductivity_matrix(Tarr, k_mat, delta, r_flat, horizon_mask, true_indices):
    lambda_flat = get_thermal_conductivity(Tarr.flatten(), k_mat, delta)
    N = len(r_flat)
    thermal_conductivity_matrix = np.zeros((N, N))
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
                   distance_matrix, horizon_mask, true_indices, r_flat, k_mat, delta):
    N = len(r_flat)
    cond_mat = compute_thermal_conductivity_matrix(Tarr, k_mat, delta, r_flat, horizon_mask, true_indices)
    K1 = np.zeros((N, N))

    # Off-diagonal terms (i ≠ j)
    valid_offdiag = horizon_mask & (~np.eye(N, dtype=bool))
    K1[valid_offdiag] = (
        factor_mat[valid_offdiag] *
        partial_area_matrix[valid_offdiag] *
        shape_factor_matrix[valid_offdiag] *
        cond_mat.T[valid_offdiag] /
        (distance_matrix[valid_offdiag] ** 2)
    )

    # Diagonal terms to ensure row sums to zero
    valid_diag = horizon_mask & (~np.eye(N, dtype=bool))
    DiagTerm = np.zeros((N, N))
    DiagTerm[valid_diag] = factor_mat[valid_diag] * (
        partial_area_matrix[valid_diag] *
        shape_factor_matrix[valid_diag] *
        cond_mat[valid_diag] * (-1.0) /
        (distance_matrix[valid_diag] ** 2)
    )
    row_sum = DiagTerm.sum(axis=1)
    np.fill_diagonal(K1, row_sum)
    return K1

# Apply mixed boundary conditions (mirror and Dirichlet) to temperature array


def apply_mixed_bc(Tarr, z_all, Nr_tot, Nz_tot, ghost_nodes):
    # Bottom ghost layers reflect interior


    ghost_inds_bottom = np.arange(ghost_nodes)
    interior_inds_bottom = 2 * ghost_nodes - ghost_inds_bottom - 1
    Tarr[ghost_inds_bottom[:, None], :] = Tarr[interior_inds_bottom[:, None], :]

    # Top ghost layers reflect interior
    ghost_inds_top = np.arange(Nz_tot - 1, Nz_tot - ghost_nodes - 1, -1)
    i_local = np.arange(ghost_nodes)
    interior_inds_top = (Nz_tot - 1) - (2 * ghost_nodes - i_local - 1)
    Tarr[ghost_inds_top[:, None], :] = Tarr[interior_inds_top[:, None], :]

    # Left ghost columns reflect interior
    ghost_inds_left = np.arange(ghost_nodes)
    interior_inds_left = 2 * ghost_nodes - ghost_inds_left - 1
    Tarr[:, ghost_inds_left] = Tarr[:, interior_inds_left]

    # Right ghost columns reflect interior
    ghost_inds_right = np.arange(Nr_tot - 1, Nr_tot - ghost_nodes - 1, -1)
    j_local = np.arange(ghost_nodes)
    interior_inds_right = (Nr_tot - 1) - (2 * ghost_nodes - j_local - 1)
    Tarr[:, ghost_inds_right] = Tarr[:, interior_inds_right]


    # Partial Dirichlet condition on right edge (region 0.3 <= z <= 0.5)
    T_hot = 500.0
    z_mask = (z_all >= 0.3 - 1e-6) & (z_all <= 0.5 + 1e-6)
    z_sel = np.where(z_mask)[0]  # Affected z rows
    ghost_cols = np.arange(Nr_tot - 1, Nr_tot - ghost_nodes - 1, -1)  # Ghost columns on right
    i_local = np.arange(ghost_nodes)
    offsets = 2 * ghost_nodes - 1 - 2 * i_local
    interior_cols = ghost_cols - offsets

    # Apply mirrored temperature to enforce Dirichlet BC: T_ghost = 2*T_hot - T_interior
    Tarr[z_sel[:, None], ghost_cols[None, :]] = 2.0 * T_hot - Tarr[z_sel[:, None], interior_cols[None, :]]

    return Tarr
