import numpy as np

def adr_initial_velocity(F0, dt, lambda_diag_matrix):
    """
    Compute initial half-step velocity using Eq. (13), adapted for 2D shape.

    Parameters
    ----------
    F0 : ndarray
        Initial total force, shape (N, N)

    dt : float
        Time step size

    lambda_diag_matrix : ndarray
        Full diagonal matrix of artificial density, shape (N*N, N*N)

    Returns
    -------
    U_dot_half : ndarray
        Initial half-step velocity, shape same as F0 (i.e. N x N)
    """
    original_shape = F0.shape           # e.g. (N, N)
    F0_flat = F0.reshape(-1)            # → (N*N,)
    lambda_flat = np.diag(lambda_diag_matrix)  # → (N*N,)

    V_half_flat = (dt / 2) * (F0_flat / lambda_flat)

    # Reshape back to original 2D shape
    V_half = V_half_flat.reshape(original_shape)

    return V_half



def compute_lambda_diag_matrix(partial_area_matrix, distance_matrix, c, horizon_mask):
    """
    Compute diagonal stiffness matrix (as full matrix) using Eq. (15) logic,
    only considering entries where horizon_mask is True.

    Parameters
    ----------
    partial_area_matrix : ndarray (n_nodes, n_nodes)
        Matrix of partial bond interaction areas.

    distance_matrix : ndarray (n_nodes, n_nodes)
        Matrix of pairwise distances.

    c : float or ndarray
        Bond stiffness, either scalar or matrix matching area matrix.

    horizon_mask : ndarray (bool)
        Boolean matrix of same shape, indicating valid bonds.

    Returns
    -------
    K_diag_matrix : ndarray (n_nodes, n_nodes)
        Diagonal matrix where K[i, i] approximates stiffness row sum.
    """

    # Ensure distance is safe to divide
    distance_safe = np.where(distance_matrix > 1e-12, distance_matrix, 1e-12)

    # Compute contrib_matrix only where mask is True
    contrib_matrix = np.zeros_like(distance_matrix)
    contrib_matrix[horizon_mask] = (c * partial_area_matrix[horizon_mask]) / distance_safe[horizon_mask]

    # Sum row-wise to get K_scalar (only for valid rows)
    lambda_scalar = np.sum(np.abs(contrib_matrix), axis=1)

    # Convert to diagonal matrix
    lambda_diag_matrix = np.diag(lambda_scalar) * 1/4

    return lambda_diag_matrix


def compute_local_damping_coefficient(F_curr, F_prev, velocity_half, lambda_diag_matrix, U, dt):
    """
    Compute local stiffness matrix (1Kn_ii) as a flattened vector using Eq. (19).

    Parameters
    ----------
    F_curr : ndarray (n_nodes, n_dim)
        Current total force at time n

    F_prev : ndarray (n_nodes, n_dim)
        Previous total force at time n-1 (or zeros if n == 0)

    velocity_half : ndarray (n_nodes, n_dim)
        Velocity at time n-1/2

    K_diag : ndarray (n_nodes, n_dim)
        Artificial density matrix diagonal (Λ)

    dt : float
        Time step size

    Returns
    -------
    Kn_local_flat : ndarray (n_nodes * n_dim,)
        Flattened local stiffness values
    """

    # Flatten all arrays
    F_curr_flat = F_curr.reshape(-1,1)
    F_prev_flat = F_prev.reshape(-1,1)
    vel_half_flat = velocity_half.reshape(-1,1)
    K_diag_flat = np.diag(lambda_diag_matrix).reshape(-1,1)
    U = U.reshape(-1, 1)  # shape: (N, 1)

    # Initialize output
    Kn_local_flat = np.zeros_like(F_curr_flat)

    # Only compute where velocity ≠ 0
    nonzero_mask = vel_half_flat != 0.0

    delta_force = (F_curr_flat - F_prev_flat) / K_diag_flat
    Kn_local_flat[nonzero_mask] = -delta_force[nonzero_mask] / (dt * vel_half_flat[nonzero_mask])
    Kn_local_matrix = np.diag(Kn_local_flat.flatten())

    # velocity == 0 → Kn_local remains 0

    numerator = (U.T @ Kn_local_matrix @ U)
    denominator = (U.T @ U)

    if denominator <= 0:
        cn = 0.0
    elif numerator / denominator < 0:
        cn = 0.0
    else:
        cn = 2 * np.sqrt(numerator / denominator)

    return cn


def adr_update_velocity_displacement(U_prev, V_dot_half_prev, F_curr, c_n, D_diag, dt):
    """
    ADR update for flattened arrays using Eq. (17) with flattened vectors.

    Parameters
    ----------
    u_prev : ndarray
        Previous displacement (2D array, e.g. (n_nodes, n_dim))

    u_dot_half_prev : ndarray
        Previous half-step velocity (same shape as u_prev)

    F_curr : ndarray
        Current force (same shape as u_prev)

    c_n : float
        Adaptive damping coefficient at time n

    D_diag : ndarray
        Diagonal of artificial density matrix (same shape as u_prev)

    dt : float
        Time step size

    Returns
    -------
    u_dot_half : ndarray
        Updated half-step velocity, same shape as u_prev

    u_next : ndarray
        Updated displacement, same shape as u_prev
    """
    # Flatten everything
    U_flat = U_prev.reshape(-1)
    V_dot_half_flat = V_dot_half_prev.reshape(-1)
    F_flat = F_curr.reshape(-1)
    D_vector = np.diag(D_diag)


    # Compute updated half-step velocity
    numerator = (2 - c_n * dt) * V_dot_half_flat + 2 * dt * (F_flat / D_vector)
    denominator = 2 + c_n * dt
    V_dot_half_flat = numerator / denominator

    # Update displacement
    U_next_flat = U_flat + dt * V_dot_half_flat

    # Reshape back to original shape
    V_dot_half = V_dot_half_flat.reshape(U_prev.shape)
    U_next = U_next_flat.reshape(U_prev.shape)

    return V_dot_half, U_next

