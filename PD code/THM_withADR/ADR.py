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
        Initial half-step velocity
    """

    V_half = (dt / 2) * (F0 / lambda_diag_matrix)


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

    # Compute contrib_matrix only where mask is True
    contrib_matrix = np.zeros_like(distance_matrix)
    contrib_matrix[horizon_mask] = (c * partial_area_matrix[horizon_mask]) / distance_matrix[horizon_mask]

    # Sum row-wise to get K_scalar (only for valid rows)
    lambda_scalar = np.sum(np.abs(contrib_matrix), axis=1) * 1/4

    return lambda_scalar


def compute_local_damping_coefficient(F_curr, F_prev, velocity_half, lambda_diag_matrix, U, dt):
    """
    Compute local stiffness matrix (1Kn_ii) as a flattened vector using Eq. (19).

    Parameters
    ----------
    F_curr :  Current total force at time n

    F_prev : Previous total force at time n-1 (or zeros if n == 0)

    velocity_half : Velocity at time n-1/2

    K_diag : Artificial density matrix diagonal (Λ)

    dt : float
        Time step size

    Returns
    -------
    Kn_local :   local stiffness values
    """

    # Initialize output
    Kn_local = np.zeros_like(F_curr)

    # Only compute where velocity ≠ 0
    nonzero_mask = velocity_half != 0.0

    delta_force = (F_curr - F_prev) / lambda_diag_matrix
    Kn_local[nonzero_mask] = -delta_force[nonzero_mask] / (dt * velocity_half[nonzero_mask])


    # velocity == 0 → Kn_local remains 0

    numerator = np.dot(U, Kn_local * U)
    denominator = np.dot(U, U)

    if denominator <= 0 or numerator / denominator < 0:
        cn = 0.0
    else:
        cn = 2 * np.sqrt(numerator / denominator)

    return cn


def adr_update_velocity_displacement(U_prev, V_dot_half_prev, F_curr, c_n, D_diag, dt):
    """
    ADR update for flattened arrays using Eq. (17) with flattened vectors.

    Parameters
    ----------
    u_prev : Previous displacement
    u_dot_half_prev : Previous half-step velocity (same shape as u_prev)

    F_curr :Current force (same shape as u_prev)

    c_n : Adaptive damping coefficient at time n

    D_diag :  Diagonal of artificial density matrix

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


    # Compute updated half-step velocity
    numerator = (2 - c_n * dt) * V_dot_half_prev + 2 * dt * (F_curr / D_diag)
    denominator = 2 + c_n * dt
    V_dot_half = numerator / denominator

    # Update displacement
    U_next = U_prev + dt * V_dot_half

    return V_dot_half, U_next

