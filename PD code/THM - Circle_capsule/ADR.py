import numpy as np



def compute_lambda_diag_matrix(partial_area_matrix, distance_matrix, c, horizon_mask):

    # Compute contrib_matrix only where mask is True
    contrib_matrix = np.zeros_like(distance_matrix)
    contrib_matrix[horizon_mask] = (c[horizon_mask] * partial_area_matrix[horizon_mask]) / distance_matrix[horizon_mask]

    # Sum row-wise to get K_scalar (only for valid rows)
    lambda_diag_matrix = np.sum(np.abs(contrib_matrix), axis=1) * 1/4

    return lambda_diag_matrix


def compute_local_damping_coefficient(F_curr, F_prev, velocity_half, lambda_diag_matrix, U, dt):

    Kn_local = np.zeros_like(F_curr)

    # Only compute where velocity ≠ 0
    nonzero_mask = velocity_half != 0.0
    delta_force = -(F_curr - F_prev) / lambda_diag_matrix
    Kn_local[nonzero_mask] = delta_force[nonzero_mask] / (dt * velocity_half[nonzero_mask])
    # velocity == 0 → Kn_local remains 0

    numerator = np.dot(Kn_local * U, U)
    denominator = np.dot(U, U)

    if denominator == 0 or numerator / denominator < 0:
        cn = 0
    else:
        cn = 2 * np.sqrt(numerator / denominator)

    return cn


def adr_update_velocity_displacement(U_prev, V_dot_half_prev, F_curr, c_n, D_diag, dt):

    # Compute updated half-step velocity
    numerator = (2 - c_n * dt) * V_dot_half_prev + 2 * dt * (F_curr / D_diag)
    denominator = 2 + c_n * dt
    V_dot_half = numerator / denominator

    # Update displacement
    U_next = U_prev + dt * V_dot_half

    return V_dot_half, U_next
