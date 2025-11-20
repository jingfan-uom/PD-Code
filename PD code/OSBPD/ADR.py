import numpy as np



def compute_local_damping_coefficient(F_curr, F_prev, velocity_half, lambda_diag_matrix, U, dt):

    # Initialize output
    Kn_local_flat = np.zeros_like(F_curr)

    # Only compute where velocity ≠ 0
    nonzero_mask = velocity_half != 0.0

    delta_force = (F_curr - F_prev) / lambda_diag_matrix
    Kn_local_flat[nonzero_mask] = -delta_force[nonzero_mask] / (dt * velocity_half[nonzero_mask])

    numerator = np.dot(U, Kn_local_flat * U)
    denominator = np.dot(U, U)

    if denominator <= 0:
        cn = 0.0
    elif numerator / denominator < 0:
        cn = 0.0
    else:
        cn = 2 * np.sqrt(numerator / denominator)

    return cn


def adr_update_velocity_displacement(U_prev, V_dot_half_prev, F_curr, c_n, mass, dt):

    # Compute updated half-step velocity
    numerator = (2 - c_n * dt) * V_dot_half_prev + 2 * dt * (F_curr/ mass)
    denominator = 2 + c_n * dt
    V_dot_half = numerator / denominator
    # Update displacement
    U_next = U_prev + dt * V_dot_half

    return V_dot_half, U_next

