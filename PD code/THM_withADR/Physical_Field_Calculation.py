import numpy as np



def compute_direction_matrix(X, Y, Ux, Uz, horizon_mask):
    """
    Compute updated direction matrix based on current relative positions: (x' + u') - (x + u)

    Inputs:
    - X, Y: coordinate grids (Ny, Nx)
    - Ux, Uz: displacement fields (Ny, Nx)
    - horizon_mask: (N, N) interaction mask

    Outputs:
    - dir_x, dir_z: direction unit vectors (N, N)
    """
    x = X.flatten()
    y = Y.flatten()
    ux = Ux.flatten()
    uz = Uz.flatten()

    # Current relative positions: (x' + u') - (x + u)
    dx_eff = (x[None, :] + ux[None, :]) - (x[:, None] + ux[:, None])
    dz_eff = (y[None, :] + uz[None, :]) - (y[:, None] + uz[:, None])

    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)
    dist_eff[~horizon_mask] = 1.0  # avoid division by zero outside horizon

    dir_x = np.where(horizon_mask, dx_eff / dist_eff, 0.0)
    dir_z = np.where(horizon_mask, dz_eff / dist_eff, 0.0)

    return dir_x, dir_z


def compute_s_matrix(X, Y, Ux, Uz, horizon_mask):
    """
    Compute the elongation matrix s_matrix (N, N) using 2D grid input and horizon_mask.

    Parameters:
        X, Y: original mesh coordinates (Ny, Nx)
        Ux, Uz: displacement fields at corresponding points (Ny, Nx)
        horizon_mask: boolean array of shape (N, N)

    Returns:
        s_matrix: elongation matrix of shape (N, N)
    """
    # Deformed coordinates
    x_def = (Ux + X).flatten()
    y_def = (Uz + Y).flatten()
    # Original coordinates
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Initial lengths
    dx0 = x_flat[None, :] - x_flat[:, None]
    dz0 = y_flat[None, :] - y_flat[:, None]
    L0 = np.sqrt(dx0 ** 2 + dz0 ** 2)

    # Deformed lengths
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Avoid division by zero
    L0[L0 == 0] = 1.0

    # Elongation computation
    s_matrix = np.zeros_like(L0)
    s_matrix[horizon_mask] = (L1[horizon_mask] - L0[horizon_mask]) / L0[horizon_mask]

    return s_matrix


def compute_delta_temperature(T_grid, horizon_mask, T_prev_avg):
    """
    Compute the average temperature matrix T_avg (N, N), and optionally return the delta
    compared to the previous time step average temperature.

    Parameters:
        T_grid: 2D temperature field array (Ny, Nx)
        horizon_mask: boolean array of shape (N, N)
        T_prev_avg: previous time step average temperature matrix, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """

    T_flat = T_grid.flatten()  # (N,)
    T_i = T_flat[:, np.newaxis]  # shape (N, 1)
    T_j = T_flat[np.newaxis, :]  # shape (1, N)
    T_avg = 0.5 * (T_i + T_j)  # shape (N, N)

    # Zero out values outside horizon
    T_avg[~horizon_mask] = 0.0
    T_prev_avg[~horizon_mask] = 0.0
    T_delta = T_avg - T_prev_avg

    return T_delta


"""explicit differential scheme:


def compute_accelerated_velocity(Ur_curr, Uz_curr):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr

    Relative_elongation = pfc.compute_s_matrix(Rmat, Zmat, Ur_new, Uz_new, horizon_mask)

    Ar_new = dir_r * c * (Relative_elongation) * partial_area_matrix / rho_s
    Az_new = dir_z * c * (Relative_elongation) * partial_area_matrix / rho_s

    Ar_new = np.sum(Ar_new, axis=1).reshape(Ur_curr.shape)  # Shape matches Ur_curr
    Az_new = np.sum(Az_new, axis=1).reshape(Uz_curr.shape) + bz / rho_s

    return Ar_new, Az_new  # Or return other desired quantities

def compute_next_displacement_field(Ur_curr, Uz_curr, Vr_curr, Vz_curr, Ar_new, Az_new):
    Vr_half = Vr_curr + 0.5 * dt * Ar_new
    Vz_half = Vz_curr + 0.5 * dt * Az_new
    Ur_next = Ur_curr + dt * Vr_half
    Uz_next = Uz_curr + dt * Vz_half
    return Ur_next, Uz_next, Vr_half, Vz_half

def compute_next_velocity_third_step(Vr_half, Vz_half, Ur_next, Uz_next, dt):
    Ar_next, Az_next = compute_accelerated_velocity(Ur_next, Uz_next)
    Vr_new = Vr_half + 0.5 * dt * Ar_next
    Vz_new = Vz_half + 0.5 * dt * Az_next
    return Vr_new, Vz_new, Ar_next, Az_next
    """
