import numpy as np



def compute_direction_matrix(x, y, ux, uz, horizon_mask): 
    """
    Compute updated direction matrix based on current relative positions: (x' + u') - (x + u)

    Inputs:
    - x, y: coordinate vectors (N,)
    - ux, uz: displacement vectors (N,)
    - horizon_mask: (N, N) interaction mask

    Outputs:
    - dir_x, dir_z: direction unit vectors (N, N)
    """
    # Current relative positions: (x' + u') - (x + u)
    dx_eff = (x[None, :] + ux[None, :]) - (x[:, None] + ux[:, None])
    dz_eff = (y[None, :] + uz[None, :]) - (y[:, None] + uz[:, None])

    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)

    # 只对 horizon_mask == True 的元素计算，其他点直接设为 0
    dir_x = np.zeros_like(dx_eff)
    dir_z = np.zeros_like(dz_eff)

    dir_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    dir_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    return dir_x, dir_z



def compute_s_matrix(x_flat, y_flat, Ux, Uz, horizon_mask, distance_matrix):
    """
    Compute the elongation matrix s_matrix (N, N) using 2D grid input and horizon_mask.

    Parameters:
        X, Y: original mesh coordinates
        Ux, Uz: displacement fields at corresponding points
        horizon_mask: boolean array of shape
        distance_matrix :Initial lengths

    Returns:
        s_matrix: elongation matrix of shape (N, N)
    """
    # Deformed coordinates
    x_def = (Ux + x_flat)
    y_def = (Uz + y_flat)
    # Original coordinates

    # Deformed lengths
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Elongation computation
    s_matrix = np.zeros_like(distance_matrix)
    s_matrix[horizon_mask] = (L1[horizon_mask] - distance_matrix[horizon_mask]) / distance_matrix[horizon_mask]

    return s_matrix


def compute_delta_temperature(T_grid,  Tpre_avg):
    """
    Compute the average temperature matrix T_avg (N, N), and optionally return the delta
    compared to the previous time step average temperature.

    Parameters:
        T_grid: 2D temperature field array (Ny, Nx)

        T_prev_avg: previous time step average temperature matrix, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """

    Tcurr_flat = T_grid.flatten()  # (N,)
    T_i = Tcurr_flat [:, np.newaxis]  # shape (N, 1)
    T_j = Tcurr_flat [np.newaxis, :]  # shape (1, N)
    Tcurr_avg = 0.5 * (T_i + T_j)  # shape (N, N)


    T_delta = Tcurr_avg - Tpre_avg

    return T_delta


def compute_velocity_third_step(Vr_half, Vz_half, Ar_next, Az_next, dt):
    """
    Implements step 3 in Equation (14): update velocity to time step n+1 using next-step acceleration.

    Parameters:
        Vr_half, Vz_half: intermediate velocities at (n+1/2) in r and z directions
        Ar_next, Az_next: next-step accelerations in r and z directions
        dt: time step

    Returns:
        Vr_new, Vz_new: updated velocities at full step (n+1) in r and z directions
    """
    Vr_new = Vr_half + 0.5 * dt * Ar_next
    Vz_new = Vz_half + 0.5 * dt * Az_next
    return Vr_new, Vz_new

