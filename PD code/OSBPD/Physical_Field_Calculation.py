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
    Y_matrix = np.zeros_like(L0)
    s_matrix[horizon_mask] = (L1[horizon_mask] - L0[horizon_mask]) / L0[horizon_mask]
    Y_matrix[horizon_mask] = (L1[horizon_mask] - L0[horizon_mask])

    return s_matrix, Y_matrix


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


import numpy as np

def compute_dilation_2d(
    distance_matrix,        # |ξ_ij|
    partial_area_matrix,    # A_ij
    horizon_mask,           # True 表示在视界内
    nlength,                # |y_ij|
    dot_xy,                 # ξ_ij · y_ij
    sij,                    # (|y|-|ξ|)/|ξ|
    weight,
    alpha,
    dtemp,
    d
):
    """
    计算每个点的膨胀 θ_i（2D 版本）。

    参数 nlength, dot_xy, sij 由主程序预先计算，这里只做加权求和。
    """
    idist = distance_matrix
    mask = horizon_mask
    # term_ij = d * w_ij * (s_ij - alpha*dtemp) * (ξ·y / |y|) * A_ij
    term = np.zeros_like(idist)
    term[mask] = (
        d * weight[mask]
        * (sij[mask] - alpha * dtemp)
        * (dot_xy[mask] / nlength[mask])
        * partial_area_matrix[mask]
    )

    # 对 j 求和得到每个 i 的 θ_i
    dilation = np.sum(term, axis=1) + 3.0 * alpha * dtemp

    return dilation

