from scipy.spatial import cKDTree
import numpy as np

def compute_direction_matrix(x, y, ux, uz, horizon_mask): 
    """
    Compute updated direction matrix based on current relative positions: (x' + u') - (x + u)

    Inputs:
    - x, y: coordinate vectors (N,)
    - ux, uz: displacement vectors (N,)
    - horizon_mask: (N, N) boolean mask indicating valid interactions

    Outputs:
    - dir_x, dir_z: direction unit vectors (N, N)
    """
    # Current relative positions after deformation: (x + ux)' - (x + ux)
    dx_eff = (x[None, :] + ux[None, :]) - (x[:, None] + ux[:, None])
    dz_eff = (y[None, :] + uz[None, :]) - (y[:, None] + uz[:, None])

    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)

    # Only compute values where horizon_mask is True; set others to zero
    dir_x = np.zeros_like(dx_eff)
    dir_z = np.zeros_like(dz_eff)

    dir_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    dir_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    return dir_x, dir_z



def compute_s_matrix(x_flat, y_flat, Ux, Uz, horizon_mask, distance_matrix):
    """
    Compute the elongation matrix s_matrix (N, N) using 2D grid input and horizon_mask.

    Parameters:
        x_flat, y_flat: original mesh coordinates (flattened)
        Ux, Uz: displacement fields at corresponding points
        horizon_mask: boolean array of shape (N, N)
        distance_matrix: matrix of original bond lengths

    Returns:
        s_matrix: elongation matrix of shape (N, N)
    """
    # Deformed coordinates
    x_def = (Ux + x_flat)
    y_def = (Uz + y_flat)

    # Compute deformed bond lengths
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Compute relative elongation
    s_matrix = np.zeros_like(distance_matrix)
    s_matrix[horizon_mask] = (L1[horizon_mask] - distance_matrix[horizon_mask]) / distance_matrix[horizon_mask]

    return s_matrix


def compute_delta_temperature(T_grid, Tpre_avg):
    """
    Compute the average temperature matrix and the difference from the previous step.

    Parameters:
        T_grid: current temperature field (1D or flattened)
        Tpre_avg: average temperature matrix from previous time step, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """
    T_i = T_grid[:, np.newaxis]  # shape (N, 1)
    T_j = T_grid[np.newaxis, :]  # shape (1, N)
    Tcurr_avg = 0.5 * (T_i + T_j) - Tpre_avg  # shape (N, N)

    return Tcurr_avg



def shrink_Tth_by_matching_coords(Rmat_m, Zmat_m, Rmat_th, Zmat_th):
    """
    Remove Tth values whose coordinates do not match with Rmat_m / Zmat_m;
    only retain matching coordinate points.

    Parameters:
        Rmat_m     - target region r coordinates (2D)
        Zmat_m     - target region z coordinates (2D)
        Rmat_th    - Tth's corresponding r coordinates (2D)
        Zmat_th    - Tth's corresponding z coordinates (2D)

    Returns:
        indices    - indices of matched coordinates for filtering Tth
    """
    # All original coordinates (reference grid)
    points_th = np.column_stack((Rmat_th.ravel(), Zmat_th.ravel()))
    # Target coordinates (to keep)
    points_m = np.column_stack((Rmat_m.ravel(), Zmat_m.ravel()))

    # Use KDTree to find index mapping from target to reference
    tree = cKDTree(points_th)
    _, indices = tree.query(points_m)
    return indices


def filter_array_by_indices_keep_only(Tarr, indices):
    """
    Keep only elements in Tarr whose indices are in the given list.

    Parameters:
        Tarr: 1D array to filter
        indices: indices to retain

    Returns:
        Filtered 1D array with only selected elements
    """
    keep_mask = np.zeros_like(Tarr, dtype=bool)
    keep_mask[indices] = True

    # Filter using mask
    Tth_shrunk = Tarr[keep_mask]

    return Tth_shrunk
