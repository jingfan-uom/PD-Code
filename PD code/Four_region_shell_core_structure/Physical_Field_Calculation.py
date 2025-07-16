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


def compute_Tensor_product(x, y, horizon_mask):

    # Current relative positions after deformation: x '- x
    dx_eff = x[None, :] - x[:, None]
    dz_eff = y[None, :] - y[:, None]
    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)
    # Only compute values where horizon_mask is True; set others to zero
    n_x = np.zeros_like(dx_eff)
    n_z = np.zeros_like(dz_eff)

    n_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    n_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    N = n_x.shape[0]
    Cxx = np.zeros((N, N))
    Cxz = np.zeros((N, N))
    Czx = np.zeros((N, N))
    Czz = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # 只对有效（horizon_mask为True）的键计算
            if horizon_mask[i, j]:
                n_vec = np.array([n_x[i, j], n_z[i, j]])  # 组成二维单位向量
                n_outer = np.outer(n_vec, n_vec)  # 2x2张量积

                Cxx[i, j] = n_outer[0, 0]
                Cxz[i, j] = n_outer[0, 1]
                Czx[i, j] = n_outer[1, 0]
                Czz[i, j] = n_outer[1, 1]

    return Cxx, Cxz, Czx, Czz

from numba import jit
import numpy as np

@jit(nopython=True)
def compute_s_matrix(x_flat, y_flat, Ux, Uz, horizon_mask, distance_matrix):
    """
    JIT-compatible version: compute relative elongation matrix (s_matrix)
    for all bonds within the horizon.

    Parameters:
        x_flat, y_flat: 1D arrays of positions (length N)
        Ux, Uz: 1D arrays of displacements (length N)
        horizon_mask: 2D bool array (N x N), True if bond (i,j) is valid
        distance_matrix: 2D array of original distances (N x N)

    Returns:
        s_matrix: 2D array of relative elongations (N x N)
    """
    N = x_flat.shape[0]
    s_matrix = np.zeros((N, N))

    for i in range(N):
        xi = x_flat[i]
        yi = y_flat[i]
        ui_x = Ux[i]
        ui_z = Uz[i]
        for j in range(N):
            if horizon_mask[i, j]:
                dx0 = x_flat[j] - xi
                dy0 = y_flat[j] - yi
                dux = Ux[j] - ui_x
                duz = Uz[j] - ui_z
                dx_total = dx0 + dux
                dz_total = dy0 + duz
                L1 = np.sqrt(dx_total ** 2 + dz_total ** 2)

                # Avoid division by zero
                if L1 > 1e-14:
                    s_matrix[i, j] = (L1 - distance_matrix[i, j]) / L1
                else:
                    s_matrix[i, j] = 0.0

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
    Tcurr_avg = 0.5 * (T_i + T_j)- Tpre_avg  # shape (N, N)

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



def compute_single_stiffness_tensor(n_r, n_z, c_modu, k_modu, distance):
    """
    Compute the 2D stiffness tensor of a single bond (in r and z directions).
    This function is used in EBBPD (Extended Bond-Based Peridynamic) models.

    Parameters:
        n_r:      ndarray (2,), unit vector in the r-direction
        n_z:      ndarray (2,), unit vector in the z-direction
        c_modu:   float, normal stiffness modulus
        k_modu:   float, shear stiffness modulus
        distance: float, distance between particles

    Returns:
        cr_stiffness, cz_stiffness: ndarray (2, 2), 2×2 bond stiffness tensors
    """
    I = np.eye(*k_modu.shape)

    nr_outer = np.outer(n_r, n_r)
    nz_outer = np.outer(n_z, n_z)

    cr_stiffness = (c_modu * nr_outer + k_modu * (I - nr_outer)) / distance
    cz_stiffness = (c_modu * nz_outer + k_modu * (I - nz_outer)) / distance

    return cr_stiffness, cz_stiffness



@jit(nopython=True)
def update_mu_by_failure(mu, Relative_elongation, s0):
    """
    JIT-compatible version: Update the damage state (μ) based on bond failure criteria.

    Parameters:
        mu: ndarray of int (1/0), bond damage matrix (N x N)
        Relative_elongation: ndarray of float, relative elongation (N x N)
        s0: float, failure threshold

    Returns:
        mu_new: updated μ matrix
    """
    N = mu.shape[0]
    mu_new = np.empty_like(mu)

    for i in range(N):
        for j in range(N):
            if mu[i, j] == 1 and Relative_elongation[i, j] >= s0:
                mu_new[i, j] = 0
            else:
                mu_new[i, j] = mu[i, j]

    return mu_new



def make_pairwise_material_matrices(mask_core, E_core, E_shell, alpha_core, alpha_shell, s0_core, s0_shell):
    """
    Assign pairwise material properties based on core-shell structure,
    and return fully symmetric pairwise average matrices for each property.

    Parameters:
        mask_core:   ndarray of bool, True for core particles, False for shell particles
        E_core:      float, Young’s modulus for core
        E_shell:     float, Young’s modulus for shell
        alpha_core:  float, thermal expansion coefficient for core
        alpha_shell: float, thermal expansion coefficient for shell
        s0_core:     float, critical stretch for core
        s0_shell:    float, critical stretch for shell

    Returns:
        E_avg:       ndarray, symmetric matrix of pairwise averaged Young’s modulus
        alpha_avg:   ndarray, symmetric matrix of pairwise averaged thermal expansion coefficient
        s0_avg:      ndarray, symmetric matrix of pairwise averaged critical stretch
    """
    # Assign field values
    E_field = np.where(mask_core, E_core, E_shell)
    alpha_field = np.where(mask_core, alpha_core, alpha_shell)
    s0_field = np.where(mask_core, s0_core, s0_shell)

    # Flatten to 1D arrays
    E_flat = E_field.ravel()
    alpha_flat = alpha_field.ravel()
    s0_flat = s0_field.ravel()

    # Construct symmetric pairwise average matrices
    E_avg = (E_flat[None, :] + E_flat[:, None]) / 2
    alpha_avg = (alpha_flat[None, :] + alpha_flat[:, None]) / 2
    s0_avg = (s0_flat[None, :] + s0_flat[:, None]) / 2
    return E_avg, alpha_avg, s0_avg



def mark_prebroken_bonds_from_mesh(mu, r_flat_m, z_flat_m, horizon_mask_m, crack_start, crack_end):
    """
    Mark pre-broken bonds (mu[i, j] = 0) that intersect with a given crack segment.
    Positions are automatically constructed from Rmat and Zmat.

    Parameters:
    - mu : ndarray (N, N) —— Initial damage matrix (1 = intact, 0 = pre-broken)
    - r_flat_m, z_flat_m : ndarray —— Flattened meshgrid coordinates
    - horizon_mask_m : ndarray (N, N) —— Bond connection mask
    - crack_start, crack_end : list or array —— Coordinates of the crack segment endpoints (e.g., [0.02, 0.025])

    Returns:
    - Updated damage matrix mu
    """
    positions = np.column_stack((r_flat_m, z_flat_m))  # shape = (N, 2)

    # Vector representing the crack segment
    C = np.array(crack_start)
    D = np.array(crack_end)
    CD = D - C
    N = positions.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j or not horizon_mask_m[i, j]:
                continue  # Skip self-bonds or non-connected bonds

            A = positions[i]
            B = positions[j]
            AB = B - A

            AC = C - A
            AD = D - A
            CA = A - C
            CB = B - C

            cross1 = np.cross(AC, AB) * np.cross(AD, AB)
            cross2 = np.cross(CA, CD) * np.cross(CB, CD)

            if cross1 < 0 and cross2 < 0:
                mu[i, j] = 0
                mu[j, i] = 0  # Ensure symmetry
    return mu
