import numpy as np

# Apply mixed boundary conditions (zero-flux and Dirichlet-mirror) to the temperature array

def apply_bc_zero_flux(Tarr, ghost_inds, interior_inds, axis):
    """
    Apply zero-flux (Neumann-type) boundary condition: T_ghost = T_interior

    Parameters:
    -----------
    Tarr : np.ndarray
        Temperature array, either 1D (Nr_tot,) or 2D (Nz_tot, Nr_tot)
    ghost_inds : array-like
        Indices of ghost nodes. If axis=0, they refer to rows; if axis=1, they refer to columns.
    interior_inds : array-like
        Indices of interior nodes corresponding to each ghost node.
    axis : int
        0 for top/bottom (row-wise), 1 for left/right (column-wise)
    """
    if Tarr.ndim == 1:
        # 1D case
        Tarr[ghost_inds] = Tarr[interior_inds]
    else:
        if axis == 0:
            # Row-wise application (top/bottom)
            Tarr[ghost_inds, :] = Tarr[interior_inds, :]
        else:
            # Column-wise application (left/right)
            Tarr[:, ghost_inds] = Tarr[:, interior_inds]
    return Tarr


def apply_bc_dirichlet_mirror(Tarr, ghost_inds, interior_inds, T_bc,
                              axis, z_mask=None, r_mask=None):
    """
    Apply Dirichlet mirror boundary condition: T_ghost = 2*T_bc - T_interior

    Parameters:
    -----------
    Tarr : np.ndarray
        Temperature array, 1D (Nr_tot,) or 2D (Nz_tot, Nr_tot)
    ghost_inds : 1D array
        Indices of ghost nodes. If axis=0: row indices (top/bottom); if axis=1: column indices (left/right)
    interior_inds : 1D array
        Indices of corresponding interior nodes
    T_bc : float
        Dirichlet boundary temperature
    axis : int, optional
        0 for top/bottom (rows), 1 for left/right (columns). Default is 0.
    z_mask : 1D bool array, optional
        When axis=1, apply Dirichlet only to rows where z_mask=True
    r_mask : 1D bool array, optional
        When axis=0, apply Dirichlet only to columns where r_mask=True

    Returns:
    --------
    Tarr : np.ndarray
        Temperature array after applying the boundary condition
    """

    # ---------- Case 1: 1D ----------
    if Tarr.ndim == 1:
        Tarr[ghost_inds] = 2.0 * T_bc - Tarr[interior_inds]
        return Tarr

    # ---------- Case 2: 2D ----------
    if axis == 0:
        # ghost_inds and interior_inds refer to rows (top/bottom)
        if r_mask is None:
            # Apply to entire rows
            Tarr[ghost_inds, :] = 2.0 * T_bc - Tarr[interior_inds, :]
        else:
            # Apply only to columns where r_mask is True
            col_inds = np.where(r_mask)[0]
            Tarr[ghost_inds[:, None], col_inds[None, :]] = (
                2.0 * T_bc - Tarr[interior_inds[:, None], col_inds[None, :]]
            )

    else:
        # axis == 1: ghost_inds and interior_inds refer to columns (left/right)
        if z_mask is None:
            # Apply to entire columns
            Tarr[:, ghost_inds] = 2.0 * T_bc - Tarr[:, interior_inds]
        else:
            # Apply only to rows where z_mask is True
            row_inds = np.where(z_mask)[0]
            Tarr[row_inds[:, None], ghost_inds[None, :]] = (
                2.0 * T_bc - Tarr[row_inds[:, None], interior_inds[None, :]]
            )

    return Tarr


def apply_bc_dirichlet_mirror_disp(Uarr, ghost_inds, interior_inds, U_bc, axis, z_mask=None, r_mask=None):
    """
    Apply Dirichlet mirror boundary condition for displacement field:
    U_ghost = 2 * U_bc - U_interior

    Parameters: （略，和温度函数类似）

    Returns:
        Uarr : np.ndarray
            Displacement array after applying the boundary condition
    """
    if Uarr.ndim == 1:
        Uarr[ghost_inds] = 2.0 * U_bc - Uarr[interior_inds]
        return Uarr

    if axis == 0:
        if r_mask is None:
            Uarr[ghost_inds, :] = 2.0 * U_bc - Uarr[interior_inds, :]
        else:
            col_inds = np.where(r_mask)[0]
            Uarr[ghost_inds[:, None], col_inds[None, :]] = (
                2.0 * U_bc - Uarr[interior_inds[:, None], col_inds[None, :]]
            )
    else:
        if z_mask is None:
            Uarr[:, ghost_inds] = 2.0 * U_bc - Uarr[:, interior_inds]
        else:
            row_inds = np.where(z_mask)[0]
            Uarr[row_inds[:, None], ghost_inds[None, :]] = (
                2.0 * U_bc - Uarr[row_inds[:, None], interior_inds[None, :]]
            )
    return Uarr



def get_top_ghost_indices(z_all, ghost_nodes_z):
    """
    Compute indices of top ghost nodes and corresponding interior nodes.
    """
    ghost_inds_top = np.arange(ghost_nodes_z)  # e.g. [0, 1, 2, ...]
    interior_inds_top = 2 * ghost_nodes_z - ghost_inds_top - 1
    return ghost_inds_top, interior_inds_top


def get_bottom_ghost_indices(z_all, ghost_nodes_z):
    """
    Compute indices of bottom ghost nodes and corresponding interior nodes.
    """
    Nz_tot = len(z_all)
    ghost_inds_bottom = np.arange(Nz_tot - 1, Nz_tot - ghost_nodes_z - 1, -1)
    # e.g. if Nz_tot=50, ghost_nodes_z=3 => [49, 48, 47]

    z_local = np.arange(ghost_nodes_z)  # [0, 1, 2]
    interior_inds_bottom = (Nz_tot - 1) - (2 * ghost_nodes_z - z_local - 1)
    # Matches ghost_inds_bottom one-to-one

    return ghost_inds_bottom, interior_inds_bottom


def get_left_ghost_indices(r_all, ghost_nodes_x):
    """
    Compute indices of left ghost nodes and corresponding interior nodes.
    """
    ghost_inds_left = np.arange(ghost_nodes_x)
    # e.g. ghost_nodes_x=3 => [0, 1, 2]

    interior_inds_left = 2 * ghost_nodes_x - ghost_inds_left - 1
    # Mirror indices corresponding to the left ghost nodes

    return ghost_inds_left, interior_inds_left


def get_right_ghost_indices(r_all, ghost_nodes_x):
    """
    Compute indices of right ghost nodes and corresponding interior nodes.
    """
    Nr_tot = len(r_all)
    ghost_inds_right = np.arange(Nr_tot - 1, Nr_tot - ghost_nodes_x - 1, -1)
    # e.g. Nr_tot=50, ghost_nodes_x=3 => [49, 48, 47]

    x_local = np.arange(ghost_nodes_x)       # [0, 1, 2]
    offsets = 2 * ghost_nodes_x - 1 - 2 * x_local  # e.g. [5, 3, 1]
    interior_inds_right = ghost_inds_right - offsets

    return ghost_inds_right, interior_inds_right


def find_mirror_pairs(coords_ghost, coords_phys, dr):
    """
    Find mirrored index pairs between ghost and physical coordinates,
    assuming mirror symmetry along x-axis (left ghost region).

    Args:
        coords_ghost: ndarray of ghost node coordinates (on the left)
        coords_phys: ndarray of physical node coordinates (right)
        dr: spacing to determine match tolerance

    Returns:
        ghost_indices: ndarray of indices in coords_ghost
        phys_indices: ndarray of corresponding indices in coords_phys
    """
    ghost_indices = []
    phys_indices = []

    for i, (xg, zg) in enumerate(coords_ghost):
        # Mirror point along x-axis
        mirror_coord = (-xg, zg)

        # Compute distances from this mirror_coord to all physical points
        dists = np.linalg.norm(coords_phys - mirror_coord, axis=1)
        idx_phys = np.argmin(dists)

        if dists[idx_phys] < 0.1 * dr:
            ghost_indices.append(i)
            phys_indices.append(idx_phys)
        else:
            raise ValueError(f"No matching physical point found for ghost point {i} at {mirror_coord}")

    return np.array(ghost_indices), np.array(phys_indices)

def find_circle_mirror_pairs_multilayer(coords_ghost_circle, coords_phys, dr, R):
    """
    Find mirror pairs between ghost_circle points and physical region points,
    using radial symmetry across a circular arc boundary with multiple ghost layers.

    Parameters:
        coords_ghost_circle : np.ndarray of shape (N, 2)
            Coordinates of ghost nodes along the circular arc.
        coords_phys : np.ndarray of shape (M, 2)
            Coordinates of physical (interior) nodes.
        dr : float
            Grid spacing between particle centers.
        R : float
            Radius of the circular physical boundary (centered at (0, R)).

    Returns:
        ghost_indices : np.ndarray
            Indices of ghost points.
        phys_indices : np.ndarray
            Corresponding indices of physical mirror points.

    Raises:
        RuntimeError if any ghost point cannot find a valid mirror point.
    """
    ghost_indices = []
    phys_indices = []

    for i, (xg, zg) in enumerate(coords_ghost_circle):
        # Compute radial distance from ghost point to circle center (0, R)
        r_ghost = np.sqrt(xg**2 + (zg - R)**2)
        k = int(round((r_ghost - R) / dr))  # Determine ghost layer index

        # Compute unit outward normal vector from center to ghost point
        n = np.array([xg, zg - R]) / r_ghost

        # Reflect the ghost point inward across the circular boundary
        mirror_point = np.array([xg, zg]) - 2 * (k - 0.5) * dr * n

        # Find nearest physical point to the mirrored coordinate
        dists = np.linalg.norm(coords_phys - mirror_point, axis=1)
        idx_phys = np.argmin(dists)

        if dists[idx_phys] < 0.4 * dr:
            ghost_indices.append(i)
            phys_indices.append(idx_phys)
        else:
            raise RuntimeError(
                f"Failed to find mirror match for ghost point {i} "
                f"at ({xg:.4f}, {zg:.4f}), "
                f"expected mirror at ({mirror_point[0]:.4f}, {mirror_point[1]:.4f}), "
                f"closest match distance = {dists[idx_phys]:.4f}"
            )

    return np.array(ghost_indices), np.array(phys_indices)



