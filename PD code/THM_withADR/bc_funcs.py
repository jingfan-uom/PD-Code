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
                2.0 * T_bc  - Tarr[interior_inds[:, None], col_inds[None, :]]
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


def apply_bc_free_boundary(Uarr, ghost_inds, interior_inds, axis):
    """
    Apply free boundary condition (zero transverse shear): U_ghost = U_interior

    Parameters:
    -----------
    Uarr : np.ndarray
        Displacement array (e.g., Uz or Ur), either 1D (Nr_tot * Nz_tot,) or 2D (Nz_tot, Nr_tot)

    ghost_inds : array-like or tuple
        Indices of ghost nodes. If axis=0, they refer to rows; if axis=1, they refer to columns.

    interior_inds : array-like or tuple
        Indices of the interior nodes adjacent to ghost nodes.

    axis : int
        0 for top/bottom (rows), 1 for left/right (columns)

    Returns:
    --------
    Uarr : np.ndarray
        Updated displacement array after applying free boundary condition.
    """
    if Uarr.ndim == 1:
        raise ValueError("Uarr should be reshaped to 2D before applying free boundary condition.")

    if axis == 1:
        # Row-wise (top/bottom ghost layer)
        Uarr[ghost_inds, :] = Uarr[interior_inds, :]
    elif axis == 0:
        # Column-wise (left/right ghost layer)
        Uarr[:, ghost_inds] = Uarr[:, interior_inds]
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    return Uarr


def get_left_ghost_indices(r_all, ghost_nodes_x, Nz_tot):
    """
    Compute left ghost node indices and corresponding interior node indices.
    """
    Nr_tot = len(r_all)
    ghost_inds_left = np.arange(ghost_nodes_x)
    interior_inds_left = 2 * ghost_nodes_x - ghost_inds_left - 1

    # Flattened 1D indices
    ghost_inds_left_1d = [i * Nr_tot + j for i in range(Nz_tot) for j in range(ghost_nodes_x)]
    return ghost_inds_left, interior_inds_left, ghost_inds_left_1d


def get_right_ghost_indices(r_all, ghost_nodes_x, Nz_tot):
    """
    Compute right ghost node indices and corresponding interior node indices.
    """
    Nr_tot = len(r_all)
    ghost_inds_right = np.arange(Nr_tot - 1, Nr_tot - ghost_nodes_x - 1, -1)
    x_local = np.arange(ghost_nodes_x)
    offsets = 2 * ghost_nodes_x - 1 - 2 * x_local
    interior_inds_right = ghost_inds_right - offsets

    # Flattened 1D indices
    ghost_inds_right_1d = [i * Nr_tot + j for i in range(Nz_tot) for j in range(Nr_tot - ghost_nodes_x, Nr_tot)]
    return ghost_inds_right, interior_inds_right, ghost_inds_right_1d


def get_top_ghost_indices(z_all, ghost_nodes_z, Nr_tot):
    """
    Compute top ghost node indices and corresponding interior node indices.
    """
    ghost_inds_top = np.arange(ghost_nodes_z)
    interior_inds_top = 2 * ghost_nodes_z - ghost_inds_top - 1

    # Flattened 1D indices

    ghost_inds_top_1d = [i * Nr_tot + j for i in range(ghost_nodes_z) for j in range(Nr_tot)]

    return ghost_inds_top, interior_inds_top, ghost_inds_top_1d


def get_bottom_ghost_indices(z_all, ghost_nodes_z, Nr_tot):
    """
    Compute bottom ghost node indices and corresponding interior node indices.
    """
    Nz_tot = len(z_all)
    ghost_inds_bottom = np.arange(Nz_tot - 1, Nz_tot - ghost_nodes_z - 1, -1)
    z_local = np.arange(ghost_nodes_z)
    interior_inds_bottom = (Nz_tot - 1) - (2 * ghost_nodes_z - z_local - 1)

    # Flattened 1D indices
    ghost_inds_bottom_1d = [i * Nr_tot + j for i in range(Nz_tot - ghost_nodes_z, Nz_tot) for j in range(Nr_tot)]

    return ghost_inds_bottom, interior_inds_bottom, ghost_inds_bottom_1d

