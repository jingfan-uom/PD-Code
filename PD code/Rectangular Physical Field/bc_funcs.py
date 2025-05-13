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


def apply_bc_dirichlet_displacement(Uarr, ghost_inds, interior_inds, U_target, axis):
    """
    对指定边界施加镜像型 Dirichlet 位移边界条件。

    参数：
    - Uarr: 当前位移场 (二维数组，例如 Ur 或 Uz)
    - ghost_inds: ghost 区域粒子索引（列表或数组）
    - interior_inds: ghost 区域镜像对应的 interior 粒子索引（列表或数组）
    - U_target: 目标边界位移值（标量，例如 0.0 或 0.001）
    - axis: 方向，0 表示 z 方向（上下边界），1 表示 r 方向（左右边界）

    返回：
    - 更新后的 Uarr（已在 ghost 区域施加边界）
    """
    # 遍历 ghost 行或列
    if axis == 0:
        # z方向：逐列处理每个 ghost 行
        for g_idx, i_idx in zip(ghost_inds, interior_inds):
            Uarr[g_idx, :] = 2 * U_target - Uarr[i_idx, :]
    elif axis == 1:
        # r方向：逐行处理每个 ghost 列
        for g_idx, i_idx in zip(ghost_inds, interior_inds):
            Uarr[:, g_idx] = 2 * U_target - Uarr[:, i_idx]
    else:
        raise ValueError("axis 只能为 0（z方向）或 1（r方向）")

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
