
import numpy as np
# Apply mixed boundary conditions (mirror and Dirichlet) to temperature array

def apply_bc_zero_flux(Tarr, ghost_inds, interior_inds, axis):
    """
    axis=0 表示 ghost_inds 是行索引
    axis=1 表示 ghost_inds 是列索引
    """
    if Tarr.ndim == 1:
        # 一维情况
        Tarr[ghost_inds] = Tarr[interior_inds]
    else:
        if axis == 0:
            # 行
            Tarr[ghost_inds, :] = Tarr[interior_inds, :]
        else:
            # 列
            Tarr[:, ghost_inds] = Tarr[:, interior_inds]
    return Tarr

def apply_bc_dirichlet_mirror(Tarr, ghost_inds, interior_inds, T_bc,
                              axis=0, z_mask=None, r_mask=None):
    """
    对幽单元(ghost_inds)施加Dirichlet镜像边界: T_ghost = 2*T_bc - T_interior.

    参数说明:
    ----------
    Tarr : np.ndarray
        温度场数组, 可能是一维 (Nr_tot,) 或二维 (Nz_tot, Nr_tot).
    ghost_inds : 1D array
        幽单元索引。如果 axis=0, 表示行索引(顶部/底部); axis=1, 表示列索引(左/右).
    interior_inds : 1D array
        与 ghost_inds 对应的内部索引.
    T_bc : float
        Dirichlet 边界温度(定值).
    axis : int, {0, 1}, optional
        0 表示对行操作(顶部或底部), 1 表示对列操作(左或右).
        默认为 0.
    z_mask : 1D bool array, optional
        当 axis=1 时, 可以只对 z_mask=True 的行施加 Dirichlet.
    r_mask : 1D bool array, optional
        当 axis=0 时, 可以只对 r_mask=True 的列施加 Dirichlet.

    返回:
    ----------
    Tarr : np.ndarray
        已施加边界条件的温度场。
    """

    # ------------------------------------------------
    # 情况1: 一维
    # ------------------------------------------------
    if Tarr.ndim == 1:
        # 只有 r 方向, 没有 z
        # 直接用镜像法
        Tarr[ghost_inds] = 2.0 * T_bc - Tarr[interior_inds]
        return Tarr

    # ------------------------------------------------
    # 情况2: 二维
    # ------------------------------------------------
    if axis == 0:
        # ghost_inds / interior_inds 表示【行】索引 => 顶部/底部
        if r_mask is None:
            # 整个行施加镜像Dirichlet
            Tarr[ghost_inds, :] = 2.0 * T_bc - Tarr[interior_inds, :]
        else:
            # 只对列满足 r_mask 的位置施加Dirichlet
            col_inds = np.where(r_mask)[0]
            Tarr[ghost_inds[:, None], col_inds[None, :]] = (
                2.0 * T_bc - Tarr[interior_inds[:, None], col_inds[None, :]]
            )

    else:
        # axis == 1, ghost_inds / interior_inds 表示【列】索引 => 左侧/右侧
        if z_mask is None:
            # 整个列施加镜像Dirichlet
            Tarr[:, ghost_inds] = 2.0 * T_bc - Tarr[:, interior_inds]
        else:
            # 只对行满足 z_mask 的位置施加Dirichlet
            row_inds = np.where(z_mask)[0]
            Tarr[row_inds[:, None], ghost_inds[None, :]] = (
                2.0 * T_bc - Tarr[row_inds[:, None], interior_inds[None, :]]
            )

    return Tarr



def get_top_ghost_indices(z_all, ghost_nodes_z):

    ghost_inds_top = np.arange(ghost_nodes_z)  # 例如 [0,1,2,...]
    interior_inds_top = 2 * ghost_nodes_z - ghost_inds_top - 1
    return ghost_inds_top, interior_inds_top


def get_bottom_ghost_indices(z_all, ghost_nodes_z):
    """
    计算底部(下边界)幽单元及对应内部索引
    """
    Nz_tot = len(z_all)
    ghost_inds_bottom = np.arange(Nz_tot - 1, Nz_tot - ghost_nodes_z - 1, -1)
    # 例如如果 Nz_tot=50, ghost_nodes_z=3 => [49,48,47]

    z_local = np.arange(ghost_nodes_z)  # [0,1,2]
    interior_inds_bottom = (Nz_tot - 1) - (2 * ghost_nodes_z - z_local - 1)
    # 与 ghost_inds_bottom 一一对应

    return ghost_inds_bottom, interior_inds_bottom


def get_left_ghost_indices(r_all, ghost_nodes_x):
    """
    计算左边界幽单元及对应内部索引
    """
    ghost_inds_left = np.arange(ghost_nodes_x)
    # 例如 ghost_nodes_x=3 => ghost_inds_left=[0,1,2]

    interior_inds_left = 2 * ghost_nodes_x - ghost_inds_left - 1
    # 与 ghost_inds_left 对应的镜像索引

    return ghost_inds_left, interior_inds_left


def get_right_ghost_indices(r_all, ghost_nodes_x):
    """
    计算右边界幽单元及对应内部索引
    """
    Nr_tot = len(r_all)
    ghost_inds_right = np.arange(Nr_tot - 1, Nr_tot - ghost_nodes_x - 1, -1)
    # 例如如果 Nr_tot=50, ghost_nodes_x=3 => [49,48,47]

    x_local = np.arange(ghost_nodes_x)  # [0,1,2]
    offsets = 2 * ghost_nodes_x - 1 - 2 * x_local  # [5,3,1] => 用于镜像
    interior_inds_right = ghost_inds_right - offsets

    return ghost_inds_right, interior_inds_right
