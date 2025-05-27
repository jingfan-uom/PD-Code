import numpy as np

def generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz,
    ghost_nodes_x, ghost_nodes_z,
    r_ghost_left, r_ghost_right,
    z_ghost_top, z_ghost_bot
):
    """
    生成包含 ghost 区域的 r 和 z 坐标数组，根据布尔值参数控制是否生成对应的 ghost 区域。

    参数：
    - r_start, z_start: 物理域的起始坐标
    - dr, dz: 网格间距
    - Lr, Lz: 物理域长度
    - Nr, Nz: 物理域中的节点数
    - ghost_nodes_x, ghost_nodes_z: 每个方向上的 ghost 节点数量
    - r_ghost_left, r_ghost_right, z_ghost_top, z_ghost_bot: 是否生成对应方向的 ghost 区域（布尔值）

    返回：
    - r_all, z_all: 包含 ghost 节点的 r 和 z 坐标数组
    - Nr_tot, Nz_tot: 包括 ghost 节点的总节点数
    """

    # r方向
    r_phys = np.linspace(r_start + dr / 2, r_start + Lr - dr / 2, Nr)

    r_parts = []
    if r_ghost_left:
        r_ghost_l = np.linspace(r_start - ghost_nodes_x * dr + dr / 2, r_start - dr / 2, ghost_nodes_x)
        r_parts.append(r_ghost_l)
    r_parts.append(r_phys)
    if r_ghost_right:
        r_ghost_r = np.linspace(
            r_start + Lr + dr / 2,
            r_start + Lr + dr / 2 + (ghost_nodes_x - 1) * dr,
            ghost_nodes_x
        )
        r_parts.append(r_ghost_r)

    r_all = np.concatenate(r_parts)
    Nr_tot = len(r_all)

    # z方向
    z_phys = np.linspace(z_start + Lz - dz / 2, z_start + dz / 2, Nz)

    z_parts = []
    if z_ghost_top:
        z_ghost_t = np.linspace(
            z_start + Lz + (ghost_nodes_z - 1) * dz + dz / 2,
            z_start + Lz + dz / 2,
            ghost_nodes_z
        )
        z_parts.append(z_ghost_t)
    z_parts.append(z_phys)
    if z_ghost_bot:
        z_ghost_b = np.linspace(
            z_start - dz / 2,
            z_start - ghost_nodes_z * dz + dz / 2,
            ghost_nodes_z
        )
        z_parts.append(z_ghost_b)

    z_all = np.concatenate(z_parts)
    Nz_tot = len(z_all)

    return r_all, z_all, Nr_tot, Nz_tot
