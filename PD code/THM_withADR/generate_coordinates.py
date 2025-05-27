import numpy as np

def generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz,
    ghost_nodes_x, ghost_nodes_z,
    r_ghost_left, r_ghost_right,
    z_ghost_top, z_ghost_bot
):
    """
    Generate r and z coordinate arrays including ghost regions. 
    Each ghost region can be selectively added based on boolean flags.

    Parameters:
    - r_start, z_start: starting coordinates of the physical domain
    - dr, dz: grid spacing in r and z directions
    - Lr, Lz: length of the physical domain in r and z directions
    - Nr, Nz: number of grid nodes in the physical domain
    - ghost_nodes_x, ghost_nodes_z: number of ghost nodes in r and z directions
    - r_ghost_left, r_ghost_right, z_ghost_top, z_ghost_bot: 
        booleans indicating whether to include ghost regions on each side

    Returns:
    - r_all, z_all: coordinate arrays including ghost nodes
    - Nr_tot, Nz_tot: total number of nodes including ghost regions
    """

    # r-direction (horizontal)
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

    # z-direction (vertical, reversed order: top to bottom)
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
