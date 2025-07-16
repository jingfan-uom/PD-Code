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

def generate_quarter_circle_coordinates(
    dr, R, Nr, ghost_nodes_r,
    r_ghost_left, z_ghost_bot,
    quarter_circle
):
    """
    Generate r_all and z_all
    including the ghost area, and then use mask to process the physical area and ghost area points.
    """

    if quarter_circle:
        # 1. r_all 和 z_all  (Center aligned)
        r_all = np.linspace(-ghost_nodes_r * dr + dr / 2, R + dr / 2 + (ghost_nodes_r - 1) * dr, Nr + 2 * ghost_nodes_r)
        z_all = np.linspace(R +(ghost_nodes_r - 1) * dr + dr / 2, -ghost_nodes_r * dr + dr / 2, Nr + 2 * ghost_nodes_r)

         # 2. Generate the entire grid
        rr, zz = np.meshgrid(r_all, z_all, indexing='xy')
        coords_all = np.column_stack([rr.ravel(), zz.ravel()])

       # 3. Physical zone mask (only for r, z both >= 0 and r^2+z^2 <= R^2)
        mask_phys = (
                (coords_all[:, 0] >= 0) & (coords_all[:, 1] >= 0) &
                (coords_all[:, 0] < R) & (coords_all[:, 1] < R) &
                (coords_all[:, 0] ** 2 + coords_all[:, 1] ** 2 <= R ** 2)
        )
        coords_phys = coords_all[mask_phys]

        # 4. Ghost Zone Left
        coords_ghost_left = np.empty((0, 2))
        if r_ghost_left :
            mask_ghost_left = (coords_all[:, 0] < 0) & (coords_all[:, 1] > 0)
            coords_ghost_left = coords_all[mask_ghost_left]

         # 5. Ghost Zone
        coords_ghost_bot = np.empty((0, 2))
        if z_ghost_bot :
            mask_ghost_bot = coords_all[:, 1] < 0
            #Note that here I have placed the lower left corner on the lower edge of the grain area.
            coords_ghost_bot = coords_all[mask_ghost_bot]

        # 6. quarter_circle ghost
        coords_ghost_circle = np.empty((0, 2))
        mask_circle = (
                (coords_all[:, 0] >= 0) & (coords_all[:, 1] >= 0) &
                (coords_all[:, 0] ** 2 + coords_all[:, 1] ** 2 > R ** 2 - 1e-8) &
                (coords_all[:, 0] ** 2 + coords_all[:, 1] ** 2 <= (R) ** 2 + 1e-8)
        )
        coords_ghost_circle = coords_all[mask_circle]

    else:
        # 1. r_all 和 z_all 
        R1 = R + ghost_nodes_r * dr
        Z1 = R + ghost_nodes_r * dr
        r_all = np.linspace(-ghost_nodes_r * dr + dr / 2, R - dr / 2, Nr + ghost_nodes_r)
        z_all = np.linspace(r - dr / 2, -ghost_nodes_r * dr + dr / 2, Nr + ghost_nodes_r)

        # 2. Generate the entire grid
        rr, zz = np.meshgrid(r_all, z_all, indexing='ij')
        coords_all = np.column_stack([rr.ravel(), zz.ravel()])

        # 3. Physical zone mask (only for r, z both >= 0 and r^2+z^2 <= R^2)
        mask_phys = (
                (coords_all[:, 0] >= 0) & (coords_all[:, 1] >= 0) &
                (coords_all[:, 0] < R) & (coords_all[:, 1] < R) &
                (coords_all[:, 0] ** 2 + coords_all[:, 1] ** 2 <= R ** 2)
        )
        coords_phys = coords_all[mask_phys]

        # 4. Ghost Zone Left
        coords_ghost_left = np.empty((0, 2))
        if r_ghost_left:
            mask_ghost_left = (coords_all[:, 0] < 0) & (coords_all[:, 1] > 0)
            coords_ghost_left = coords_all[mask_ghost_left]

              coords_ghost_bot = np.empty((0, 2))
        if z_ghost_bot:
            mask_ghost_bot = coords_all[:, 1] < 0
            # Note that here I have placed the lower left corner on the lower edge of the grain area.
            coords_ghost_bot = coords_all[mask_ghost_bot]

    return coords_phys, coords_ghost_left, coords_ghost_bot ,coords_ghost_circle

