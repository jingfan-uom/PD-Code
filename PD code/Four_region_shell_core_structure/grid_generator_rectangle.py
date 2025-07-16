import numpy as np
tol= 1e-8
def robust_key(r, z):
    return (round(r / tol * tol), round(z / tol * tol))


def get_coarse_neighbor_points(
        Rmat_coarse: np.ndarray,
        Zmat_coarse: np.ndarray,
        ghost_indices: np.ndarray,
        dr_fine: float,
        dz_fine: float,
        r_all_fine: np.ndarray,
        z_all_fine: np.ndarray,
        axis,ghost_nodes_z,ghost_nodes_x,
        tol: float = 1e-10,
):
    """
    For each ghost point on the coarse grid, map it to the 4 surrounding fine grid points.

    Returns:
        coarse2fine_indices: dict {(i_z_coarse, i_r_coarse): [(i_z_fine, i_r_fine), ...]}
    """
    Nz, Nr = Rmat_coarse.shape
    N_per_line = Nz - 2 * ghost_nodes_z if axis == 1 else Nr - 2 * ghost_nodes_x
    coarse2fine_indices = {}

    for idx in ghost_indices:
        for i in range(N_per_line):
            i_z_coarse, i_r_coarse = (i + ghost_nodes_z, idx) if axis == 1 else (idx, i + ghost_nodes_x)
            r = Rmat_coarse[i_z_coarse, i_r_coarse]
            z = Zmat_coarse[i_z_coarse, i_r_coarse]

            # Coordinates of four adjacent fine grid points
            neighbors = [
                (r - dr_fine / 2, z - dz_fine / 2),
                (r - dr_fine / 2, z + dz_fine / 2),
                (r + dr_fine / 2, z - dz_fine / 2),
                (r + dr_fine / 2, z + dz_fine / 2),
            ]

            fine_indices = []
            for r_n, z_n in neighbors:
                try:
                    i_r_fine = np.where(np.abs(r_all_fine - r_n) < tol)[0][0]
                    i_z_fine = np.where(np.abs(z_all_fine - z_n) < tol)[0][0]
                    fine_indices.append((i_z_fine, i_r_fine))
                except IndexError:
                    raise ValueError(f"Point ({r_n:.6f}, {z_n:.6f}) not found in fine grid.")

            coarse2fine_indices[(i_z_coarse, i_r_coarse)] = fine_indices
    print("Sample mapping from coarse points to fine neighbors:\n")
    for i, (coarse_idx, fine_list) in enumerate(coarse2fine_indices.items()):
        print(f"Coarse point {coarse_idx} maps to fine points: {fine_list}")
        if i >= 4:  # Print only the first 5
              break

    return coarse2fine_indices


def interpolate_temperature_for_coarse(
        T_coarse: np.ndarray,
        T_fine: np.ndarray,
        coarse2fine_indices: dict
) -> np.ndarray:
    """
    For ghost points in the coarse grid, interpolate based on the fine grid temperature.
    coarse2fine_indices: dict[(row_coarse, col_coarse)] = [(z1, r1), (z2, r2), ...]
    """
    T_result = T_coarse.copy()

    for coarse_key, fine_locs in coarse2fine_indices.items():
        fine_vals = [T_fine[i_z, i_r] for i_z, i_r in fine_locs]
        T_result[coarse_key] = np.mean(fine_vals)

    return T_result

    return T_coarse



def get_fine_neighbor_points(
    Rmat_fine: np.ndarray,
    Zmat_fine: np.ndarray,
    target_indices: np.ndarray,
    dr_fine: float,
    dz_fine: float,
    Rmat_coarse: np.ndarray,
    Zmat_coarse: np.ndarray,
    r_all_coarse: np.ndarray,
    z_all_coarse: np.ndarray,
    axis,ghost_nodes_z,ghost_nodes_x,
    tol=1e-8
):
    Nz, Nr = Rmat_fine.shape
    N_per_line = Nz - 2*ghost_nodes_z if axis == 1 else Nr - 2*ghost_nodes_x
    fine2coarse_indices = {}

    R_flat = Rmat_coarse.reshape(-1)
    Z_flat = Zmat_coarse.reshape(-1)
    coords_flat = np.stack([R_flat, Z_flat], axis=-1)

    radius = np.sqrt(10) / 2 * dr_fine + 1e-8

    for idx in target_indices:
        for i in range(N_per_line):
            i_z_fine, i_r_fine = (i + ghost_nodes_z, idx) if axis == 1 else (idx, i + ghost_nodes_x)

            r_fine = Rmat_fine[i_z_fine, i_r_fine]
            z_fine = Zmat_fine[i_z_fine, i_r_fine]

            # Searching for coarse points
            dists = np.sqrt((coords_flat[:, 0] - r_fine)**2 + (coords_flat[:, 1] - z_fine)**2)
            nearby_mask = (dists <= radius)
            nearby_coords = coords_flat[nearby_mask]

            if len(nearby_coords) != 3:
                raise ValueError(f"Expected exactly 3 coarse neighbors for fine point ({r_fine}, {z_fine}), found {len(nearby_coords)}.")

            coarse_indices_list = []

            for r_c, z_c in nearby_coords:
                try:
                    i_r_coarse = np.where(np.abs(r_all_coarse - r_c) < tol)[0][0]
                    i_z_coarse = np.where(np.abs(z_all_coarse - z_c) < tol)[0][0]
                except IndexError:
                    raise ValueError(f"Coarse point ({r_c:.6f}, {z_c:.6f}) not found in coarse grid arrays.")

                coarse_indices_list.append((i_z_coarse, i_r_coarse))

            fine2coarse_indices[(i_z_fine, i_r_fine)] = coarse_indices_list
    print("Fine to coarse indices (sample):")
    for (fine_idx, coarse_idxs) in list(fine2coarse_indices.items())[:3]:
        print(f"Fine point {fine_idx}: Coarse indices {coarse_idxs}")


    return fine2coarse_indices


def interpolate_temperature_for_fine(
        T_fine: np.ndarray,
        T_coarse: np.ndarray,
        fine2coarse_indices: dict
) -> np.ndarray:
    """
    For ghost points in the fine grid, interpolate based on the coarse grid temperature.
    fine2coarse_indices: dict[(row_fine, col_fine)] = [(z1, r1), (z2, r2), ...]
    """
    T_result = T_fine.copy()

    for fine_key, coarse_locs in fine2coarse_indices.items():
        coarse_vals = [T_coarse[i_z, i_r] for i_z, i_r in coarse_locs]
        T_result[fine_key] = np.mean(coarse_vals)

    return T_result

def get_exact_neighbor_points_same_spacing(
        Rmat: np.ndarray,
        Zmat: np.ndarray,
        ghost_indices: np.ndarray,
        r_all_target: np.ndarray,
        z_all_target: np.ndarray,
        axis,ghost_nodes_z,ghost_nodes_x,
        tol: float = 1e-10,
):
    """
    For same-resolution neighbor regions (e.g., region 2 → 3), find the 1-to-1 matching
    index on the target grid for each ghost node.

    Returns:
        mapping: dict {(i_z_src, i_r_src): (i_z_tgt, i_r_tgt)}
    """
    Nz, Nr = Rmat.shape
    N_per_line = Nz - 2 * ghost_nodes_z if axis == 1 else Nr - 2 * ghost_nodes_x
    mapping = {}

    for idx in ghost_indices:
        for i in range(N_per_line):
            i_z_src, i_r_src = (i + ghost_nodes_z, idx) if axis == 1 else (idx, i + ghost_nodes_x)
            r = Rmat[i_z_src, i_r_src]
            z = Zmat[i_z_src, i_r_src]

            # Find the corresponding point in the target grid
            try:
                i_r_tgt = np.where(np.abs(r_all_target - r) < tol)[0][0]
                i_z_tgt = np.where(np.abs(z_all_target - z) < tol)[0][0]
                mapping[(i_z_src, i_r_src)] = (i_z_tgt, i_r_tgt)
            except IndexError:
                raise ValueError(f"Matching point ({r:.6e}, {z:.6e}) not found in target grid.")

    print("Sample exact mapping from ghost to target points:\n")
    for i, (src_idx, tgt_idx) in enumerate(mapping.items()):
        print(f"Ghost point {src_idx} → Target point {tgt_idx}")
        if i >= 4: break

    return mapping


def interpolate_temperature_direct_match(
        T_src: np.ndarray,
        T_target: np.ndarray,
        index_mapping: dict
) -> np.ndarray:

    for (i_z_src, i_r_src), (i_z_tgt, i_r_tgt) in index_mapping.items():
        T_src[i_z_src, i_r_src] = T_target[i_z_tgt, i_r_tgt]
    return T_src
