import numpy as np

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
        axis,
        tol: float = 1e-10,
):
    """
    For each ghost point on the coarse grid, map it to the 4 surrounding fine grid points.

    Returns:
        coarse2fine_indices: dict {(i_z_coarse, i_r_coarse): [(i_z_fine, i_r_fine), ...]}
    """
    Nz, Nr = Rmat_coarse.shape
    N_per_line = Nz - 6 if axis == 1 else Nr - 6
    coarse2fine_indices = {}

    for idx in ghost_indices:
        for i in range(N_per_line):
            i_z_coarse, i_r_coarse = (i + 3, idx) if axis == 1 else (idx, i + 3)
            r = Rmat_coarse[i_z_coarse, i_r_coarse]
            z = Zmat_coarse[i_z_coarse, i_r_coarse]

            # 四个相邻细网格点的坐标
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
    coarse2fine_indices: dict,
    ghost_indices: np.ndarray,
    axis
):
    Nz_coarse, Nr_coarse = T_coarse.shape
    N_per_line = Nz_coarse - 6 if axis == 1 else Nr_coarse - 6

    for idx in ghost_indices:
        for i in range(N_per_line):
            row, col = (i + 3, idx) if axis == 1 else (idx, i + 3)
            coarse_key = (row, col)

            if coarse_key not in coarse2fine_indices:
                raise ValueError(f"Coarse point {coarse_key} not found in coarse2fine_indices.")

            fine_indices = coarse2fine_indices[coarse_key]
            temps = [T_fine[i_z, i_r] for i_z, i_r in fine_indices]

            T_coarse[row, col] = np.mean(temps)

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
    axis,
    tol=1e-8
):
    Nz, Nr = Rmat_fine.shape
    N_per_line = Nz - 6 if axis == 1 else Nr - 6
    fine2coarse_indices = {}

    R_flat = Rmat_coarse.reshape(-1)
    Z_flat = Zmat_coarse.reshape(-1)
    coords_flat = np.stack([R_flat, Z_flat], axis=-1)

    radius = np.sqrt(10) / 2 * dr_fine + 1e-8

    for idx in target_indices:
        for i in range(N_per_line):
            i_z_fine, i_r_fine = (i + 3, idx) if axis == 1 else (idx, i + 3)

            r_fine = Rmat_fine[i_z_fine, i_r_fine]
            z_fine = Zmat_fine[i_z_fine, i_r_fine]

            # Look for the coarse points.
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
    fine2coarse_indices: dict,
    ghost_indices: np.ndarray,
    axis
):
    T_result = T_fine.copy()

    Nz_fine, Nr_fine = T_fine.shape
    N_per_line = Nz_fine - 6 if axis == 1 else Nr_fine - 6

    for idx in ghost_indices:
        for i in range(N_per_line):
            if axis == 1:
                row, col = i + 3, idx
            else:
                row, col = idx, i + 3

            fine_key = (row, col)

            if fine_key not in fine2coarse_indices:
                raise ValueError(f"Fine point {fine_key} not found in fine2coarse_indices.")

            coarse_indices = fine2coarse_indices[fine_key]

            # Extract the corresponding temperature values from coarse indices
            coarse_temps = [T_coarse[i_z, i_r] for i_z, i_r in coarse_indices]

            # Take the average directly as the temperature of the fine point
            T_result[row, col] = np.mean(coarse_temps)

    return T_result

