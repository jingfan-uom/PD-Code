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
        axis: int = 0,
        tol: float = 1e-10,
):
    """
    Generate neighbor points for ghost nodes and build coord2index map from fine grid.

    Returns:
        neighbor_points: ndarray of shape (N_ghost, 4, 2)
        coord2index: dict {robust_key(r, z): (i_z, i_r)} based on r_all_fine/z_all_fine
    """
    Nz, Nr = Rmat_coarse.shape
    N_per_line = Nz - 6 if axis == 0 else Nr - 6
    N_ghost = len(ghost_indices) * N_per_line

    neighbor_points = np.empty((N_ghost, 4, 2))
    coord2index = {}

    index = 0
    for idx in ghost_indices:
        for i in range(N_per_line):
            if axis == 0:
                r = Rmat_coarse[i + 3, idx]
                z = Zmat_coarse[i + 3, idx]
            else:
                r = Rmat_coarse[idx, i + 3]
                z = Zmat_coarse[idx, i + 3]

            # 定义四个邻接点
            neighbors = [
                (r - dr_fine / 2, z - dz_fine / 2),
                (r - dr_fine / 2, z + dz_fine / 2),
                (r + dr_fine / 2, z - dz_fine / 2),
                (r + dr_fine / 2, z + dz_fine / 2),
            ]

            neighbor_points[index] = neighbors

            # 为这4个点创建坐标索引（在细网格上）
            for r_n, z_n in neighbors:
                key = robust_key(r_n, z_n)
                # 只添加一次
                if key not in coord2index:
                    # 找对应的 r, z 索引（在一维坐标向量中）
                    try:
                        i_r = np.where(np.abs(r_all_fine - r_n) < tol)[0][0]
                        i_z = np.where(np.abs(z_all_fine - z_n) < tol)[0][0]
                        coord2index[key] = (i_z, i_r)
                    except IndexError:
                        raise ValueError(f"Point ({r_n:.6f}, {z_n:.6f}) not found in fine grid.")

            index += 1

    return neighbor_points, coord2index


def interpolate_temperature_for_coarse(
        T_coarse: np.ndarray,
        neighbor_points: np.ndarray,
        T_fine: np.ndarray,
        coord2index: dict,
        ghost_indices: np.ndarray,
        axis,
        ):

    Nz_coarse, Nr_coarse = T_coarse.shape
    N_per_line = Nz_coarse-6 if axis == 0 else Nr_coarse-6

    index = 0
    for idx in ghost_indices:
        for i in range(N_per_line):
            temps = []
            for r, z in neighbor_points[index]:
                key = robust_key(r, z)
                if key in coord2index:
                    idx_z, idx_r = coord2index[key]
                    temps.append(T_fine[idx_z, idx_r])
                else:
                    raise ValueError(f"Coordinate {key} not found in fine grid.")
            if axis == 0:
                row, col = i+3, idx
            else:
                row, col = idx, i+3

            T_coarse[row, col] = np.mean(temps)
            index += 1

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
    axis: int = 0,
):
    Nz, Nr = Rmat_fine.shape
    N_per_line = Nz - 6 if axis == 0 else Nr - 6
    N_target = len(target_indices) * N_per_line

    neighbor_points = np.empty((N_target, 3, 2))
    coord2index = {}

    R_flat = Rmat_coarse.reshape(-1)
    Z_flat = Zmat_coarse.reshape(-1)
    coords_flat = np.stack([R_flat, Z_flat], axis=-1)

    index = 0
    radius = np.sqrt(10) / 2 * dr_fine + 1e-8

    for idx in target_indices:
        for i in range(N_per_line):
            if axis == 0:
                r = Rmat_fine[i + 3, idx]
                z = Zmat_fine[i + 3, idx]
            else:
                r = Rmat_fine[idx, i + 3]
                z = Zmat_fine[idx, i + 3]

            dists = np.sqrt((coords_flat[:, 0] - r) ** 2 + (coords_flat[:, 1] - z) ** 2)
            nearby_mask = (dists <= radius)
            nearby_coords = coords_flat[nearby_mask]

            if len(nearby_coords) < 3:
                raise ValueError(f"Not enough coarse neighbors found for fine point ({r}, {z})")

            neighbor_points[index] = nearby_coords

            for r_c, z_c in nearby_coords:
                key = robust_key(r_c, z_c)
                if key not in coord2index:
                    try:
                        i_r = np.where(np.abs(r_all_coarse - r_c) < tol)[0][0]
                        i_z = np.where(np.abs(z_all_coarse - z_c) < tol)[0][0]
                        coord2index[key] = (i_z, i_r)
                    except IndexError:
                        raise ValueError(f"Coarse point ({r_c:.6f}, {z_c:.6f}) not found in grid.")

            index += 1

    return neighbor_points, coord2index

def interpolate_temperature_for_fine(
    T_fine: np.ndarray,
    neighbor_points: np.ndarray,
    T_coarse: np.ndarray,
    coord2index: dict,
    ghost_indices: np.ndarray,
    axis: int,
):
    Nz_fine, Nr_fine = T_fine.shape
    N_per_line = Nz_fine - 6 if axis == 0 else Nr_fine - 6
    T_result = T_fine.copy()

    index = 0
    for idx in ghost_indices:
        for i in range(N_per_line):
            if axis == 0:
                row, col = i+3, idx
            else:
                row, col = idx, i+3

            tri_coords = neighbor_points[index]
            coords = []
            values = []

            for r, z in tri_coords:
                key = robust_key(r, z)
                if key not in coord2index:
                    raise ValueError(f"Neighbor coordinate {key} not found in coord2index.")
                i_z, i_r = coord2index[key]
                coords.append([r, z])
                values.append(T_coarse[i_z, i_r])

            coords = np.array(coords)
            values = np.array(values)

            r_target = np.mean(coords[:, 0])  # 插值中心点坐标（简化）
            z_target = np.mean(coords[:, 1])

            A = np.array([
                [coords[0][0], coords[1][0], coords[2][0]],
                [coords[0][1], coords[1][1], coords[2][1]],
                [1.0,         1.0,         1.0        ]
            ])
            b = np.array([r_target, z_target, 1.0])
            lambdas = np.linalg.solve(A, b)

            T_result[row, col] = np.dot(lambdas, values)
            index += 1

    return T_result
