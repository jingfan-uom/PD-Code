import numpy as np


def find_mirror_pairs(coords_ghost, coords_phys,tol):
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

        if dists[idx_phys] < tol:
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
        r_ghost = np.sqrt( xg ** 2 + (zg - R) ** 2 )
        k = int(round((r_ghost - R) / dr))  # Determine ghost layer index
        # Compute unit outward normal vector from center to ghost point
        n = np.array([xg, zg - R]) / r_ghost

        # Reflect the ghost point inward across the circular boundary
        mirror_point = np.array([xg, zg]) - 2 * (k - 0.5) * dr * n

        # Find nearest physical point to the mirrored coordinate
        dists = np.linalg.norm(coords_phys - mirror_point, axis=1)
        idx_phys = np.argmin(dists)

        ghost_indices.append(i)
        phys_indices.append(idx_phys)

    return np.array(ghost_indices), np.array(phys_indices)

def get_same_neighbor_points(
    ghost_coords: np.ndarray,
    coords: np.ndarray,
    tol: float = 1e-8
):
    """
    Check whether ghost points and phys points are exactly matched in the r-z coordinates (ignoring layer_id).

    Parameters:
        ghost_coords: shape=(N1, 3), including r, z, layer_id
        phys_coords: shape=(N2, 3), including r, z, layer_id
        tol: Allowable coordinate tolerance

    Returns:
        ghost_indices: indices of matching points in ghost
        phys_indices: indices of corresponding matching points in phys
    """
    ghost_indices = []
    phys_indices = []

    # Match only r and z coordinates
    ghost_rz = ghost_coords[:, :2]
    phys_rz = coords[:, :2]

    for i, g in enumerate(ghost_rz):
        diffs = np.linalg.norm(phys_rz - g, axis=1)
        match_idx = np.where(diffs < tol)[0]
        if match_idx.size == 0:
            raise ValueError(f"No matching point found for ghost point {i}: {g}")
        elif match_idx.size > 1:
            raise ValueError(f"Multiple matches found for ghost point {i}: {g}")
        ghost_indices.append(i)
        phys_indices.append(match_idx[0])

    return ghost_indices, phys_indices


def get_fine_neighbor_points(
    ghost_coords: np.ndarray,
    phys_coords: np.ndarray,
    dr_fine: float,
    tol: float = 1e-12
):
    ghost_indices = []          # ghost Point guidance
    phys_indices = []      # List of physical point guides corresponding to each ghost

    for i, g in enumerate(ghost_coords):
        r, z = g

        # Construct the coordinates of the four target physical points surrounding the ghost point.
        neighbor_targets = np.array([
            [r - dr_fine / 2, z - dr_fine / 2],
            [r - dr_fine / 2, z + dr_fine / 2],
            [r + dr_fine / 2, z - dr_fine / 2],
            [r + dr_fine / 2, z + dr_fine / 2],
        ])

        matched_indices = []
        for target in neighbor_targets:
            diffs = np.linalg.norm(phys_coords - target, axis=1)
            idx = np.where(diffs < tol)[0]
            if idx.size == 1:
                matched_indices.append(idx[0])
            elif idx.size == 0:
                continue
            else:
                raise ValueError(f"Target {target} matched multiple physical points: {idx}")

        if len(matched_indices) < 2:
            raise ValueError(f"Ghost point {g} matched less than 2 physical points.")

        if len(matched_indices) > 4:
            raise ValueError(f"Ghost point {g} matched over than 4 physical points.")

        ghost_indices.append(i)
        phys_indices.append(np.array(matched_indices, dtype=int))

    return ghost_indices, phys_indices



def get_coarse_neighbor_points(
    ghost_coords: np.ndarray,
    phys_coords: np.ndarray,
    dr_fine: float,
    tol: float = 1e-12
):
    radius = np.sqrt(10) / 2 * dr_fine + tol

    ghost_indices = []
    phys_indices_list = []
    dist_list = []

    for i, g in enumerate(ghost_coords):
        r_fine, z_fine = g

        # Calculate the distance from each coarse point to the ghost point
        dists = np.sqrt((phys_coords[:, 0] - r_fine)**2 + (phys_coords[:, 1] - z_fine)**2)
        nearby_mask = dists <= radius
        nearby_indices = np.where(nearby_mask)[0]
        nearby_dists = dists[nearby_indices]

        ghost_indices.append(i)
        phys_indices_list.append(nearby_indices)
        dist_list.append(nearby_dists)

    return ghost_indices, phys_indices_list, dist_list


def interpolate_temperature_for_fine(
    T_ghost: np.ndarray,
    T_fine: np.ndarray,
    ghost_indices: np.ndarray,
    phys_indices: np.ndarray
):
    T_result = T_ghost.copy()

    for i, g_idx in enumerate(ghost_indices):
        p_list = phys_indices[i]
        T_result[g_idx] = np.mean([T_fine[j] for j in p_list])

    return T_result

def interpolate_temperature_for_coarse(
    T_ghost: np.ndarray,
    T_coarse: np.ndarray,
    ghost_indices: np.ndarray,
    phys_indices: list,
    dist_list: list
):

    T_result = T_ghost.copy()

    for g_idx, p_idx, dists in zip(ghost_indices, phys_indices, dist_list):

        weights = 1.0 / dists
        weights /= np.sum(weights)
        T_result[g_idx] = np.sum(weights * T_coarse[p_idx])

    return T_result


def interpolate_temperature_for_same(
    T_ghost: np.ndarray,
    T_phys: np.ndarray,
    ghost_indices: np.ndarray,
    phys_indices: np.ndarray
) -> np.ndarray:
    T_result = T_ghost.copy()
    T_result[ghost_indices] = T_phys[phys_indices]
    return T_result


def get_void_z_range_from_fraction( D, t, f_void):

    R_o = D
    R_i = R_o - t           # 内核半径

    V_inner = (4.0 / 3.0) * np.pi * R_i**3     # 内核体积
    V_void = f_void * V_inner                 # 空腔体积

    # 解球冠体积公式 V = π h² (3R - h) / 3，求 h
    def V_cap(h): return (np.pi / 3.0) * (3 * R_i * h**2 - h**3)

    a, b = 0.0, R_i
    for _ in range(100):
        m = 0.5 * (a + b)
        if V_cap(m) < V_void:
            a = m
        else:
            b = m
    h = 0.5 * (a + b)

    z0 = R_i - h   # 空腔底部
    z1 = R_i       # 空腔顶部（即内核顶部）

    return z0, z1


def get_void_z_range_from_latent(latent_base, latent_capsule, rho_s, rho_shell, D, t_shell):
    """
    根据微胶囊潜热计算空腔在 z 方向的范围 z0 ~ z1。
    所有长度单位应为米，密度单位 kg/m³，潜热单位 J/g。
    """
    # 1. 内核半径（m）
    R_i = (D - 2 * t_shell) / 2.0

    # 2. 计算空腔体积比例（假设潜热损失仅由空腔造成）
    latent_ratio = latent_capsule / latent_base
    volume_void_ratio = 1 - latent_ratio  # 空腔体积分数，例如 1 - 455.7/515 ≈ 0.114

    # 3. 计算空腔体积（单位：m³）
    V_inner = (4 / 3) * np.pi * R_i**3  # 内核总体积（m³）
    V_void = volume_void_ratio * V_inner  # 空腔体积

    # 4. 定义球冠体积函数
    def V_cap(h):
        return (np.pi / 3.0) * (3 * R_i * h**2 - h**3)

    # 5. 用二分法反解 h，使 V_cap(h) ≈ V_void
    a, b = 0.0, R_i
    for _ in range(100):
        m = 0.5 * (a + b)
        if V_cap(m) < V_void:
            a = m
        else:
            b = m
    h = 0.5 * (a + b)

    # 6. z 轴方向范围：空腔底部 ~ 内核顶部
    z1 = R_i
    z0 = R_i - h

    return z0, z1

import numpy as np

def build_axis_firstlayer_pair_mask(coords_all_t, dr, z_tol=None):
    """
    找到最靠近 r=0 的一层（r ≈ r_min）与 r ≈ r_min+dr 的那一层，
    并按 z 一一对应，返回两个等长索引数组 (idx_inner, idx_outer)。

    coords_all_t: (N,2) [r,z]
    dr: 径向层间距
    z_tol: z 匹配容差，None 时默认 0.25*dr（你也可以传 dz/10）
    """
    r = coords_all_t[:, 0]
    z = coords_all_t[:, 1]

    if z_tol is None:
        z_tol = 0.25 * dr

    # 只考虑 r>0 的点（排除轴线 r=0 或数值噪声）
    pos = r > 0.0
    r_pos = r[pos]
    if r_pos.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    # 第一层：r 最小的那一圈（用容差聚类）
    r_min = r_pos.min()
    r_tol = 0.25 * dr
    mask1 = np.abs(r - r_min) <= r_tol

    # 第二层：r_min + dr 那一圈
    r2 = r_min + dr
    mask2 = np.abs(r - r2) <= r_tol

    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]
    if idx1.size == 0 or idx2.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    # --- 按 z 一一对应 ---
    # 用“z 分桶/唯一化”保证一一对应（对浮点用 tol）
    # 做法：对第二层建立 z->index 的映射（同一 z 可能有多个点时选 r 最接近 r2 的那个）
    idx_outer_match = np.empty(idx1.size, dtype=np.int64)

    z2 = z[idx2]
    r2_vals = r[idx2]

    for k, i1 in enumerate(idx1):
        zi = z[i1]
        # 找到第二层里 z 最接近的点
        jloc = np.argmin(np.abs(z2 - zi))
        if np.abs(z2[jloc] - zi) > z_tol:
            # 找不到合适 z，就跳过：这里我选择不配对（返回空会更安全）
            # 你也可以改成 idx_outer_match[k] = idx2[jloc] 强行配
            return np.empty(0, np.int64), np.empty(0, np.int64)
        idx_outer_match[k] = idx2[jloc]

    # 可能出现多个 idx1 指向同一个 idx2（z 重复/容差问题）
    # 我们做一次去重：按 idx2 去重保留第一个（或你也可以用更严格的 z 量化）
    seen = {}
    keep_i1 = []
    keep_i2 = []
    for i1, i2 in zip(idx1, idx_outer_match):
        if i2 in seen:
            continue
        seen[i2] = 1
        keep_i1.append(i1)
        keep_i2.append(i2)

    return np.array(keep_i1, dtype=np.int64), np.array(keep_i2, dtype=np.int64)

def apply_T_copy_with_mask(T, idx_inner, idx_outer):
    """
    T[idx_inner] = T[idx_outer]
    """
    if idx_inner.size == 0:
        return T
    T[idx_inner] = T[idx_outer]
    return T