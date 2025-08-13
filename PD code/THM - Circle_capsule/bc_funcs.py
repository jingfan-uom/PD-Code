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
    tol: float = 1e-12
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

    for i, g in enumerate(ghost_coords):
        r_fine, z_fine = g

        # Calculate the distance from each coarse point to the ghost point
        dists = np.sqrt((phys_coords[:, 0] - r_fine)**2 + (phys_coords[:, 1] - z_fine)**2)
        nearby_mask = dists <= radius
        nearby_indices = np.where(nearby_mask)[0]

        ghost_indices.append(i)
        phys_indices_list.append(nearby_indices)

    return ghost_indices, phys_indices_list


def interpolate_temperature_for_coarse_and_fine(
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

def interpolate_temperature_for_same(
    T_ghost: np.ndarray,
    T_phys: np.ndarray,
    ghost_indices: np.ndarray,
    phys_indices: np.ndarray
) -> np.ndarray:

    T_result = T_ghost.copy()
    for i, g_idx in enumerate(ghost_indices):
        p_idx = phys_indices[i]
        T_result[g_idx] = T_phys[p_idx]

    return T_result
