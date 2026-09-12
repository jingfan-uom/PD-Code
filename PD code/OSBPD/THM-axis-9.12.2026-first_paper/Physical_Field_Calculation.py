import numpy as np
import geometry_utils as geom
from scipy.spatial import cKDTree
from numba import njit, prange


@njit(fastmath=True)
def compute_dilation_axisym_csr(
    r_flat, z_flat, Ur, Uz,
    edge_i, indices,
    dist_edge,
    area_edge,
    shape_edge,
    coeff, csr,         # 鏍囬噺
    eps=1e-30
):
    N = r_flat.shape[0]
    nnz = indices.shape[0]

    dilation = np.zeros(N, dtype=np.float64)
    eij_edge = np.empty(nnz, dtype=np.float64)
    n_r_edge = np.empty(nnz, dtype=np.float64)
    n_z_edge = np.empty(nnz, dtype=np.float64)

    for k in range(nnz):

        i = edge_i[k]
        j = indices[k]
        if j == i:
            eij_edge[k] = 0.0
            n_r_edge[k] = 0.0
            n_z_edge[k] = 0.0
            continue

        dx_r = r_flat[j] - r_flat[i]
        dx_z = z_flat[j] - z_flat[i]
        dist = dist_edge[k] + eps  # L0

        du_r = Ur[j] - Ur[i]
        du_z = Uz[j] - Uz[i]

        # exact bond strain: eij = (L1 - L0) / L0
        dx1_r = dx_r + du_r
        dx1_z = dx_z + du_z
        L1 = np.sqrt(dx1_r * dx1_r + dx1_z * dx1_z)

        eij = (L1 - dist)
        eij_edge[k] = eij / dist

        inv = 1.0 / dist
        n_r_edge[k] = dx1_r * inv
        n_z_edge[k] = dx1_z * inv

        dilation[i] += eij * area_edge[k] * coeff * shape_edge[k]

    return dilation, eij_edge, n_r_edge, n_z_edge


@njit(parallel=True, fastmath=True)
def compute_dilation_axisym_csr_rows_numba(
    indptr, indices,
    r_flat, z_flat, Ur, Uz,
    dist_edge,
    area_edge,
    shape_edge,
    coeff, csr,
    eps=1e-30,
):
    N = indptr.shape[0] - 1
    nnz = indices.shape[0]

    dilation = np.zeros(N, dtype=np.float64)
    eij_edge = np.empty(nnz, dtype=np.float64)
    n_r_edge = np.empty(nnz, dtype=np.float64)
    n_z_edge = np.empty(nnz, dtype=np.float64)

    for i in prange(N):
        dil_i = 0.0
        ri = r_flat[i]
        zi = z_flat[i]
        uri = Ur[i]
        uzi = Uz[i]

        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j == i:
                eij_edge[p] = 0.0
                n_r_edge[p] = 0.0
                n_z_edge[p] = 0.0
                continue

            dx_r = r_flat[j] - ri
            dx_z = z_flat[j] - zi
            dist = dist_edge[p] + eps

            dx1_r = dx_r + Ur[j] - uri
            dx1_z = dx_z + Uz[j] - uzi
            L1 = np.sqrt(dx1_r * dx1_r + dx1_z * dx1_z)

            eij = L1 - dist
            eij_edge[p] = eij / dist

            inv = 1.0 / dist
            n_r_edge[p] = dx1_r * inv
            n_z_edge[p] = dx1_z * inv

            dil_i += eij * area_edge[p] * coeff * shape_edge[p]

        dilation[i] = dil_i

    return dilation, eij_edge, n_r_edge, n_z_edge


@njit(parallel=True, fastmath=True)
def compute_dilation_axisym_csr_rows_crack_numba(
    indptr, indices,
    r_flat, z_flat, Ur, Uz,
    dist_edge,
    area_edge,
    shape_edge,
    coeff, csr,
    crack,
    eps=1e-30,
):
    N = indptr.shape[0] - 1
    nnz = indices.shape[0]

    dilation = np.zeros(N, dtype=np.float64)
    eij_edge = np.empty(nnz, dtype=np.float64)
    n_r_edge = np.empty(nnz, dtype=np.float64)
    n_z_edge = np.empty(nnz, dtype=np.float64)

    for i in prange(N):
        dil_i = 0.0
        ri = r_flat[i]
        zi = z_flat[i]
        uri = Ur[i]
        uzi = Uz[i]

        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j == i:
                eij_edge[p] = 0.0
                n_r_edge[p] = 0.0
                n_z_edge[p] = 0.0
                continue

            dx_r = r_flat[j] - ri
            dx_z = z_flat[j] - zi
            dist = dist_edge[p] + eps

            dx1_r = dx_r + Ur[j] - uri
            dx1_z = dx_z + Uz[j] - uzi
            L1 = np.sqrt(dx1_r * dx1_r + dx1_z * dx1_z)

            eij = L1 - dist
            eij_edge[p] = eij / dist

            inv = 1.0 / dist
            n_r_edge[p] = dx1_r * inv
            n_z_edge[p] = dx1_z * inv

            if crack[p] != 0:
                dil_i += eij * area_edge[p] * coeff * shape_edge[p]

        dilation[i] = dil_i

    return dilation, eij_edge, n_r_edge, n_z_edge


@njit(fastmath=True)
def compute_dilation_axisym_csr_weighted(
    r_flat, z_flat, Ur, Uz,
    edge_i, indices,
    dist_edge,
    area_edge,
    shape_edge,
    coeff,
    csr,
    use_csr_in_dilation,
    eps=1e-30
):
    N = r_flat.shape[0]
    nnz = indices.shape[0]

    dilation = np.zeros(N, dtype=np.float64)
    eij_edge = np.empty(nnz, dtype=np.float64)
    n_r_edge = np.empty(nnz, dtype=np.float64)
    n_z_edge = np.empty(nnz, dtype=np.float64)

    for k in range(nnz):
        i = edge_i[k]
        j = indices[k]
        if j == i:
            eij_edge[k] = 0.0
            n_r_edge[k] = 0.0
            n_z_edge[k] = 0.0
            continue

        dx_r = r_flat[j] - r_flat[i]
        dx_z = z_flat[j] - z_flat[i]
        dist = dist_edge[k] + eps

        du_r = Ur[j] - Ur[i]
        du_z = Uz[j] - Uz[i]

        dx1_r = dx_r + du_r
        dx1_z = dx_z + du_z
        L1 = np.sqrt(dx1_r * dx1_r + dx1_z * dx1_z)

        eij = (L1 - dist)
        eij_edge[k] = eij / dist

        inv = 1.0 / dist
        n_r_edge[k] = dx1_r * inv
        n_z_edge[k] = dx1_z * inv

        w = area_edge[k] * coeff * shape_edge[k]
        if use_csr_in_dilation:
            w *= csr[k]
        dilation[i] += eij * w

    return dilation, eij_edge, n_r_edge, n_z_edge


@njit(parallel=True, fastmath=True)
def compute_direction_edges_csr_numba(coords, edge_i, edge_j, dist, eps=1e-30):
    Nnz = dist.size
    dir_r = np.zeros(Nnz, dtype=np.float64)
    dir_z = np.zeros(Nnz, dtype=np.float64)

    x = coords[:, 0]
    z = coords[:, 1]

    for p in prange(Nnz):
        i = edge_i[p]
        j = edge_j[p]
        dij = dist[p]
        if dij > eps and i != j:
            dir_r[p] = (x[j] - x[i]) / dij
            dir_z[p] = (z[j] - z[i]) / dij
        else:
            dir_r[p] = 0.0
            dir_z[p] = 0.0

    return dir_r, dir_z


def find_inner_surface_band(
        coords_phys_m,
        outer_radius,
        inner_radius,
        dr,
        center_r=0.0,
        width_factor=1.0,
        shell_mask=None,
        modify_coordinates=False,
        coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):
    coords = np.asarray(coords_phys_m)
    if coords.size == 0:
        return {
            "indices": np.empty(0, dtype=np.int64),
            "unit_outward": np.zeros((0, 2), dtype=np.float64),
            "unit_to_center": np.zeros((0, 2), dtype=np.float64),
            "void_face_count": np.zeros(0, dtype=np.int64),
        }

    dx = coords[:, 0] - center_r
    dz = coords[:, 1] - outer_radius
    if modify_coordinates:
        dshell_local = outer_radius - inner_radius
        inner_axis_x, inner_axis_z = geom.capsule_axes(
            outer_radius,
            dshell=dshell_local,
            modify_coordinates=True,
            coordinate_x_scale=coordinate_x_scale,
            inner=True,
        )
        level = geom.ellipse_level(
            coords[:, 0], coords[:, 1],
            center_r, outer_radius,
            inner_axis_x, inner_axis_z,
        )
        band = width_factor * dr / min(inner_axis_x, inner_axis_z)
        select = (level > 1.0) & (level < (1.0 + band) ** 2)
    else:
        rho = np.sqrt(dx ** 2 + dz ** 2)
        select = (rho > inner_radius) & (rho < inner_radius + width_factor * dr)

    if shell_mask is not None:
        shell_mask = np.asarray(shell_mask, dtype=bool)
        if shell_mask.shape[0] != coords.shape[0]:
            raise ValueError("shell_mask must have the same length as coords_phys_m")
        select &= shell_mask

    idx_layer = np.where(select)[0].astype(np.int64)
    if idx_layer.size == 0:
        return {
            "indices": np.empty(0, dtype=np.int64),
            "unit_outward": np.zeros((0, 2), dtype=np.float64),
            "unit_to_center": np.zeros((0, 2), dtype=np.float64),
            "void_face_count": np.zeros(0, dtype=np.int64),
        }

    if modify_coordinates:
        unit_outward = geom.capsule_outward_normals(
            coords[idx_layer, 0], coords[idx_layer, 1],
            center_r, outer_radius,
            inner_axis_x, inner_axis_z,
        )
        rho = np.sqrt(dx ** 2 + dz ** 2)
    else:
        rho_layer = np.maximum(rho[idx_layer], 1.0e-30)
        unit_outward = np.zeros((idx_layer.size, 2), dtype=np.float64)
        unit_outward[:, 0] = dx[idx_layer] / rho_layer
        unit_outward[:, 1] = dz[idx_layer] / rho_layer

    return {
        "indices": idx_layer,
        "unit_outward": unit_outward,
        "unit_to_center": -unit_outward,
        "void_face_count": np.ones(idx_layer.size, dtype=np.int64),
        "rho": rho[idx_layer],
    }


def compute_Tensor_product(x, y, horizon_mask):

    # Current relative positions after deformation: x '- x
    dx_eff = x[None, :] - x[:, None]
    dz_eff = y[None, :] - y[:, None]
    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)
    # Only compute values where horizon_mask is True; set others to zero
    n_x = np.zeros_like(dx_eff)
    n_z = np.zeros_like(dz_eff)

    n_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    n_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    N = n_x.shape[0]
    Cxx = np.zeros((N, N))
    Cxz = np.zeros((N, N))
    Czx = np.zeros((N, N))
    Czz = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # 鍙鏈夋晥锛坔orizon_mask涓篢rue锛夌殑閿绠?            if horizon_mask[i, j]:
                n_vec = np.array([n_x[i, j], n_z[i, j]])  # 缁勬垚浜岀淮鍗曚綅鍚戦噺
                n_outer = np.outer(n_vec, n_vec)  # 2x2寮犻噺绉?
                Cxx[i, j] = n_outer[0, 0]
                Cxz[i, j] = n_outer[0, 1]
                Czx[i, j] = n_outer[1, 0]
                Czz[i, j] = n_outer[1, 1]

    return Cxx, Cxz, Czx, Czz

def compute_s_matrix(coords, Ux, Uz, horizon_mask):

    """Compute elongation matrix s_matrix (N, N) using vectorized matrix operations.

    Parameters:
        coords: (N, 2) array of original coordinates (x, y)
        Ux, Uz: displacement arrays (N,)
        horizon_mask: boolean array of shape (N, N), True if bond (i, j) is valid

    Returns:
        s_matrix: elongation matrix (N, N)"""

    # Original coordinates
    x_flat = coords[:, 0]
    y_flat = coords[:, 1]

    # Deformed coordinates
    x_def = x_flat + Ux
    y_def = y_flat + Uz

    # Initial lengths L0
    dx0 = x_flat[None, :] - x_flat[:, None]
    dz0 = y_flat[None, :] - y_flat[:, None]
    L0 = np.sqrt(dx0 ** 2 + dz0 ** 2)

    # Deformed lengths L1
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Elongation computation (vectorized)
    s_matrix = np.zeros_like(L0)
    mask = horizon_mask & (L0 > 0)
    s_matrix[mask] = (L1[mask] - L0[mask]) / L0[mask]

    return s_matrix

def compute_delta_temperature(T_grid, Tpre_avg):

    """Compute the average temperature matrix and the difference from the previous step.

    Parameters:
        T_grid: current temperature field (1D or flattened)
        Tpre_avg: average temperature matrix from previous time step, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """
    T_i = T_grid[:, np.newaxis]  # shape (N, 1)
    T_j = T_grid[np.newaxis, :]  # shape (1, N)
    Tcurr_avg = 0.5 * (T_i + T_j)- Tpre_avg  # shape (N, N)

    return Tcurr_avg


def shrink_Tth_by_matching_coords(coords_m, coords_t):

    coords_m = np.asarray(coords_m)
    coords_t = np.asarray(coords_t)

    # If there are three columns, take the first two columns to match
    if coords_m.shape[1] >= 3:
        coords_m = coords_m[:, :2]
    if coords_t.shape[1] >= 3:
        coords_t = coords_t[:, :2]

    # Nearest Neighbor Matching
    tree = cKDTree(coords_t)
    _, indices = tree.query(coords_m, k=1)

    return indices.astype(np.int64)


def shrink_shell_only_by_matching_coords(coords_shell_only, coords_shell_core):
    return shrink_Tth_by_matching_coords(coords_shell_only, coords_shell_core)



def filter_array_by_indices_keep_only(Tarr, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return np.ascontiguousarray(Tarr[indices])



def update_mu_by_failure(mu, Relative_elongation, s0):

    failure_mask = (mu == 1) & (Relative_elongation >= s0)
    mu_new = mu.copy()
    mu_new[failure_mask] = 0

    return mu_new


def find_inner_surface_layer(
    coords_phys_m,
    r,
    dshell,
    dr,
    center_r=0.0,
    modify_coordinates=False,
    coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):


    coords = np.asarray(coords_phys_m)
    if coords.size == 0:
        return {
            "indices": np.empty(0, dtype=np.int64),
            "unit_outward": np.zeros((0, 2), dtype=np.float64),
            "unit_to_center": np.zeros((0, 2), dtype=np.float64),
            "void_face_count": np.zeros(0, dtype=np.int64),
        }

    x_arr = coords[:, 0]
    z_arr = coords[:, 1]

    xc, zc = float(center_r), float(r)

    dx = x_arr - xc
    dz = z_arr - zc

    r_inner = float(r) - float(dshell)
    inner_axis_x, inner_axis_z = geom.capsule_axes(
        r,
        dshell=dshell,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
        inner=True,
    )

    # ---------- numerical tolerance ----------
    scale = max(
        abs(float(r)),
        abs(float(r_inner)),
        abs(float(dr)),
        1.0,
    )

    tol = max(
        2.0e-14,
        64.0 * np.finfo(float).eps * scale,
    )

    # ---------- select particles whose grid faces touch the inner cavity ----------
    dr_abs = abs(float(dr))
    key_r = np.rint(x_arr / dr_abs).astype(np.int64)
    key_z = np.rint(z_arr / dr_abs).astype(np.int64)
    shell_keys = set(zip(key_r.tolist(), key_z.tolist()))

    void_face_count = np.zeros(x_arr.shape[0], dtype=np.int64)
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for idx in range(x_arr.shape[0]):
        count = 0
        for off_r, off_z in offsets:
            neighbor_key = (key_r[idx] + off_r, key_z[idx] + off_z)
            if neighbor_key in shell_keys:
                continue

            x_neighbor = x_arr[idx] + off_r * dr_abs
            z_neighbor = z_arr[idx] + off_z * dr_abs
            inner_level = geom.ellipse_level(
                x_neighbor, z_neighbor,
                xc, zc,
                inner_axis_x, inner_axis_z,
            )
            if inner_level < 1.0 - tol / max(inner_axis_x, inner_axis_z):
                count += 1
        void_face_count[idx] = count

    # Select every particle whose grid face touches the inner cavity.
    # Do not add an extra radial band filter here; the free-face test already
    # identifies the inner boundary layer on the stair-stepped grid.
    idx_layer = np.where(void_face_count >= 1)[0]

    if idx_layer.size == 0 or r_inner <= 0.0 or dr_abs <= 0.0:
        return {
            "indices": np.empty(0, dtype=np.int64),
            "unit_outward": np.zeros((0, 2), dtype=np.float64),
            "unit_to_center": np.zeros((0, 2), dtype=np.float64),
            "void_face_count": np.zeros(0, dtype=np.int64),
        }

    if modify_coordinates:
        unit_outward = geom.capsule_outward_normals(
            x_arr[idx_layer], z_arr[idx_layer],
            xc, zc,
            inner_axis_x, inner_axis_z,
        )
    else:
        dist = np.sqrt(dx**2 + dz**2)
        rho_layer = np.maximum(dist[idx_layer], 1.0e-30)
        unit_outward = np.zeros((idx_layer.size, 2), dtype=np.float64)
        unit_outward[:, 0] = dx[idx_layer] / rho_layer
        unit_outward[:, 1] = dz[idx_layer] / rho_layer

    return {
        "indices": idx_layer,
        "unit_outward": unit_outward,
        "unit_to_center": -unit_outward,
        "void_face_count": void_face_count[idx_layer],
    }


def compute_inner_surface_arc_length_correction(
    coords_phys_m,
    surface_indices,
    inner_radius,
    dr,
    center_r=0.0,
    center_z=0.0,
    unit_outward=None,
    theta_min=-0.5 * np.pi,
    theta_max=0.5 * np.pi,
    method="angular",
    outer_radius=None,
    modify_coordinates=False,
    coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):
    """
    Compute the inner-boundary arc length represented by each selected boundary cell.

    Returns component factors intended for pressure body-force loading:
        factor_r = arc_length * n_r / dr
        factor_z = arc_length * n_z / dr

    A scalar body load p/dr can be multiplied by factor_r/factor_z directly.
    method="uniform_angular" keeps the total angular arc length but gives each
    selected surface particle the same length factor.
    """
    coords = np.asarray(coords_phys_m, dtype=np.float64)
    idx = np.asarray(surface_indices, dtype=np.int64)
    dr_abs = abs(float(dr))
    radius = float(inner_radius)

    if idx.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "indices": idx,
            "arc_length": empty,
            "length_factor": empty,
            "factor_r": empty,
            "factor_z": empty,
            "unit_outward": np.zeros((0, 2), dtype=np.float64),
        }

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords_phys_m must be an (N, 2) coordinate array")
    if radius <= 0.0 or dr_abs <= 0.0:
        raise ValueError("inner_radius and dr must be positive")
    if np.any(idx < 0) or np.any(idx >= coords.shape[0]):
        raise IndexError("surface_indices contains out-of-range indices")
    outer_radius_for_axes = radius if outer_radius is None else float(outer_radius)
    dshell_for_axes = max(outer_radius_for_axes - radius, 0.0)
    axis_x, axis_z = geom.capsule_axes(
        outer_radius_for_axes,
        dshell=dshell_for_axes,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
        inner=True,
    )

    pts = coords[idx, :2]
    dx = pts[:, 0] - float(center_r)
    dz = pts[:, 1] - float(center_z)
    rho = np.maximum(np.sqrt(dx ** 2 + dz ** 2), 1.0e-30)

    if unit_outward is None:
        if modify_coordinates:
            unit = geom.capsule_outward_normals(
                pts[:, 0], pts[:, 1],
                float(center_r), float(center_z),
                axis_x, axis_z,
            )
        else:
            unit = np.column_stack([dx / rho, dz / rho])
    else:
        unit = np.asarray(unit_outward, dtype=np.float64)
        if unit.shape != (idx.size, 2):
            raise ValueError("unit_outward must have shape (len(surface_indices), 2)")

    domain_lo = float(theta_min)
    domain_hi = float(theta_max)
    if domain_hi < domain_lo:
        domain_lo, domain_hi = domain_hi, domain_lo

    if method in ("angular", "uniform_angular"):
        theta = np.arctan2(dz / axis_z, dx / axis_x)
        theta = np.clip(theta, domain_lo, domain_hi)
        order = np.argsort(theta)
        theta_sorted = theta[order]

        bounds = np.empty(idx.size + 1, dtype=np.float64)
        bounds[0] = domain_lo
        bounds[-1] = domain_hi
        if idx.size > 1:
            bounds[1:-1] = 0.5 * (theta_sorted[:-1] + theta_sorted[1:])

        theta_width_sorted = np.maximum(bounds[1:] - bounds[:-1], 0.0)
        arc_length = np.zeros(idx.size, dtype=np.float64)
        theta_measure = np.zeros(idx.size, dtype=np.float64)
        theta_mid_sorted = bounds[:-1] + 0.5 * theta_width_sorted
        arc_speed_sorted = np.sqrt(
            (axis_x * np.sin(theta_mid_sorted)) ** 2
            + (axis_z * np.cos(theta_mid_sorted)) ** 2
        )
        arc_length[order] = arc_speed_sorted * theta_width_sorted
        theta_measure[order] = theta_width_sorted

        if method == "uniform_angular" and idx.size > 0:
            arc_length.fill(float(np.sum(arc_length)) / float(idx.size))
            theta_measure.fill(float(np.sum(theta_measure)) / float(idx.size))

        length_factor = arc_length / dr_abs
        return {
            "indices": idx,
            "arc_length": arc_length,
            "theta_measure": theta_measure,
            "length_factor": length_factor,
            "factor_r": length_factor * unit[:, 0],
            "factor_z": length_factor * unit[:, 1],
            "unit_outward": unit,
        }

    if method != "cell_intersection":
        raise ValueError("method must be 'angular', 'uniform_angular', or 'cell_intersection'")
    if modify_coordinates:
        raise NotImplementedError("cell_intersection arc correction is only implemented for circular geometry")

    tol = max(1.0e-14, 64.0 * np.finfo(float).eps * max(radius, dr_abs, 1.0))

    def _clip_unit(value):
        return min(1.0, max(-1.0, value))

    arc_length = np.zeros(idx.size, dtype=np.float64)
    theta_measure = np.zeros(idx.size, dtype=np.float64)

    for k, (x0, z0) in enumerate(pts):
        x_min = x0 - 0.5 * dr_abs
        x_max = x0 + 0.5 * dr_abs
        z_min = z0 - 0.5 * dr_abs
        z_max = z0 + 0.5 * dr_abs

        candidates = [domain_lo, domain_hi]

        s_min = (z_min - float(center_z)) / radius
        s_max = (z_max - float(center_z)) / radius
        if s_min <= 1.0 + tol and s_max >= -1.0 - tol:
            for s_val in (_clip_unit(s_min), _clip_unit(s_max)):
                theta = np.arcsin(s_val)
                if domain_lo - tol <= theta <= domain_hi + tol:
                    candidates.append(float(np.clip(theta, domain_lo, domain_hi)))

        c_min = (x_min - float(center_r)) / radius
        c_max = (x_max - float(center_r)) / radius
        if c_min <= 1.0 + tol and c_max >= -1.0 - tol:
            for c_val in (_clip_unit(c_min), _clip_unit(c_max)):
                theta = np.arccos(c_val)
                for signed_theta in (-theta, theta):
                    if domain_lo - tol <= signed_theta <= domain_hi + tol:
                        candidates.append(float(np.clip(signed_theta, domain_lo, domain_hi)))

        candidates = np.array(sorted(set(np.round(candidates, 15))), dtype=np.float64)
        if candidates.size < 2:
            continue

        total_theta = 0.0
        for a, b in zip(candidates[:-1], candidates[1:]):
            if b <= a + tol:
                continue
            mid = 0.5 * (a + b)
            x_mid = float(center_r) + radius * np.cos(mid)
            z_mid = float(center_z) + radius * np.sin(mid)
            if (
                x_min - tol <= x_mid <= x_max + tol
                and z_min - tol <= z_mid <= z_max + tol
            ):
                total_theta += b - a

        theta_measure[k] = total_theta
        arc_length[k] = radius * total_theta

    length_factor = arc_length / dr_abs
    return {
        "indices": idx,
        "arc_length": arc_length,
        "theta_measure": theta_measure,
        "length_factor": length_factor,
        "factor_r": length_factor * unit[:, 0],
        "factor_z": length_factor * unit[:, 1],
        "unit_outward": unit,
    }

def compute_density_increment_pressure_temperature(
    rho0,
    compressibility,
    pressure,
    beta=0.0,
    deltaT=0.0,
    eps=1e-30,
):
    """Density increment used by the pressure-coupled phase-change formula."""
    rho0 = float(rho0)
    K = float(compressibility)
    pressure = float(pressure)
    beta = float(beta)
    deltaT = float(deltaT)

    if rho0 <= eps or K <= 0.0 or pressure <= 0.0:
        return 0.0

    thermal = 1.0 + beta * deltaT
    pressure_factor = 1.0 - K * pressure
    denom = thermal * pressure_factor
    if denom <= eps:
        denom = eps
    return float(rho0 * K * pressure / denom)


def compute_phase_volume_change_ratio_from_density(
    rho_s0,
    rho_l0,
    f_melt,
    pressure=0.0,
    compressibility_s=0.0,
    compressibility_l=0.0,
    beta_s=0.0,
    beta_l=0.0,
    deltaT=0.0,
    eps=1e-30,
):
    """
    Relative PCM volume change from phase fraction and pressure-dependent density.

    At zero compressibility this reduces to:
        DeltaV/V0 = f_l * (rho_s0 / rho_l0 - 1)
    """
    rho_s0 = float(rho_s0)
    rho_l0 = float(rho_l0)
    f_l = float(np.clip(f_melt, 0.0, 1.0))

    if rho_s0 <= eps or rho_l0 <= eps:
        raise ValueError("rho_s0 and rho_l0 must be positive")

    delta_rho_s = compute_density_increment_pressure_temperature(
        rho_s0, compressibility_s, pressure, beta_s, deltaT, eps=eps
    )
    delta_rho_l = compute_density_increment_pressure_temperature(
        rho_l0, compressibility_l, pressure, beta_l, deltaT, eps=eps
    )
    rho_s_eff = max(rho_s0 + delta_rho_s, eps)
    rho_l_eff = max(rho_l0 + delta_rho_l, eps)

    solid_density_volume_ratio = rho_s0 / rho_s_eff - 1.0
    liquid_density_volume_ratio = rho_s0 / rho_l_eff - 1.0
    solid_relative = (1.0 - f_l) * solid_density_volume_ratio
    liquid_relative = f_l * liquid_density_volume_ratio
    relative_deltaV = solid_relative + liquid_relative

    return {
        "relative_deltaV": float(relative_deltaV),
        "solid_relative_deltaV": float(solid_relative),
        "liquid_relative_deltaV": float(liquid_relative),
        "solid_density_volume_ratio": float(solid_density_volume_ratio),
        "liquid_density_volume_ratio": float(liquid_density_volume_ratio),
        "f_melt": float(f_l),
        "delta_rho_s": float(delta_rho_s),
        "delta_rho_l": float(delta_rho_l),
        "rho_s_eff": float(rho_s_eff),
        "rho_l_eff": float(rho_l_eff),
        "pressure": float(pressure),
        "deltaT": float(deltaT),
    }


def compute_melt_and_thermal_expansion(
    T,
    mask_core,
    rho_s, rho_l,
    Ts, Tl,
    cell_volume,     # 鏍囬噺 鎴?(N,) 鎴?(N_core,)
    beta_s,
    beta_l,
    T_ref,
    core_radius=None,
    return_details=False,
    pressure=0.0,
    compressibility_s=0.0,
    compressibility_l=0.0,
    density_beta_s=0.0,
    density_beta_l=0.0,
    density_deltaT=0.0,
):
    T = np.asarray(T, dtype=float)
    core = np.asarray(mask_core, dtype=bool)

    T_core = T[core]

    cv = np.asarray(cell_volume, dtype=float)
    if cv.ndim == 0:
        cv_core = np.full(T_core.shape, float(cv), dtype=float)
    elif cv.shape[0] == T.shape[0]:
        cv_core = cv[core]
    elif cv.shape[0] == T_core.shape[0]:
        cv_core = cv
    else:
        raise ValueError("cell_volume must be scalar, T-sized, or core-sized")

    dT_phase = Tl - Ts
    if dT_phase <= 0.0:
        raise ValueError("Tl must be greater than Ts")

    alpha = (T_core - Ts) / dT_phase
    alpha = np.clip(alpha, 0.0, 1.0)


    V_melt_equiv = float(np.sum(alpha * cv_core))
    core_volume = float(np.sum(cv_core))
    if core_radius is not None:
        core_volume = float((4.0 / 3.0) * np.pi * float(core_radius) ** 3)
    melt_fraction = 0.0 if core_volume <= 0.0 else V_melt_equiv / core_volume
    melt_fraction = float(np.clip(melt_fraction, 0.0, 1.0))

    phase_density = compute_phase_volume_change_ratio_from_density(
        rho_s,
        rho_l,
        melt_fraction,
        pressure=pressure,
        compressibility_s=compressibility_s,
        compressibility_l=compressibility_l,
        beta_s=density_beta_s,
        beta_l=density_beta_l,
        deltaT=density_deltaT,
    )
    if compressibility_s > 0.0 or compressibility_l > 0.0 or pressure > 0.0:
        solid_volume_ref = float(np.sum((1.0 - alpha) * cv_core))
        liquid_volume_ref = V_melt_equiv
        deltaV_phase = (
            solid_volume_ref * phase_density["solid_density_volume_ratio"]
            + liquid_volume_ref * phase_density["liquid_density_volume_ratio"]
        )
    else:
        kappa = (rho_s / rho_l) - 1.0
        deltaV_phase = V_melt_equiv * kappa

    dT_vec = T_core - T_ref
    beta_eff = beta_s * (1.0 - alpha) + beta_l * alpha

    deltaV_thermal = float(np.sum(beta_eff * dT_vec * cv_core))

    if return_details:
        return {
            "deltaV_phase": float(deltaV_phase),
            "deltaV_thermal": float(deltaV_thermal),
            "melt_fraction": float(melt_fraction),
            "melt_volume_equiv": float(V_melt_equiv),
            "core_volume": float(core_volume),
            "phase_density": phase_density,
        }

    return float(deltaV_phase), float(deltaV_thermal)


def compute_pcm_volume_change_consistent(
    T,
    mask_core,
    rho_s, rho_l,
    Ts, Tl,
    cell_volume,
    beta_s,
    beta_l,
    T_ref,
    pressure=0.0,
    compressibility_s=0.0,
    compressibility_l=0.0,
    core_volume_ref=None,
    return_details=False,
    eps=1e-30,
):
    """
    Grid-integrated PCM volume change relative to the initial solid cell volume.

    The liquid term uses mass conservation:
        V_l / V0 = (rho_s / rho_l) * (1 + beta_l*dT) * (1 - k_l*p)
    and the solid term uses:
        V_s / V0 = (1 + beta_s*dT) * (1 - k_s*p)
    where beta is the volumetric expansion coefficient used by this model.
    """
    T = np.asarray(T, dtype=float)
    core = np.asarray(mask_core, dtype=bool)
    T_core = T[core]

    cv = np.asarray(cell_volume, dtype=float)
    if cv.ndim == 0:
        cv_core = np.full(T_core.shape, float(cv), dtype=float)
    elif cv.shape[0] == T.shape[0]:
        cv_core = cv[core]
    elif cv.shape[0] == T_core.shape[0]:
        cv_core = cv
    else:
        raise ValueError("cell_volume must be scalar, T-sized, or core-sized")

    dT_phase = Tl - Ts
    if dT_phase <= 0.0:
        raise ValueError("Tl must be greater than Ts")
    if rho_s <= eps or rho_l <= eps:
        raise ValueError("rho_s and rho_l must be positive")

    alpha = np.clip((T_core - Ts) / dT_phase, 0.0, 1.0)
    dT_vec = T_core - T_ref
    pressure = max(0.0, float(pressure))
    ks = max(0.0, float(compressibility_s))
    kl = max(0.0, float(compressibility_l))

    solid_compression = np.maximum(eps, 1.0 - ks * pressure)
    liquid_compression = np.maximum(eps, 1.0 - kl * pressure)
    rho_ratio_l = float(rho_s) / float(rho_l)
    if pressure > 0.0 and ks > 0.0:
        denom_s = np.maximum(eps, (1.0 + beta_s * dT_vec) * solid_compression)
        delta_rho_s_vec = float(rho_s) * ks * pressure / denom_s
    else:
        delta_rho_s_vec = np.zeros_like(dT_vec)
    if pressure > 0.0 and kl > 0.0:
        denom_l = np.maximum(eps, (1.0 + beta_l * dT_vec) * liquid_compression)
        delta_rho_l_vec = float(rho_l) * kl * pressure / denom_l
    else:
        delta_rho_l_vec = np.zeros_like(dT_vec)

    solid_volume_ratio = (1.0 + beta_s * dT_vec) * solid_compression - 1.0
    liquid_volume_ratio = rho_ratio_l * (1.0 + beta_l * dT_vec) * liquid_compression - 1.0
    mixed_volume_ratio = (1.0 - alpha) * solid_volume_ratio + alpha * liquid_volume_ratio

    deltaV_total = float(np.sum(mixed_volume_ratio * cv_core))
    deltaV_phase = float(np.sum(alpha * (rho_ratio_l - 1.0) * cv_core))
    deltaV_thermal = float(np.sum(
        ((1.0 - alpha) * beta_s + alpha * rho_ratio_l * beta_l) * dT_vec * cv_core
    ))
    deltaV_compression = float(deltaV_total - deltaV_phase - deltaV_thermal)
    melt_volume_equiv = float(np.sum(alpha * cv_core))
    solid_density_weight = float(np.sum((1.0 - alpha) * cv_core))
    liquid_density_weight = float(np.sum(alpha * cv_core))
    if solid_density_weight > eps:
        delta_rho_s_avg = float(np.sum(delta_rho_s_vec * (1.0 - alpha) * cv_core) / solid_density_weight)
    else:
        delta_rho_s_avg = 0.0
    if liquid_density_weight > eps:
        delta_rho_l_avg = float(np.sum(delta_rho_l_vec * alpha * cv_core) / liquid_density_weight)
    else:
        delta_rho_l_avg = 0.0

    core_volume = float(np.sum(cv_core))
    if core_volume_ref is not None:
        core_volume = float(core_volume_ref)
    if core_volume <= eps:
        melt_fraction = 0.0
        relative_deltaV = 0.0
    else:
        melt_fraction = float(np.clip(melt_volume_equiv / core_volume, 0.0, 1.0))
        relative_deltaV = float(deltaV_total / core_volume)

    if return_details:
        return {
            "deltaV_total": deltaV_total,
            "deltaV_phase": deltaV_phase,
            "deltaV_thermal": deltaV_thermal,
            "deltaV_compression": deltaV_compression,
            "relative_deltaV": relative_deltaV,
            "melt_fraction": melt_fraction,
            "melt_volume_equiv": melt_volume_equiv,
            "core_volume": core_volume,
            "pressure": float(pressure),
            "delta_rho_s_avg": float(delta_rho_s_avg),
            "delta_rho_l_avg": float(delta_rho_l_avg),
            "rho_s_eff_avg": float(rho_s + delta_rho_s_avg),
            "rho_l_eff_avg": float(rho_l + delta_rho_l_avg),
            "solid_density_weight": float(solid_density_weight),
            "liquid_density_weight": float(liquid_density_weight),
        }

    return deltaV_total


def compute_inner_pressure_from_melt_formula(
    E,
    nu,
    R,
    r_m0,
    rho_s,
    rho_l,
    f_melt,
    alpha,
    deltaT,
    compressibility_s=0.0,
    compressibility_l=0.0,
    beta_s=0.0,
    beta_l=0.0,
    deltaT_pcm=None,
    solve_pressure_density=True,
    max_iter=80,
    return_details=False,
    eps=1e-30,
):
    if R <= eps:
        raise ValueError("R must be positive")
    if r_m0 <= eps or r_m0 >= R:
        raise ValueError("r_m0 must satisfy 0 < r_m0 < R")

    f_l = float(np.clip(f_melt, 0.0, 1.0))
    shell_growth = 1.0 + float(alpha) * float(deltaT)
    denom = R ** 3 * (1.0 + nu) + r_m0 ** 3 * (2.0 - 4.0 * nu)
    if abs(denom) <= eps:
        raise ZeroDivisionError("pressure formula denominator is too small")
    coeff = 2.0 * (R ** 3 - r_m0 ** 3) * E / denom
    if deltaT_pcm is None:
        deltaT_pcm = deltaT

    def evaluate_at_pressure(pressure_value):
        density_info = compute_phase_volume_change_ratio_from_density(
            rho_s,
            rho_l,
            f_l,
            pressure=pressure_value,
            compressibility_s=compressibility_s,
            compressibility_l=compressibility_l,
            beta_s=beta_s,
            beta_l=beta_l,
            deltaT=deltaT_pcm,
            eps=eps,
        )
        relative_deltaV = density_info["relative_deltaV"]
        pcm_growth = np.cbrt(max(0.0, 1.0 + relative_deltaV))
        pressure_formula = float(coeff * (pcm_growth - shell_growth))
        density_info.update({
            "pressure_formula": pressure_formula,
            "pcm_growth": float(pcm_growth),
            "shell_growth": float(shell_growth),
            "density_pressure_solved": False,
        })
        return pressure_formula, density_info

    density_coupled = (
        solve_pressure_density
        and (float(compressibility_s) > 0.0 or float(compressibility_l) > 0.0)
    )
    if not density_coupled:
        pressure_formula, density_info = evaluate_at_pressure(0.0)
        if return_details:
            density_info["density_pressure_solved"] = False
            density_info["pressure"] = float(pressure_formula)
            return density_info
        return float(pressure_formula)

    pressure_zero, density_zero = evaluate_at_pressure(0.0)
    if pressure_zero <= 0.0:
        if return_details:
            density_zero["density_pressure_solved"] = True
            density_zero["pressure"] = float(pressure_zero)
            density_zero["iterations"] = 0
            return density_zero
        return float(pressure_zero)

    max_K = max(float(compressibility_s), float(compressibility_l), 0.0)
    pressure_limit = np.inf if max_K <= 0.0 else 0.95 / max_K
    lo = 0.0
    hi = min(max(pressure_zero * 2.0, 1.0), pressure_limit)

    def residual(pressure_value):
        pressure_formula, _ = evaluate_at_pressure(pressure_value)
        return pressure_value - pressure_formula

    res_hi = residual(hi)
    expand_count = 0
    while res_hi < 0.0 and hi < pressure_limit and expand_count < 80:
        hi = min(hi * 2.0, pressure_limit)
        res_hi = residual(hi)
        expand_count += 1

    if res_hi < 0.0:
        pressure_value = hi
        pressure_formula, density_info = evaluate_at_pressure(pressure_value)
        density_info["density_pressure_solved"] = False
        density_info["pressure"] = float(pressure_value)
        density_info["pressure_formula"] = float(pressure_formula)
        density_info["iterations"] = int(expand_count)
        if return_details:
            return density_info
        return float(pressure_value)

    pressure_value = hi
    iterations = 0
    for iterations in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        res_mid = residual(mid)
        pressure_value = mid
        if abs(res_mid) <= max(1.0e-8 * max(1.0, mid), 1.0e-3):
            break
        if res_mid >= 0.0:
            hi = mid
        else:
            lo = mid

    pressure_formula, density_info = evaluate_at_pressure(pressure_value)
    density_info["density_pressure_solved"] = True
    density_info["pressure"] = float(pressure_value)
    density_info["pressure_formula"] = float(pressure_formula)
    density_info["iterations"] = int(iterations + 1)
    if return_details:
        return density_info
    return float(pressure_value)



def find_region_and_index(phys_coords_list_t, target=(0.0, 20e-6)):
    """
    鍦ㄦ瘡涓尯鍩熺殑鐗╃悊鐐瑰垪琛ㄤ腑鎵惧埌涓?target 鏈€杩戠殑鐐广€?    phys_coords_list_t[i] 褰㈠ [r, z, region_id]銆?    Returns: (region_id, index_in_region)
    """
    best_region, best_idx, best_d2 = None, None, np.inf
    tgt = np.asarray(target, dtype=float)

    for region_id, arr in enumerate(phys_coords_list_t):
        if arr.size == 0:
            continue
        coords = arr[:, :2]          # 鍙彇 r,z
        d2 = np.sum((coords - tgt)**2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] < best_d2:
            best_region, best_idx, best_d2 = region_id, idx, float(d2[idx])

    if best_region is None:
        raise ValueError("phys_coords_list_t is empty")

    return best_region, best_idx


def map_Tth_to_mech(coords_m, coords_th, T_th, tol):
    pts_m  = np.ascontiguousarray(coords_m[:, :2])
    pts_th = np.ascontiguousarray(coords_th[:, :2])

    tree = cKDTree(pts_th)
    dist, idx = tree.query(pts_m, k=1)

    # 鍏抽敭锛氭寜 mech 椤哄簭鐩存帴閲嶆帓
    return T_th[idx]


def temperature_sphere_dirichlet(rho, t, a, kappa, Tinit, Tsurr, n_terms=200):

    rho = np.asarray(rho, dtype=float)
    T = np.empty_like(rho)

    n = np.arange(1, n_terms + 1, dtype=float)
    decay = np.exp(-(n**2) * (np.pi**2) * kappa * t / (a**2))

    mask = rho > 1e-14

    if np.any(mask):
        rr = rho[mask][:, None]
        series = np.sum(
            ((-1.0)**(n + 1) / n) * np.sin(n * np.pi * rr / a) * decay,
            axis=1
        )
        ratio = (2.0 * a / (np.pi * rho[mask])) * series
        T[mask] = Tsurr + (Tinit - Tsurr) * ratio

    # rho = 0 澶勫彇鏋侀檺
    if np.any(~mask):
        series0 = np.sum(((-1.0)**(n + 1)) * decay)
        ratio0 = 2.0 * series0
        T[~mask] = Tsurr + (Tinit - Tsurr) * ratio0

    return T

def radial_disp_theory_sphere(rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
                              n_terms=200, n_r_int=2000):

    if rho_p < 1e-14:
        return 0.0

    rho_grid = np.linspace(0.0, a, n_r_int)
    T_grid = temperature_sphere_dirichlet(
        rho_grid, t, a, kappa, Tinit, Tsurr, n_terms=n_terms
    )
    dT_grid = T_grid - Tinit

    Ia = np.trapz(dT_grid * rho_grid ** 2, rho_grid)


    mask = rho_grid <= rho_p
    rho_sub = rho_grid[mask]
    dT_sub = dT_grid[mask]
    Ir = np.trapz(dT_sub * rho_sub ** 2, rho_sub)

    u = (alpha / (1.0 - nu)) * (
        (1.0 + nu) * Ir / (rho_p**2)
        + 2.0 * (1.0 - 2.0 * nu) * rho_p * Ia / (a**3)
    )
    return u

def horizontal_disp_theory_at_point(r_pt, z_pt, t, a, center_r, center_z,
                                    kappa, alpha, nu, Tinit, Tsurr,
                                    n_terms=200, n_r_int=2000):

    dr = r_pt - center_r
    dz = z_pt - center_z
    rho_p = np.sqrt(dr**2 + dz**2)

    if rho_p < 1e-14:
        return 0.0

    if rho_p > a + 1e-12:
        raise ValueError(f"Point outside sphere: rho_p={rho_p:.6e}, a={a:.6e}")

    u_radial = radial_disp_theory_sphere(
        rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
        n_terms=n_terms, n_r_int=n_r_int
    )

    # 鎶曞奖鍒版按骞?r 鏂瑰悜
    return u_radial * (dr / rho_p)


from numba import njit
import numpy as np
import geometry_utils as geom
@njit(fastmath=True)
def segment_lengths_in_core_shell_axisym(p1, p2, r_core, r_center, z_center, tol=1e-14):
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    L = np.sqrt(dx * dx + dz * dz)

    if L < tol:
        return 0.0, 0.0, 0.0

    x1 = p1[0] - r_center
    z1 = p1[1] - z_center
    x2 = p2[0] - r_center
    z2 = p2[1] - z_center

    in1 = (x1 * x1 + z1 * z1) <= r_core * r_core + tol
    in2 = (x2 * x2 + z2 * z2) <= r_core * r_core + tol

    if in1 and in2:
        return L, 0.0, L

    a = dx * dx + dz * dz
    b = 2.0 * (x1 * dx + z1 * dz)
    c = x1 * x1 + z1 * z1 - r_core * r_core

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t1 > t2:
        tmp = t1
        t1 = t2
        t2 = tmp

    has_t1 = (-tol <= t1 <= 1.0 + tol)
    has_t2 = (-tol <= t2 <= 1.0 + tol)

    if in1 != in2:
        t_hit = t1 if has_t1 else t2

        if t_hit < 0.0:
            t_hit = 0.0
        elif t_hit > 1.0:
            t_hit = 1.0

        if in1:
            L_core = t_hit * L
        else:
            L_core = (1.0 - t_hit) * L

        L_shell = L - L_core
        return L_core, L_shell, L

    if has_t1 and has_t2:
        t_in = t1
        t_out = t2

        if t_in < 0.0:
            t_in = 0.0
        if t_out > 1.0:
            t_out = 1.0

        if t_out > t_in + tol:
            L_core = (t_out - t_in) * L
            L_shell = L - L_core
            return L_core, L_shell, L

    return 0.0, L, L
@njit(parallel=True, fastmath=True)
def precompute_edge_core_shell_lengths(coords_all, edge_i, edge_j, r_core, r_center, z_center, tol=1e-14):
    nnz = edge_i.shape[0]

    L_core_edge = np.empty(nnz, dtype=np.float64)
    L_shell_edge = np.empty(nnz, dtype=np.float64)
    L_total_edge = np.empty(nnz, dtype=np.float64)

    for p in prange(nnz):
        i = edge_i[p]
        j = edge_j[p]

        p1 = coords_all[i]
        p2 = coords_all[j]

        Lc, Ls, Lt = segment_lengths_in_core_shell_axisym(
            p1, p2, r_core, r_center, z_center, tol
        )

        L_core_edge[p] = Lc
        L_shell_edge[p] = Ls
        L_total_edge[p] = Lt

    return L_core_edge, L_shell_edge, L_total_edge


@njit(fastmath=True)
def segment_lengths_in_core_shell_axisym_ellipse(
        p1,
        p2,
        core_axis_x,
        core_axis_z,
        r_center,
        z_center,
        tol=1e-14,
):
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    L = np.sqrt(dx * dx + dz * dz)

    if L < tol:
        return 0.0, 0.0, 0.0

    ax = max(core_axis_x, tol)
    az = max(core_axis_z, tol)
    x1 = p1[0] - r_center
    z1 = p1[1] - z_center
    x2 = p2[0] - r_center
    z2 = p2[1] - z_center

    level1 = (x1 / ax) * (x1 / ax) + (z1 / az) * (z1 / az)
    level2 = (x2 / ax) * (x2 / ax) + (z2 / az) * (z2 / az)
    in1 = level1 <= 1.0 + tol
    in2 = level2 <= 1.0 + tol

    if in1 and in2:
        return L, 0.0, L

    a = (dx / ax) * (dx / ax) + (dz / az) * (dz / az)
    b = 2.0 * (x1 * dx / (ax * ax) + z1 * dz / (az * az))
    c = level1 - 1.0

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 0.0, L, L

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t1 > t2:
        tmp = t1
        t1 = t2
        t2 = tmp

    has_t1 = (-tol <= t1 <= 1.0 + tol)
    has_t2 = (-tol <= t2 <= 1.0 + tol)

    if in1 != in2:
        t_hit = t1 if has_t1 else t2
        if t_hit < 0.0:
            t_hit = 0.0
        elif t_hit > 1.0:
            t_hit = 1.0

        if in1:
            L_core = t_hit * L
        else:
            L_core = (1.0 - t_hit) * L

        L_shell = L - L_core
        return L_core, L_shell, L

    if has_t1 and has_t2:
        t_in = t1
        t_out = t2

        if t_in < 0.0:
            t_in = 0.0
        if t_out > 1.0:
            t_out = 1.0

        if t_out > t_in + tol:
            L_core = (t_out - t_in) * L
            L_shell = L - L_core
            return L_core, L_shell, L

    return 0.0, L, L


@njit(parallel=True, fastmath=True)
def precompute_edge_core_shell_lengths_ellipse(
        coords_all,
        edge_i,
        edge_j,
        core_axis_x,
        core_axis_z,
        r_center,
        z_center,
        tol=1e-14,
):
    nnz = edge_i.shape[0]

    L_core_edge = np.empty(nnz, dtype=np.float64)
    L_shell_edge = np.empty(nnz, dtype=np.float64)
    L_total_edge = np.empty(nnz, dtype=np.float64)

    for p in prange(nnz):
        i = edge_i[p]
        j = edge_j[p]

        p1 = coords_all[i]
        p2 = coords_all[j]

        Lc, Ls, Lt = segment_lengths_in_core_shell_axisym_ellipse(
            p1,
            p2,
            core_axis_x,
            core_axis_z,
            r_center,
            z_center,
            tol,
        )

        L_core_edge[p] = Lc
        L_shell_edge[p] = Ls
        L_total_edge[p] = Lt

    return L_core_edge, L_shell_edge, L_total_edge


@njit(parallel=True, fastmath=True)
def build_edge_property_harmonic_from_lengths(
    prop_node, is_core_node, edge_i, edge_j,
    L_core_edge, L_shell_edge, L_total_edge,
    eps=1e-30
):
    nnz = edge_i.shape[0]
    out = np.empty(nnz, dtype=np.float64)

    core_sum = 0.0
    shell_sum = 0.0
    core_count = 0
    shell_count = 0
    for q in range(prop_node.shape[0]):
        if is_core_node[q]:
            core_sum += prop_node[q]
            core_count += 1
        else:
            shell_sum += prop_node[q]
            shell_count += 1

    fallback_prop = prop_node[0] if prop_node.shape[0] > 0 else 0.0
    if core_count > 0:
        p_core_ref = core_sum / core_count
    else:
        p_core_ref = fallback_prop
    if shell_count > 0:
        p_shell_ref = shell_sum / shell_count
    else:
        p_shell_ref = fallback_prop

    for p in prange(nnz):
        i = edge_i[p]
        j = edge_j[p]

        pi = prop_node[i]
        pj = prop_node[j]

        Lt = L_total_edge[p]
        Lc = L_core_edge[p]
        Ls = L_shell_edge[p]
        in_i = is_core_node[i]
        in_j = is_core_node[j]

        if Lt < eps:
            out[p] = pi
            continue

        mixed_segment = (Lc > eps) and (Ls > eps)

        if not mixed_segment and in_i == in_j:
            denom = 1.0 / (pi + eps) + 1.0 / (pj + eps)
            out[p] = 2.0 / (denom + eps)
            continue

        if in_i:
            p_core = pi
        elif in_j:
            p_core = pj
        else:
            p_core = p_core_ref

        if not in_i:
            p_shell = pi
        elif not in_j:
            p_shell = pj
        else:
            p_shell = p_shell_ref

        Wt = Lc + Ls
        denom = Lc / (p_core + eps) + Ls / (p_shell + eps)
        out[p] = Wt / (denom + eps)

    return out


def temperature_sphere_dirichlet(rho, t, a, kappa, Tinit, Tsurr, n_terms=200):

    rho = np.asarray(rho, dtype=float)
    T = np.empty_like(rho)

    n = np.arange(1, n_terms + 1, dtype=float)
    decay = np.exp(-(n**2) * (np.pi**2) * kappa * t / (a**2))

    mask = rho > 1e-14

    if np.any(mask):
        rr = rho[mask][:, None]
        series = np.sum(
            ((-1.0)**(n + 1) / n) * np.sin(n * np.pi * rr / a) * decay,
            axis=1
        )
        ratio = (2.0 * a / (np.pi * rho[mask])) * series
        T[mask] = Tsurr + (Tinit - Tsurr) * ratio

    # rho = 0 澶勫彇鏋侀檺
    if np.any(~mask):
        series0 = np.sum(((-1.0)**(n + 1)) * decay)
        ratio0 = 2.0 * series0
        T[~mask] = Tsurr + (Tinit - Tsurr) * ratio0

    return T

def radial_disp_theory_sphere(rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
                              n_terms=100, n_r_int=1000):
    """
    3D 瀹炲績鐞冨湪鑷敱杈圭晫鏉′欢涓嬬殑鐞冨绉板緞鍚戜綅绉荤悊璁鸿В u(rho_p, t)
    """
    if rho_p < 1e-14:
        return 0.0

    rho_grid = np.linspace(0.0, a, n_r_int)
    T_grid = temperature_sphere_dirichlet(
        rho_grid, t, a, kappa, Tinit, Tsurr, n_terms=n_terms
    )
    dT_grid = T_grid - Tinit

    Ia = np.trapz(dT_grid * rho_grid ** 2, rho_grid)


    mask = rho_grid <= rho_p
    rho_sub = rho_grid[mask]
    dT_sub = dT_grid[mask]
    Ir = np.trapz(dT_sub * rho_sub ** 2, rho_sub)

    u = (alpha / (1.0 - nu)) * (
        (1.0 + nu) * Ir / (rho_p**2)
        + 2.0 * (1.0 - 2.0 * nu) * rho_p * Ia / (a**3)
    )
    return u

def horizontal_disp_theory_at_point(r_pt, z_pt, t, a, center_r, center_z,
                                    kappa, alpha, nu, Tinit, Tsurr,
                                    n_terms=100, n_r_int=1000):
    """
    缁欏畾鎴潰鐐?(r_pt, z_pt)锛岃繑鍥炵悊璁烘按骞充綅绉?u_r
    """
    dr = r_pt - center_r
    dz = z_pt - center_z
    rho_p = np.sqrt(dr**2 + dz**2)

    if rho_p < 1e-14:
        return 0.0

    if rho_p > a + 1e-12:
        raise ValueError(f"Point outside sphere: rho_p={rho_p:.6e}, a={a:.6e}")

    u_radial = radial_disp_theory_sphere(
        rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
        n_terms=n_terms, n_r_int=n_r_int
    )

    # 鎶曞奖鍒版按骞?r 鏂瑰悜
    return u_radial * (dr / rho_p)


def compute_melt_and_thermal_expansion(
    T,
    mask_core,
    rho_s, rho_l,
    Ts, Tl,
    cell_volume,
    beta_s,
    beta_l,
    T_ref,
    core_radius=None,
    return_details=False,
    pressure=0.0,
    compressibility_s=0.0,
    compressibility_l=0.0,
    density_beta_s=0.0,
    density_beta_l=0.0,
    density_deltaT=0.0,
):
    T = np.asarray(T, dtype=float)
    core = np.asarray(mask_core, dtype=bool)

    T_core = T[core]

    cv = np.asarray(cell_volume, dtype=float)
    if cv.ndim == 0:
        cv_core = np.full(T_core.shape, float(cv), dtype=float)
    elif cv.shape[0] == T.shape[0]:
        cv_core = cv[core]
    elif cv.shape[0] == T_core.shape[0]:
        cv_core = cv
    else:
        raise ValueError("cell_volume must be scalar, T-sized, or core-sized")

    dT_phase = Tl - Ts
    if dT_phase <= 0.0:
        raise ValueError("Tl must be greater than Ts")

    alpha = (T_core - Ts) / dT_phase
    alpha = np.clip(alpha, 0.0, 1.0)

    V_melt_equiv = float(np.sum(alpha * cv_core))
    core_volume = float(np.sum(cv_core))
    if core_radius is not None:
        core_volume = float((4.0 / 3.0) * np.pi * float(core_radius) ** 3)
    melt_fraction = 0.0 if core_volume <= 0.0 else V_melt_equiv / core_volume
    melt_fraction = float(np.clip(melt_fraction, 0.0, 1.0))

    phase_density = compute_phase_volume_change_ratio_from_density(
        rho_s,
        rho_l,
        melt_fraction,
        pressure=pressure,
        compressibility_s=compressibility_s,
        compressibility_l=compressibility_l,
        beta_s=density_beta_s,
        beta_l=density_beta_l,
        deltaT=density_deltaT,
    )
    if compressibility_s > 0.0 or compressibility_l > 0.0 or pressure > 0.0:
        solid_volume_ref = float(np.sum((1.0 - alpha) * cv_core))
        liquid_volume_ref = V_melt_equiv
        deltaV_phase = (
            solid_volume_ref * phase_density["solid_density_volume_ratio"]
            + liquid_volume_ref * phase_density["liquid_density_volume_ratio"]
        )
    else:
        kappa = (rho_s / rho_l) - 1.0
        deltaV_phase = V_melt_equiv * kappa

    dT_vec = T_core - T_ref
    beta_eff = beta_s * (1.0 - alpha) + beta_l * alpha

    deltaV_thermal = float(np.sum(beta_eff * dT_vec * cv_core))

    if return_details:
        return {
            "deltaV_phase": float(deltaV_phase),
            "deltaV_thermal": float(deltaV_thermal),
            "melt_fraction": float(melt_fraction),
            "melt_volume_equiv": float(V_melt_equiv),
            "core_volume": float(core_volume),
            "phase_density": phase_density,
        }

    return float(deltaV_phase), float(deltaV_thermal)

