
import numpy as np
from numba import njit, prange
import ADR
import Physical_Field_Calculation as pfc
import geometry_utils as geom


def find_closest_point_in_phys(
        phys_coords_list_m,
        target_r,
        target_z,
        candidate_masks_by_slice=None,
):
    best_slice = None
    best_local_idx = None
    best_coord = None
    best_dist2 = np.inf

    for i in range(len(phys_coords_list_m)):
        coords = phys_coords_list_m[i]
        if candidate_masks_by_slice is None:
            candidate_indices = np.arange(coords.shape[0], dtype=np.int64)
        else:
            candidate_mask = np.asarray(candidate_masks_by_slice[i], dtype=bool)
            candidate_indices = np.where(candidate_mask)[0]
            if candidate_indices.size == 0:
                continue

        r_coords = coords[candidate_indices, 0]
        z_coords = coords[candidate_indices, 1]

        dist2 = (r_coords - target_r) ** 2 + (z_coords - target_z) ** 2
        local_pos = np.argmin(dist2)
        local_idx = int(candidate_indices[local_pos])

        if dist2[local_pos] < best_dist2:
            best_dist2 = dist2[local_pos]
            best_slice = i
            best_local_idx = local_idx
            best_coord = coords[local_idx].copy()

    return best_slice, best_local_idx, best_coord


def find_closest_point_to_coord(
        phys_coords_list_m,
        target_coord,
        candidate_masks_by_slice=None,
):
    return find_closest_point_in_phys(
        phys_coords_list_m,
        target_coord[0],
        target_coord[1],
        candidate_masks_by_slice=candidate_masks_by_slice,
    )


def displacement_component_for_point(
        ur_val,
        uz_val,
        coord,
        center_r,
        center_z,
        surface_type=None,
        modify_coordinates=False,
        inner_axis_x=None,
        inner_axis_z=None,
        outer_axis_x=None,
        outer_axis_z=None,
):
    coord = np.asarray(coord, dtype=np.float64)
    dx = float(coord[0] - center_r)
    dz = float(coord[1] - center_z)
    radius_value = float(np.sqrt(dx ** 2 + dz ** 2))
    if radius_value <= 1.0e-30:
        raise RuntimeError("tracked point is too close to the center for displacement projection")

    if modify_coordinates and surface_type in ("inner", "outer"):
        if surface_type == "inner":
            axis_x, axis_z = inner_axis_x, inner_axis_z
        else:
            axis_x, axis_z = outer_axis_x, outer_axis_z
        if axis_x is None or axis_z is None:
            raise ValueError("ellipse axes are required for modified-coordinate surface projection")
        normal = geom.capsule_outward_normals(
            np.array([coord[0]]),
            np.array([coord[1]]),
            center_r,
            center_z,
            axis_x,
            axis_z,
        )[0]
        return float(ur_val * normal[0] + uz_val * normal[1]), radius_value

    radial_displacement = (ur_val * dx + uz_val * dz) / radius_value
    return float(radial_displacement), radius_value


def tracked_displacement_components(pt, Ur_state, Uz_state):
    i_slice = pt["slice"]
    i_local = pt["local_idx"]
    if i_slice is None or i_local is None:
        raise RuntimeError("tracked point is unavailable")
    if pt.get("tracking_mode") == "midline_symmetric_average":
        mirror_slice = pt["mirror_slice"]
        mirror_local_idx = pt["mirror_local_idx"]
        ur_val = 0.5 * (Ur_state[i_slice][i_local] + Ur_state[mirror_slice][mirror_local_idx])
        uz_val = 0.5 * (Uz_state[i_slice][i_local] + Uz_state[mirror_slice][mirror_local_idx])
    else:
        ur_val = Ur_state[i_slice][i_local]
        uz_val = Uz_state[i_slice][i_local]
    return float(ur_val), float(uz_val)


def tracked_radial_displacement(pt, Ur_state, Uz_state, **projection_kwargs):
    ur_val, uz_val = tracked_displacement_components(pt, Ur_state, Uz_state)
    return displacement_component_for_point(
        ur_val,
        uz_val,
        pt["coord"],
        surface_type=pt.get("surface_type"),
        **projection_kwargs,
    )


def make_displacement_phys(phys_coords_list_state, Ur_state, Uz_state):
    U_state = {}
    for i_state in range(len(phys_coords_list_state)):
        n_phys_state = phys_coords_list_state[i_state].shape[0]
        ur_phys = Ur_state[i_state][:n_phys_state]
        uz_phys = Uz_state[i_state][:n_phys_state]
        U_state[i_state] = {
            "Ur": ur_phys,
            "Uz": uz_phys,
            "Umag": np.sqrt(ur_phys ** 2 + uz_phys ** 2),
        }
    return U_state


def state_displacement_components_at_point(
        state,
        Ur_state,
        Uz_state,
        point,
        center_r=0.0,
        center_z=0.0,
        tolerance=0.0,
        modify_coordinates=False,
        inner_axis_x=None,
        inner_axis_z=None,
        outer_axis_x=None,
        outer_axis_z=None,
):
    best_slice, best_local_idx, best_coord = find_closest_point_in_phys(
        state["phys_coords_list_m"],
        point["target_r"],
        point["target_z"],
    )
    if best_slice is None or best_local_idx is None or best_coord is None:
        return np.nan, np.nan, np.nan

    ur_val = Ur_state[best_slice][best_local_idx]
    uz_val = Uz_state[best_slice][best_local_idx]
    track_coord = best_coord
    if abs(point["target_z"] - center_z) <= tolerance:
        mirror_target = np.array([best_coord[0], 2.0 * point["target_z"] - best_coord[1]], dtype=np.float64)
        mirror_slice, mirror_local_idx, mirror_coord = find_closest_point_to_coord(
            state["phys_coords_list_m"],
            mirror_target,
        )
        if mirror_coord is not None and (
                mirror_slice != best_slice or mirror_local_idx != best_local_idx
        ):
            ur_val = 0.5 * (ur_val + Ur_state[mirror_slice][mirror_local_idx])
            uz_val = 0.5 * (uz_val + Uz_state[mirror_slice][mirror_local_idx])
            track_coord = 0.5 * (best_coord + mirror_coord)

    try:
        displacement_value, _ = displacement_component_for_point(
            ur_val,
            uz_val,
            track_coord,
            center_r,
            center_z,
            point.get("surface_type"),
            modify_coordinates,
            inner_axis_x,
            inner_axis_z,
            outer_axis_x,
            outer_axis_z,
        )
    except RuntimeError:
        displacement_value = np.nan
    return float(ur_val), float(uz_val), float(displacement_value)


def state_radial_displacement_at_point(state, Ur_state, Uz_state, point, **projection_kwargs):
    _, _, displacement_value = state_displacement_components_at_point(
        state,
        Ur_state,
        Uz_state,
        point,
        **projection_kwargs,
    )
    return displacement_value


def max_abs_displacement_components(phys_coords_list_state, Ur_state, Uz_state):
    urmax = 0.0
    uzmax = 0.0
    for i_state in range(len(phys_coords_list_state)):
        n_phys_i = phys_coords_list_state[i_state].shape[0]
        if n_phys_i <= 0:
            continue
        ur_i = np.asarray(Ur_state[i_state][:n_phys_i], dtype=np.float64)
        uz_i = np.asarray(Uz_state[i_state][:n_phys_i], dtype=np.float64)
        if ur_i.size:
            urmax = max(urmax, float(np.nanmax(np.abs(ur_i))))
        if uz_i.size:
            uzmax = max(uzmax, float(np.nanmax(np.abs(uz_i))))
    return urmax, uzmax

# ---- 1) è®¡æ•°ï¼šæ¯è¡Œæœ‰å¤šå°‘é‚»å±…ï¼ˆCSR è¡Œé•¿åº¦ï¼‰----
@njit(parallel=True, fastmath=True)
def _count_neighbors_bruteforce(x, z, cutoff2, tol):
    N = x.size
    counts = np.zeros(N, dtype=np.int64)
    tol2 = tol * tol
    for i in prange(N):
        xi = x[i]
        zi = z[i]
        c = 1  # <-- self-edge
        for j in range(N):
            if j == i:
                continue
            dx = xi - x[j]
            dz = zi - z[j]
            d2 = dx*dx + dz*dz
            if d2 <= cutoff2 and d2 > tol2:
                c += 1
        counts[i] = c
    return counts


# ---- 2) å¡«å…… CSRï¼šindices + distï¼ˆæ¯è¡Œå†™è‡ªå·±çš„ç‰‡æ®µï¼Œçº¿ç¨‹å®‰å…¨ï¼‰----
@njit(parallel=True, fastmath=True)
def _fill_neighbors_bruteforce(x, z, indptr, cutoff2, tol, indices, dist):
    N = x.size
    tol2 = tol * tol
    for i in prange(N):
        xi = x[i]
        zi = z[i]
        p = indptr[i]
        end = indptr[i+1]

        # --- write self first ---
        indices[p] = i
        dist[p] = 0.0
        p += 1

        for j in range(N):
            if j == i:
                continue
            dx = xi - x[j]
            dz = zi - z[j]
            d2 = dx*dx + dz*dz
            if d2 <= cutoff2 and d2 > tol2:
                indices[p] = j
                dist[p] = np.sqrt(d2)
                p += 1
                if p == end:
                    break


@njit(parallel=True, fastmath=True)
def compute_partial_area_flat_csr_numba(
    x_flat, z_flat, dx, dz, delta, tolerance,
    indptr, indices, dist,
    sx, sz
):
    # ï¼ˆè¿™é‡Œå°±æ˜¯ä½  docx é‡Œé‚£æ®µå‡½æ•°ä½“ï¼‰
    N = x_flat.size
    nnz = indices.size
    out = np.zeros(nnz, dtype=np.float64)

    diag = (dx*dx + dz*dz) ** 0.5
    cutoff_exclude = delta + 0.5 * diag + tolerance
    delta2 = delta*delta + tolerance
    sub = sx.size

    for i in prange(N):
        cx = x_flat[i]
        cz = z_flat[i]
        a = indptr[i]
        b = indptr[i+1]
        if a == b:
            continue

        for p in range(a, b):
            j = indices[p]
            if j == i:
                out[p] = dx * dz
                continue
            d = dist[p]
            if d > cutoff_exclude:
                out[p] = 0.0
                continue
            xj = x_flat[j]
            zj = z_flat[j]
            x_left  = xj - 0.5*dx
            x_right = xj + 0.5*dx
            z_down  = zj - 0.5*dz
            z_up    = zj + 0.5*dz

            dx1 = x_left  - cx; dz1 = z_down - cz
            in1 = (dx1*dx1 + dz1*dz1) <= delta2
            dx2 = x_left  - cx; dz2 = z_up   - cz
            in2 = (dx2*dx2 + dz2*dz2) <= delta2
            dx3 = x_right - cx; dz3 = z_down - cz
            in3 = (dx3*dx3 + dz3*dz3) <= delta2
            dx4 = x_right - cx; dz4 = z_up   - cz
            in4 = (dx4*dx4 + dz4*dz4) <= delta2

            all_in = in1 and in2 and in3 and in4
            any_in = in1 or in2 or in3 or in4

            if all_in:
                out[p] = dx * dz
            elif not any_in:
                out[p] = 0.0
            else:
                count_in = 0
                for ix in range(sub):
                    x_samp = x_left + sx[ix]*dx
                    dxs = x_samp - cx
                    for iz in range(sub):
                        z_samp = z_down + sz[iz]*dz
                        dzs = z_samp - cz
                        if (dxs*dxs + dzs*dzs) <= delta2:
                            count_in += 1
                out[p] = (dx * dz) * (count_in / (sub*sub))
    return out


def compute_region_matrices(args):
    if len(args) == 4:
        coords, dr, delta, tolerance = args
    else:
        coords, dr, delta, tolerance, _ = args

    x = coords[:, 0].astype(np.float64)
    z = coords[:, 1].astype(np.float64)
    N = x.size

    diag = np.sqrt(dr*dr + dr*dr)
    cutoff = delta + 0.5*diag + tolerance
    cutoff2 = cutoff * cutoff

    # 1) count neighbors
    counts = _count_neighbors_bruteforce(x, z, cutoff2, tolerance)

    # 2) build indptr
    indptr = np.empty(N + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    nnz = indptr[-1]

    # 3) fill indices + dist
    indices = np.empty(nnz, dtype=np.int64)
    dist = np.empty(nnz, dtype=np.float64)
    _fill_neighbors_bruteforce(x, z, indptr, cutoff2, tolerance, indices, dist)

    # 4) CSR partial area (10Ã—10 é‡‡æ ·ç‚¹)

    sub = 10
    sx = (np.arange(sub) + 0.5) / sub
    sz = (np.arange(sub) + 0.5) / sub

    area = compute_partial_area_flat_csr_numba(
        x, z, dr, dr, delta, tolerance,
        indptr, indices, dist,
        sx.astype(np.float64), sz.astype(np.float64)
    )

    return indptr, indices, dist, area

# mechanical_calculations.py

from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def compute_accel_osbpd_axisym_csr_numba(
    indptr, indices, area_edge,
    eij_edge,                # (nnz,) nlen - dist
    n_r_edge, n_z_edge,      # (nnz,)

    lamda_edge,              # (nnz,)
    miu_edge,                # (nnz,)
    lamda_node,              # (N,)
    miu_node,                # (N,)

    r_node,                  # (N,)
    Ur,                      # (N,)
    dilation,                # (N,)
    rho_node,                # (N,)
    br, bz,                  # (N,)
    delta,                   # float64
    T_m, Tpre_avg,
    kprime_edge,             # (nnz,)
    kprime_node,             # (N,)
    use_kprime_node,          # scalar int: 1 -> endpoint kprime_node, 0 -> edge kprime
    csr,                     # (nnz,)
    use_node_material_params=False,
    eps=1e-15
):
    N = Ur.size
    Ar = np.zeros(N, dtype=np.float64)
    Az = np.zeros(N, dtype=np.float64)

    coeff1 = 3.0 / (np.pi * (delta ** 3))
    coeff2 = 12.0 / (np.pi * (delta ** 3))

    for i in prange(N):
        ri = r_node[i]
        inv_ri = 0.0 if ri <= eps else 1.0 / ri

        Fi_r = 0.0
        Fi_z = 0.0

        theta_i = dilation[i]
        Uri = Ur[i]
        dTi = T_m[i] - Tpre_avg

        a0 = indptr[i]
        b0 = indptr[i + 1]

        lam_geom_sum = 0.0
        mu_geom_sum = 0.0
        kp_geom_sum = 0.0
        geom_weight_sum = 0.0

        for p in range(a0, b0):
            j = indices[p]
            if j == i:
                continue

            rj = r_node[j]
            theta_j = dilation[j]
            dTj = T_m[j] - Tpre_avg

            if use_node_material_params:
                lam_i = lamda_node[i]
                mu_i = miu_node[i]
                kp_i = kprime_node[i]
                lam_j = lamda_node[j]
                mu_j = miu_node[j]
                kp_j = kprime_node[j]
                kp_geom = kprime_node[i]
            else:
                lam_e = lamda_edge[p]
                mu_e = miu_edge[p]
                lam_i = lam_e
                mu_i = mu_e
                lam_j = lam_e
                mu_j = mu_e
                if use_kprime_node == 1:
                    kp_i = kprime_node[i]
                    kp_j = kprime_node[j]
                    kp_geom = kp_i
                else:
                    kp_i = kprime_edge[p]
                    kp_j = kprime_edge[p]
                    kp_geom = kprime_edge[p]

            # stretch on this bond
            stretch = eij_edge[p]

            ti = lam_i * theta_i + (lam_i + mu_i) * (Uri * inv_ri) - kp_i * dTi
            tj = lam_j * theta_j + (lam_j + mu_j) * (Uri * inv_ri) - kp_j * dTj

            base_i = coeff1 * ti + coeff2 * mu_i * stretch
            base_j = coeff1 * tj + coeff2 * mu_j * stretch
            s_i = rj / ri

            dforce_i = area_edge[p] * base_i * s_i * csr[p]
            dforce_j = area_edge[p] * base_j * s_i * csr[p]

            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce * 0.5
            Fi_z += n_z_edge[p] * dforce * 0.5

            geom_weight = np.abs(area_edge[p] * s_i * csr[p])
            if (not use_node_material_params) and geom_weight > eps:
                lam_geom_sum += geom_weight * lam_i
                mu_geom_sum += geom_weight * mu_i
                kp_geom_sum += geom_weight * kp_geom
                geom_weight_sum += geom_weight

        if use_node_material_params:
            lam_geom_i = lamda_node[i]
            mu_geom_i = miu_node[i]
            kp_geom_i = kprime_node[i]
        elif geom_weight_sum > eps:
            lam_geom_i = lam_geom_sum / geom_weight_sum
            mu_geom_i = mu_geom_sum / geom_weight_sum
            kp_geom_i = kp_geom_sum / geom_weight_sum
        else:
            lam_geom_i = 0.0
            mu_geom_i = 0.0
            kp_geom_i = 0.0

        # ---- geometric term: use row-averaged edge/interface-bond parameters ----
        br_geom = -1.0 * inv_ri * (
            theta_i * (lam_geom_i + mu_geom_i) +
            (lam_geom_i + 3.0 * mu_geom_i) * Uri * inv_ri
            - kp_geom_i * dTi
        )

        Ar[i] = (Fi_r + br_geom + br[i]) / rho_node[i]
        Az[i] = (Fi_z + bz[i]) / rho_node[i]

    return Ar, Az

from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_damage_variable_axisym_csr(
    indptr,
    indices,
    area_edge,
    shape_edge,
    csr,
    crack,
    damage_bond_mask,
    eps=1e-30,
):
    N = indptr.size - 1
    phi = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        total_weight = 0.0
        active_weight = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j == i or not damage_bond_mask[p]:
                continue
            weight = abs(area_edge[p] * shape_edge[p] * csr[p])
            if weight <= eps:
                continue
            total_weight += weight
            if crack[p] != 0:
                active_weight += weight
        if total_weight > eps:
            phi[i] = max(0.0, min(1.0, 1.0 - active_weight / total_weight))

    return phi


@njit(fastmath=True)
def has_outer_surface_damage_node(
    damage_node,
    outer_surface_node_mask,
    damage_threshold,
):
    N = damage_node.size
    for i in range(N):
        if outer_surface_node_mask[i] and damage_node[i] > damage_threshold:
            return True
    return False


@njit(parallel=True, fastmath=True)
def sum_squared_increment_numba(Ur, Uz, Ur_prev, Uz_prev):
    total = 0.0
    for i in prange(Ur.size):
        dur = Ur[i] - Ur_prev[i]
        duz = Uz[i] - Uz_prev[i]
        total += dur * dur + duz * duz
    return total


@njit(parallel=True, fastmath=True)
def compute_accel_osbpd_axisym_csr_inplace_mu(
    indptr, indices, area_edge,
    shape_edge,
    eij_edge,
    n_r_edge, n_z_edge,
    crack, s0_edge,

    lamda_edge, miu_edge,

    r_node,
    Ur,
    dilation,
    br, bz,
    delta,
    T_m, Tpre_avg,

    kprime_edge,
    alpha_edge,
    csr,
    damage_node,
    crack_pressure,
    damage_threshold,
    enable_crack_pressure_force,
    enable_cracking=True,
    eps=1e-30,
):
    N = Ur.size
    Ar = np.zeros(N, dtype=np.float64)
    Az = np.zeros(N, dtype=np.float64)

    coeff1 = 3.0 / (np.pi * (delta ** 3))
    coeff2 = 12.0 / (np.pi * (delta ** 3))
    crack_pressure_coeff = 6.0 * crack_pressure / (np.pi * (delta ** 3))

    for i in prange(N):
        ri = r_node[i]
        inv_ri = 1.0 / (ri + eps)

        Fi_r = 0.0
        Fi_z = 0.0

        theta_i = dilation[i]
        Uri = Ur[i]
        dTi = T_m[i] - Tpre_avg

        a0 = indptr[i]
        b0 = indptr[i + 1]

        lam_geom_sum = 0.0
        mu_geom_sum = 0.0
        kp_geom_sum = 0.0
        geom_weight_sum = 0.0

        for p in range(a0, b0):
            j = indices[p]

            if j == i:
                continue
            if enable_cracking and crack[p] == 0:
                if (
                    enable_crack_pressure_force
                    and crack_pressure > eps
                    and damage_node[i] > damage_threshold
                    and damage_node[j] > damage_threshold
                ):
                    n_len = np.sqrt(n_r_edge[p] * n_r_edge[p] + n_z_edge[p] * n_z_edge[p])
                    if n_len > eps:
                        ri_current = ri + Uri
                        rj_current = r_node[j] + Ur[j]
                        r_face = 0.5 * (ri_current + rj_current)
                        axisym_crack_weight = max(r_face, 0.0) / max(ri_current, eps)
                        crack_force = area_edge[p] * crack_pressure_coeff * axisym_crack_weight * csr[p]
                        Fi_r -= (n_r_edge[p] / n_len) * crack_force
                        Fi_z -= (n_z_edge[p] / n_len) * crack_force
                continue

            theta_j = dilation[j]
            dTj = T_m[j] - Tpre_avg

            lam_e = lamda_edge[p]
            mu_e = miu_edge[p]
            kp_e = kprime_edge[p]
            alpha_e = alpha_edge[p]
            stretch = eij_edge[p]

            if enable_cracking and (stretch - alpha_e * 0.5 * (dTi + dTj) > s0_edge[p]):
                crack[p] = 0
                continue

            ti = lam_e * theta_i + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTi
            tj = lam_e * theta_j + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTj

            base_i = coeff1 * ti + coeff2 * mu_e * stretch
            base_j = coeff1 * tj + coeff2 * mu_e * stretch

            s_i = shape_edge[p]
            dforce_i = area_edge[p] * base_i * s_i * csr[p]
            dforce_j = area_edge[p] * base_j * s_i * csr[p]
            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce * 0.5
            Fi_z += n_z_edge[p] * dforce * 0.5

            geom_weight = area_edge[p] * s_i * csr[p]
            if geom_weight > eps:
                lam_geom_sum += geom_weight * lam_e
                mu_geom_sum += geom_weight * mu_e
                kp_geom_sum += geom_weight * kp_e
                geom_weight_sum += geom_weight

        if geom_weight_sum > eps:
            lam_geom_i = lam_geom_sum / geom_weight_sum
            mu_geom_i = mu_geom_sum / geom_weight_sum
            kp_geom_i = kp_geom_sum / geom_weight_sum
        else:
            lam_geom_i = 0.0
            mu_geom_i = 0.0
            kp_geom_i = 0.0

        br_geom = -inv_ri * (
            theta_i * (lam_geom_i + mu_geom_i) +
            (lam_geom_i + 3.0 * mu_geom_i) * Uri * inv_ri
            - kp_geom_i * dTi
        )

        Ar[i] = Fi_r + br_geom + br[i]
        Az[i] = Fi_z + bz[i]

    return Ar, Az

def compute_mechanical_step_csr(
    indptr, indices, dist, area,
    r_flat, z_flat,
    shape_edge, coeff_dilation,

    lamda_edge, miu_edge,

    crack, s0_edge,
    Ur, Uz, br, bz,
    Fr_prev, Fz_prev,
    Vr_half, Vz_half,
    lambda_diag, dt_ADR,
    delta, T_m, T_prev,

    kprime_edge,
    alpha_edge,
    csr,
    ghost_dilation_indices=None,
    ghost_dilation_anchor_indices=None,
    enable_cracking=True,
    damage_node=None,
    crack_pressure=0.0,
    damage_threshold=0.24,
    enable_crack_pressure_force=False,
    use_crack_in_dilation=False,
):
    if damage_node is None:
        damage_node = np.zeros_like(Ur)

    # 1) dilation + eij + n ï¼ˆä¸€æ¬¡ nnz éåŽ†ï¼‰
    if use_crack_in_dilation:
        dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr_rows_crack_numba(
            indptr, indices,
            r_flat, z_flat, Ur, Uz,
            dist,
            area,
            shape_edge,
            coeff_dilation, csr,
            crack,
        )
    else:
        dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr_rows_numba(
            indptr, indices,
            r_flat, z_flat, Ur, Uz,
            dist,
            area,
            shape_edge,
            coeff_dilation, csr,
        )
    if ghost_dilation_indices is not None and ghost_dilation_anchor_indices is not None:
        if ghost_dilation_indices.size:
            dilation[ghost_dilation_indices] = dilation[ghost_dilation_anchor_indices]
    # 2) acceleration
    Fr, Fz = compute_accel_osbpd_axisym_csr_inplace_mu(
        indptr, indices, area,
        shape_edge,
        eij_edge,
        n_r_edge, n_z_edge,
        crack, s0_edge,

        lamda_edge, miu_edge,

        r_flat,
        Ur,
        dilation,
        br, bz,
        delta,
        T_m, T_prev,

        kprime_edge,
        alpha_edge,
        csr,
        damage_node,
        crack_pressure,
        damage_threshold,
        enable_crack_pressure_force,
        enable_cracking,
    )


    Fr_curr = Fr
    Fz_curr = Fz

    cr_n = ADR.compute_local_damping_coefficient_numba(
        Fr_curr, Fr_prev, Vr_half, lambda_diag, Ur, dt_ADR
    )
    cz_n = ADR.compute_local_damping_coefficient_numba(
        Fz_curr, Fz_prev, Vz_half, lambda_diag, Uz, dt_ADR
    )


    Fr_prev = Fr_curr
    Fz_prev = Fz_curr


    Vr_half, Ur = ADR.adr_update_velocity_displacement(
        Ur, Vr_half, Fr_curr, cr_n, lambda_diag, dt_ADR
    )
    Vz_half, Uz = ADR.adr_update_velocity_displacement(
        Uz, Vz_half, Fz_curr, cz_n, lambda_diag, dt_ADR
    )

    return Ur, Uz, Fr_prev, Fz_prev, Vr_half, Vz_half
