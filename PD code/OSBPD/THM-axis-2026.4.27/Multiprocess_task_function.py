
from numba import njit, prange
import ADR
import Physical_Field_Calculation as pfc

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
    coords, dr, delta, tolerance, slice_id = args

    x = coords[:, 0].astype(np.float64)
    z = coords[:, 1].astype(np.float64)
    N = x.size

    # ä¸Ž partial-area çš„ cutoff å¯¹é½ï¼ˆæ¥è‡ªä½  docx çš„å®šä¹‰ï¼‰:contentReference[oaicite:2]{index=2}
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
    #    sx/sz å»ºè®®å›ºå®šä¸º 10 ä¸ªç‚¹ï¼Œé¿å…æ¯æ¬¡åˆ›å»ºä¸åŒé•¿åº¦
    sub = 10
    # ä¾‹å¦‚ç­‰è·é‡‡æ · [0.05, 0.15, ..., 0.95]
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
    dist_edge,               # (nnz,) |x_ij|
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
    csr,                     # (nnz,)
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

        # ---- i ç‚¹ node å‚æ•°ï¼šåªç»™ br_geom ç”¨ ----
        lam_i_node = lamda_node[i]
        mu_i_node  = miu_node[i]
        kp_i_node  = kprime_node[i]

        a0 = indptr[i]
        b0 = indptr[i + 1]

        for p in range(a0, b0):
            j = indices[p]
            if j == i:
                continue

            rj = r_node[j]
            inv_rj = 0.0 if rj <= eps else 1.0 / rj

            theta_j = dilation[j]
            Urj = Ur[j]
            dTj = T_m[j] - Tpre_avg

            # ---- è¿™æ¡ bond çš„ edge å‚æ•° ----
            lam_e = lamda_edge[p]
            mu_e  = miu_edge[p]
            kp_e  = kprime_edge[p]

            # stretch on this bond
            stretch = eij_edge[p]

            # ---- bond constitutive response: ç”¨ edge å‚æ•° ----
            ti = lam_e * theta_i + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTi
            tj = lam_e * theta_j + (lam_e + mu_e) * (Urj * inv_rj) - kp_e * dTj

            base_i = coeff1 * ti + coeff2 * mu_e * stretch
            base_j = coeff1 * tj + coeff2 * mu_e * stretch
            s_i = rj / ri

            dforce_i = area_edge[p] * base_i * s_i * csr[p]
            dforce_j = area_edge[p] * base_j * s_i * csr[p]

            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce * 0.5
            Fi_z += n_z_edge[p] * dforce * 0.5

        # ---- å‡ ä½•é¡¹ï¼šç”¨ node å‚æ•° ----
        br_geom = -1.0 * inv_ri * (
            theta_i * (lam_i_node + mu_i_node) +
            (lam_i_node + 3.0 * mu_i_node) * Uri * inv_ri
            - kp_i_node * dTi
        )

        inv_rho = 1.0 / (rho_node[i] + eps)
        Ar[i] = (Fi_r + br_geom + br[i]) * inv_rho
        Az[i] = (Fi_z + bz[i]) * inv_rho

    return Ar, Az

from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_accel_osbpd_axisym_csr_inplace_mu(
    indptr, indices, area_edge,
    shape_edge,                  # (nnz,) å¯¹ i ç«¯ï¼šrj/ri
    dist_edge,                   # (nnz,)
    eij_edge,                    # (nnz,)
    n_r_edge, n_z_edge,          # (nnz,)
    crack, s0_edge,              # (nnz,)

    lamda_edge, miu_edge,        # (nnz,)
    lamda_node, miu_node,        # (N,)

    r_node,                      # (N,)
    Ur,                          # (N,)
    dilation,                    # (N,)
    rho_node,                    # (N,)
    br, bz,                      # (N,)
    delta,                       # scalar
    cracking_act, T_m, Tpre_avg,

    kprime_edge,                 # (nnz,)
    kprime_node,                 # (N,)
    alpha_edge,                  # (nnz,)
    csr,                       # (nnz,)
    eps=1e-30
):
    N = Ur.size
    Ar = np.zeros(N, dtype=np.float64)
    Az = np.zeros(N, dtype=np.float64)

    coeff1 = 3.0 / (np.pi * (delta ** 3))
    coeff2 = 12.0 / (np.pi * (delta ** 3))

    for i in prange(N):
        ri = r_node[i]
        inv_ri = 1.0 / (ri + eps)

        Fi_r = 0.0
        Fi_z = 0.0

        theta_i = dilation[i]
        Uri = Ur[i]
        dTi = T_m[i] - Tpre_avg

        # ---- i ç‚¹ node å‚æ•°ï¼šä½œä¸º fallback ----
        lam_i_node = lamda_node[i]
        mu_i_node = miu_node[i]
        kp_i_node = kprime_node[i]

        # ---- ä¸º br_geom æž„é€  edge åŠ æƒå¹³å‡å‚æ•° ----
        lam_sum = 0.0
        mu_sum = 0.0
        kp_sum = 0.0
        w_sum = 0.0

        a0 = indptr[i]
        b0 = indptr[i + 1]

        for p in range(a0, b0):
            j = indices[p]

            if j == i:
                continue
            if crack[p] == 0:
                continue

            # ---- j node fields ----
            rj = r_node[j]
            inv_rj = 1.0 / (rj + eps)

            theta_j = dilation[j]
            Urj = Ur[j]
            dTj = T_m[j] - Tpre_avg

            # ---- bond å‚æ•° ----
            lam_e = lamda_edge[p]
            mu_e = miu_edge[p]
            kp_e = kprime_edge[p]
            alpha_e = alpha_edge[p]

            # ---- æ–­è£‚åˆ¤æ® ----
            stretch = eij_edge[p]

            if cracking_act and (stretch - alpha_e * 0.5 * (dTi + dTj) > s0_edge[p]):
                crack[p] = 0
                continue

            w = area_edge[p]
            lam_sum += lam_e * w * csr[p]
            mu_sum += mu_e * w * csr[p]
            kp_sum += kp_e * w * csr[p]
            w_sum += w * csr[p]

            ti = lam_e * theta_i + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTi
            tj = lam_e * theta_j + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTj

            base_i = coeff1 * ti + coeff2 * mu_e * stretch
            base_j = coeff1 * tj + coeff2 * mu_e * stretch

            s_i = shape_edge[p]   # rj/ri

            dforce_i = area_edge[p] * base_i * s_i * csr[p]
            dforce_j = area_edge[p] * base_j * s_i * csr[p]
            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce * 0.5
            Fi_z += n_z_edge[p] * dforce * 0.5

        lam_geom_i = lam_i_node
        mu_geom_i =  mu_i_node
        kp_geom_i = kp_i_node

        br_geom = -inv_ri * (
            theta_i * (lam_geom_i + mu_geom_i) +
            (lam_geom_i + 3.0 * mu_geom_i) * Uri * inv_ri
            - kp_geom_i * dTi
        )

        Ar[i] = (Fi_r + br_geom)
        Az[i] = (Fi_z )

    return Ar, Az

def compute_mechanical_step_csr(
    indptr, indices, dist, area,
    r_flat, z_flat, edge_i,
    shape_edge, coeff_dilation,

    lamda_edge, miu_edge,        # (nnz,)
    lamda_node, miu_node,        # (N,)

    crack, s0_edge,
    rho_node,
    Ur, Uz, br, bz,
    Fr_prev, Fz_prev,
    Vr_half, Vz_half,
    lambda_diag, dt_ADR,
    delta, cracking_act, T_m, T_prev,

    kprime_edge,                 # (nnz,)
    kprime_node,                 # (N,)
    alpha_edge,                  # (nnz,)
    csr
):
    # 1) dilation + eij + n ï¼ˆä¸€æ¬¡ nnz éåŽ†ï¼‰
    dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr(
        r_flat, z_flat, Ur, Uz,
        edge_i, indices,
        dist,
        area,
        shape_edge,
        coeff_dilation,csr
    )

    # 2) acceleration
    Fr, Fz = compute_accel_osbpd_axisym_csr_inplace_mu(
        indptr, indices, area,
        shape_edge,
        dist,
        eij_edge,
        n_r_edge, n_z_edge,
        crack, s0_edge,

        lamda_edge, miu_edge,
        lamda_node, miu_node,

        r_flat,
        Ur,
        dilation,
        rho_node,
        br, bz,
        delta,
        cracking_act, T_m, T_prev,

        kprime_edge,
        kprime_node,
        alpha_edge,
        csr
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

