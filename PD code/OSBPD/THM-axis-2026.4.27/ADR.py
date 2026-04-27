
import numpy as np

def compute_lambda_diag_matrix(
    partial_area_flat, dist, indptr,
    c_edge, dt,
    edge_i
):
    N = len(indptr) - 1
    d = dist

    mask = (d > 1e-15) & (partial_area_flat != 0.0)
    base = (c_edge[mask] * partial_area_flat[mask]) / d[mask]   # (nnz_masked,)
    lambda_diag = 0.5 * np.bincount(
        edge_i[mask],
        weights=np.abs(base),
        minlength=N
    ) * (dt ** 2)

    return lambda_diag

from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def compute_lambda_diag_matrix_axsy(
    indptr, indices,
    area_edge,          # (nnz,) 体积权重 dVj（或你定义的权重）
    dist_edge,          # (nnz,) bond length
    w_maxe_edge,        # (nnz,) max(|xi·e|)/|xi|
    shape_edge,         # (nnz,) shape factor
    lamda_edge,         # (nnz,)
    miu_edge,           # (nnz,)
    r_node,             # (N,)
    delta,              # float
    q,                  # float
    eps=1e-15,
):
    """
    Compute diag-like quantity for ADR using edge-wise material parameters.

    Notes
    -----
    1. Kt_i / Kt_j use edge parameters directly.
    2. Kb is built from row-wise accumulated edge parameters.
    3. r_node is still used for axisymmetric geometric terms.
    """

    N = len(indptr) - 1
    diagKt = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        ri = r_node[i]
        if ri <= eps:
            ri = eps
        inv_ri = 1.0 / ri

        a0 = indptr[i]
        b0 = indptr[i + 1]

        acc = 0.0
        Kb_i = 0.0

        # row-wise accumulation for Kb
        sum_lam = 0.0
        sum_mu = 0.0
        n_eff = 0.0

        for p in range(a0, b0):
            j = indices[p]
            if j == i:
                continue

            dist = dist_edge[p]
            if dist <= eps:
                continue

            A = area_edge[p]
            if A == 0.0:
                continue

            shp = shape_edge[p]
            if np.abs(shp) <= eps:
                continue

            rj = r_node[j]
            if rj <= eps:
                rj = eps
            inv_rj = 1.0 / rj

            omega = delta / dist

            # ---- edge material parameters ----
            lam_e = lamda_edge[p]
            mu_e  = miu_edge[p]

            alpha_e = mu_e
            beta_e  = 8.0 * mu_e / q

            # ---- Kt(x,xi): now fully based on edge parameters ----
            Kt_i = (
                4.0 * omega * lam_e / q
                + ((lam_e + alpha_e) * (2.0 * delta / q) * inv_ri)
                + (beta_e * omega) * shp * 2.0 * np.pi
            )

            Kt_j = (
                4.0 * omega * lam_e / q
                + ((lam_e + alpha_e) * (2.0 * delta / q) * inv_rj)
                + (beta_e * omega) / shp * 2.0 * np.pi
            )

            contrib = (Kt_i + Kt_j) * A * w_maxe_edge[p]
            acc += np.abs(contrib)

            # ---- accumulate for Kb ----
            Kb_i += A * shp
            sum_lam += lam_e
            sum_mu += mu_e
            n_eff += 1.0

        # ---- row-averaged edge material for Kb ----
        if n_eff > 0.0:
            lam_bar = sum_lam / n_eff
            mu_bar  = sum_mu / n_eff
            alpha_bar = mu_bar
            gamma_bar = 3.0 * mu_bar

            Kb = (
                (lam_bar + gamma_bar) * (inv_ri * inv_ri)
                + 4.0 * delta * np.pi * Kb_i * inv_ri * (lam_bar + alpha_bar) / q
            )
        else:
            Kb = 0.0

        diagKt[i] = 0.25 * (acc + np.abs(Kb))

    return diagKt


from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_lambda_diag_csr_numba(indptr, indices, dist, area, c_edge, mu_edge, eps=1e-30):
    """
    返回 lambda_diag: (N,)
    建议形式：lambda_i = sum_{j} mu_ij * c_ij * A_ij / dist_ij
    """
    N = len(indptr) - 1
    lam = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        a = indptr[i]
        b = indptr[i+1]
        s = 0.0
        for p in range(a, b):
            j = indices[p]
            if j == i:
                continue
            dij = dist[p]
            if dij <= eps:
                continue
            aij = area[p]
            if aij == 0.0:
                continue
            if mu_edge[p] == 0:
                continue
            # 关键权重（你也可以换成 /dij^2，如果你的力公式里是这样）
            s += c_edge[p] * aij / dij
        lam[i] = max(s, eps)  # 防止为0
    return lam

def adr_update_velocity_displacement(U_prev, V_dot_half_prev, F_curr, c_n, D_diag, dt):

    # Compute updated half-step velocity
    numerator = (2 - c_n * dt) * V_dot_half_prev + 2 * dt * (F_curr / D_diag)
    denominator = 2 + c_n * dt
    V_dot_half = numerator / denominator

    # Update displacement
    U_next = U_prev + dt * V_dot_half

    return V_dot_half, U_next

@njit(parallel=True, fastmath=True)
def compute_local_damping_coefficient_numba(F_curr, F_prev, v_half, lambda_diag, U, dt, eps=1e-30):
    N = U.size
    num = 0.0
    den = 0.0

    for i in prange(N):
        ui = U[i]
        u2 = ui * ui
        den += u2

        vi = v_half[i]
        if vi != 0.0:
            lam = lambda_diag[i]
            if lam != 0.0:
                kn = - (F_curr[i] - F_prev[i]) / (lam * dt * vi)
                num += u2 * kn

    if den <= eps:
        return 0.0

    ratio = num / den
    if ratio < 0.0:
        return 0.0

    return 2.0 * np.sqrt(ratio)
