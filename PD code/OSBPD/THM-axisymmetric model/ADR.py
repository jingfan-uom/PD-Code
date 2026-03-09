
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


import numpy as np
from numba import njit, prange

#@njit(parallel=True, fastmath=True)
def compute_lambda_diag_matrix_axsy(
    indptr, indices,
    area_edge,          # (nnz,) 体积权重 dVj（或你定义的权重）
    dist_edge,          # (nnz,) |xi|
    w_maxe_edge,        # (nnz,) max(|xi·e|)/|xi|  (你要的 max-e 权重)
    shape_edge,         # (nnz,) 形状因子（若不用就传 ones）
    lamda_node,         # (N,)
    miu_node,           # (N,)
    r_node,             # (N,)
    delta,              # float
    q,                  # float
    eps=1e-15,
):
    """
    Compute diag-like quantity for ADR:
        diagKt[i] = sum_{j in Hi} | w_ij * (Kt_i + Kt_j) * dVj |
    Only integrates Kt_i and Kt_j parts (NO Kb term here).

    Returns
    -------
    diagKt : (N,) float64
    """
    N = lamda_node.size
    diagKt = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        ri = r_node[i]
        inv_ri = 1.0 / ri

        lam_i = lamda_node[i]
        mu_i  = miu_node[i]
        alpha_i = mu_i                  # alpha = mu
        beta_i  = 8.0 * mu_i / q        # beta = 8mu/q
        gamma_i = 3.0 * mu_i

        a0 = indptr[i]
        b0 = indptr[i + 1]

        acc = 0.0
        Kb_i = 0.0

        for p in range(a0, b0):
            j = indices[p]
            if j == i:
                continue

            rj = r_node[j]
            inv_rj = 1.0 / rj

            # bond geometry
            dist = dist_edge[p]
            if dist <= eps:
                continue

            omega = delta / dist

            # node j material
            lam_j = lamda_node[j]
            mu_j  = miu_node[j]
            alpha_j = mu_j
            beta_j  = 8.0 * mu_j / q

            # ---- Kt(x,xi): from your Eq.(55) structure ----
            # Kt = 4*omega*lambda'/q + (lambda'+alpha)*2*omega*<x·xi>/(q*r) + beta*omega
            Kt_i = (4.0 * omega * lam_i / q) + ((lam_i + alpha_i) * (2.0 * delta / q) * inv_ri) + (beta_i * omega) * shape_edge[p] * 2 * np.pi
            Kt_j = (4.0 * omega * lam_j / q) + ((lam_j + alpha_j) * (2.0 * delta / q) * inv_rj) + (beta_j * omega) / shape_edge[p] * 2 * np.pi

            # ---- assemble contribution ----

            contrib =  (Kt_i + Kt_j) * area_edge[p] * w_maxe_edge[p]

            Kb_i += area_edge[p] * shape_edge[p]
            acc += contrib

        Kb = (lam_i + gamma_i) * (inv_ri * inv_ri) + 4 * delta * np.pi * Kb_i * inv_ri * (lam_i + alpha_i) / q
        diagKt[i] =   acc + Kb

    return diagKt/4


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
