import numpy as np

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
