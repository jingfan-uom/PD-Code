import numpy as np
from numba import njit, prange


def compute_lambda_diag_matrix(
    partial_area_flat, dist, indptr, indices,
    r_flat, z_flat,
    c, dt,
    eps=1e-15
):
    N = len(indptr) - 1
    edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))

    # 初始相对位移（按边）
    dx0 = r_flat[indices] - r_flat[edge_i]
    dz0 = z_flat[indices] - z_flat[edge_i]

    d = dist
    safe = d > eps

    # shape_factor = max(|dx|/d, |dz|/d)
    shape_factor = np.zeros_like(d, dtype=float)
    shape_factor[safe] = np.maximum(np.abs(dx0[safe]) / d[safe],
                                    np.abs(dz0[safe]) / d[safe])

    base = (c * partial_area_flat) / np.maximum(d, eps)
    contrib_edge = shape_factor * base * (dt ** 2)

    lambda_scalar = 0.75 * np.bincount(edge_i, weights=np.abs(contrib_edge), minlength=N) * (dt ** 2)

    return lambda_scalar


@njit(parallel=True, fastmath=True)
def compute_local_damping_coefficient_numba(F_curr, F_prev, v_half, lambda_diag, U, dt, eps=1e-30):
    """
    并行计算 cn（标量），等价于原函数，但不分配 Kn_local。
    """
    N = U.size

    # 这两个是全局求和（并行归约）
    num = 0.0
    den = 0.0

    for i in prange(N):
        ui = U[i]
        u2 = ui * ui
        den += u2

        vi = v_half[i]
        if vi != 0.0:
            # delta_force = (F_curr - F_prev) / lambda_diag
            # Kn = -delta_force / (dt * v)
            # => Kn = - (F_curr - F_prev) / (lambda_diag * dt * v)
            lam = lambda_diag[i]
            if lam != 0.0:
                kn = - (F_curr[i] - F_prev[i]) / (lam * dt * vi)
                num += u2 * kn
            # lam==0 时贡献为 0（保持一致/避免除0）

    # 按你的原逻辑做判断
    if den <= eps:
        return 0.0

    ratio = num / den
    if ratio < 0.0:
        return 0.0

    return 2.0 * np.sqrt(ratio)



def adr_update_velocity_displacement(U_prev, V_dot_half_prev, F_curr, c_n, D_diag, dt):

    # Compute updated half-step velocity
    numerator = (2 - c_n * dt) * V_dot_half_prev + 2 * dt * (F_curr / D_diag)
    denominator = 2 + c_n * dt
    V_dot_half = numerator / denominator

    # Update displacement
    U_next = U_prev + dt * V_dot_half

    return V_dot_half, U_next
