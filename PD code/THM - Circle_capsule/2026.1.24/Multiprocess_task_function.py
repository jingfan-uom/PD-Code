
import numpy as np
from numba import njit, prange
import ADR

# ---- 1) 计数：每行有多少邻居（CSR 行长度）----
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


# ---- 2) 填充 CSR：indices + dist（每行写自己的片段，线程安全）----
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


# ---- 3) 你 DOCX 里的 CSR partial-area 核心（直接用）----
#    这里我“按原样”引用结构：compute_partial_area_flat_csr_numba(...)
#    建议你把这段函数放在 region_utils.py 里，或从 area_matrix_calculator_csr 导入。


@njit(parallel=True, fastmath=True)
def compute_partial_area_flat_csr_numba(
    x_flat, z_flat, dx, dz, delta, tolerance,
    indptr, indices, dist,
    sx, sz
):
    # （这里就是你 docx 里那段函数体）
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

    # 与 partial-area 的 cutoff 对齐（来自你 docx 的定义）:contentReference[oaicite:2]{index=2}
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
    indices = np.empty(nnz, dtype=np.int32)
    dist = np.empty(nnz, dtype=np.float64)
    _fill_neighbors_bruteforce(x, z, indptr, cutoff2, tolerance, indices, dist)

    # 4) CSR partial area (10×10 采样点)
    #    sx/sz 建议固定为 10 个点，避免每次创建不同长度
    sub = 10
    # 例如等距采样 [0.05, 0.15, ..., 0.95]
    sx = (np.arange(sub) + 0.5) / sub
    sz = (np.arange(sub) + 0.5) / sub

    area = compute_partial_area_flat_csr_numba(
        x, z, dr, dr, delta, tolerance,
        indptr, indices, dist,
        sx.astype(np.float64), sz.astype(np.float64)
    )

    return indptr, indices, dist, area

# mechanical_calculations.py


@njit(parallel=True, fastmath=True)
def compute_accel_initial_csr_numba(
    indptr, indices, area,
    dir_r_edge, dir_z_edge,
    c_edge, nu_edge, alpha_edge,
    rho_edge,
    rho_node,
    Ur, Uz,
    br, bz,
    T_m, T_prev,
    coords,             # ✅ 只传 coords
):
    N = Ur.size
    Ar = np.zeros(N, dtype=np.float64)
    Az = np.zeros(N, dtype=np.float64)

    dT_node = T_m - T_prev

    for i in prange(N):
        ui = Ur[i]
        wi = Uz[i]
        dTi = dT_node[i]

        a_r = 0.0
        a_z = 0.0

        a = indptr[i]
        b = indptr[i + 1]

        ri = coords[i, 0]
        zi = coords[i, 1]

        for p in range(a, b):
            j = indices[p]
            if j == i:
                continue

            rj = coords[j, 0]
            zj = coords[j, 1]

            aij = area[p]

            du = Ur[j] - ui
            dw = Uz[j] - wi

            # dense 一致：用坐标差算 L0/L1
            dx0 = rj - ri
            dz0 = zj - zi
            L0 = np.sqrt(dx0*dx0 + dz0*dz0)

            dx1 = dx0 + du
            dz1 = dz0 + dw
            L1 = np.sqrt(dx1*dx1 + dz1*dz1)

            eij = (L1 - L0) / L0

            dT_edge = 0.5 * (dTi + dT_node[j])
            eff = eij - (1.0 + nu_edge[p]) * alpha_edge[p] * dT_edge

            coef = (c_edge[p] * eff * aij) / rho_edge[p]

            a_r += coef * dir_r_edge[p]
            a_z += coef * dir_z_edge[p]

        rden = rho_node[i]

        Ar[i] = a_r + br[i] / rden
        Az[i] = a_z + bz[i] / rden

    return Ar, Az

@njit(parallel=True, fastmath=True)
def compute_accel_csr_numba_inplace_mu(
        indptr, indices,
        dist, area,
        dx0_edge, dz0_edge,  # ✅ 预计算好的参考构型分量
        dir_r_edge, dir_z_edge,  # 仍然保留（用于把标量力投影到 r/z）
        c_edge, nu_edge, alpha_edge,
        mu_edge, s0_edge,
        rho_edge,  # (nnz,)
        rho_node,  # (N,)
        Ur, Uz,  # (N,)
        br, bz,  # (N,)
        T_m, T_prev, step # (N,)
):
    N = Ur.size
    Ar = np.zeros(N, dtype=np.float64)
    Az = np.zeros(N, dtype=np.float64)

    dT_node = T_m - T_prev

    for i in prange(N):
        ui = Ur[i]
        wi = Uz[i]
        dTi = dT_node[i]

        a_r = 0.0
        a_z = 0.0

        a = indptr[i]
        b = indptr[i + 1]

        for p in range(a, b):
            j = indices[p]
            if j == i:
                continue

            dij = dist[p]  # L0
            aij = area[p]

            du = Ur[j] - ui
            dw = Uz[j] - wi

            # dense 一致：eij = (L1 - L0) / L0
            dx0 = dx0_edge[p]
            dz0 = dz0_edge[p]

            dx1 = dx0 + du
            dz1 = dz0 + dw

            L1 = np.sqrt(dx1 * dx1 + dz1 * dz1)
            eij = (L1 - dij) / dij

            # failure update（拉压都断；如只拉断就改为 eij > s0_edge[p]）
            if step > 50:
                if mu_edge[p] == 1:
                    if eij > s0_edge[p]:
                        mu_edge[p] = 0
                        continue

            dT_edge = 0.5 * (dTi + dT_node[j])
            eff = eij - (1.0 + nu_edge[p]) * alpha_edge[p] * dT_edge

            den = rho_edge[p]
            coef = (c_edge[p] * eff * aij) / den

            a_r += coef * dir_r_edge[p] * mu_edge[p]
            a_z += coef * dir_z_edge[p] * mu_edge[p]

        rden = rho_node[i]
        Ar[i] = a_r + br[i] / rden
        Az[i] = a_z + bz[i] / rden

    return Ar, Az

def compute_mechanical_step_csr(
    indptr, indices, dist, area,
    dx0_edge, dz0_edge,
    dir_r_edge, dir_z_edge,
    c_edge, nu_edge, alpha_edge,
    mu_edge, s0_edge,
    rho_edge, rho_node,
    Ur, Uz, br, bz,
    T_m, T_prev,
    Fr_prev, Fz_prev,
    Vr_half, Vz_half,
    lambda_diag, dt_ADR,step
):
    # 1) 加速度（mu_edge 在内核里原地更新）
    Ar, Az = compute_accel_csr_numba_inplace_mu(
        indptr, indices,
        dist, area,
        dx0_edge, dz0_edge,
        dir_r_edge, dir_z_edge,
        c_edge, nu_edge, alpha_edge,
        mu_edge, s0_edge,
        rho_edge,
        rho_node,
        Ur, Uz,
        br, bz,
        T_m, T_prev,step
    )

    # 2) 当前内力/等效力（与原 ADR 接口一致）
    Fr_curr = Ar * rho_node
    Fz_curr = Az * rho_node

    # 3) Numba 计算 cn（标量）
    cr_n = ADR.compute_local_damping_coefficient_numba(Fr_curr, Fr_prev, Vr_half, lambda_diag, Ur, dt_ADR)
    cz_n = ADR.compute_local_damping_coefficient_numba(Fz_curr, Fz_prev, Vz_half, lambda_diag, Uz, dt_ADR)

    # 4) 更新 F_prev（下一步用）
    Fr_prev = Fr_curr
    Fz_prev = Fz_curr

    # 5) ADR 更新半步速度与位移（你的原函数）
    Vr_half, Ur = ADR.adr_update_velocity_displacement(Ur, Vr_half, Fr_curr, cr_n, lambda_diag, dt_ADR)
    Vz_half, Uz = ADR.adr_update_velocity_displacement(Uz, Vz_half, Fz_curr, cz_n, lambda_diag, dt_ADR)

    return Ur, Uz, Fr_prev, Fz_prev, Vr_half, Vz_half

