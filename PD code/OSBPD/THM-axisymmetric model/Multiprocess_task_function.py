
import numpy as np
from numba import njit, prange
import ADR
import Physical_Field_Calculation as pfc

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

from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_accel_osbpd_axisym_csr_numba(
    indptr, indices, area_edge,
    dist_edge,               # (nnz,) |x_ij|
    eij_edge,                # (nnz,) nlen - dist
    n_r_edge, n_z_edge,      # (nnz,)
    lamda_node,              # (N,)
    miu_node,                # (N,)
    r_node,                  # (N,)
    Ur,                      # (N,)
    dilation,                # (N,)
    rho_node,                # (N,)
    br, bz,                  # (N,)
    delta,                   # float64
    T_m, Tpre_avg, kprime_node_m, csr,
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

        # ✅ i端参数全部用i点的
        lam_i = lamda_node[i]
        mu_i  = miu_node[i]
        kp_i  = kprime_node_m[i]

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

            # ✅ j端参数全部用j点的
            lam_j = lamda_node[j]
            mu_j  = miu_node[j]
            kp_j  = kprime_node_m[j]

            # strain/stretch on this bond (bond-level geom)
            strain = eij_edge[p] / (dist_edge[p] + eps)

            # ---------------------------------------------------
            lam_avg = 0.5 * (lam_i + lam_j)
            mu_avg = 0.5 * (mu_i + mu_j)
            kp_avg = 0.5 * (kp_i + kp_j)

            ti = lam_avg * theta_i + (lam_avg + mu_avg) * (Uri * inv_ri) -  kp_avg * dTi
            tj = lam_avg * theta_j + (lam_avg + mu_avg) * (Urj * inv_rj) -  kp_avg * dTj

            base_i = coeff1 * ti + coeff2 * mu_avg * strain
            base_j = coeff1 * tj + coeff2 * mu_avg * strain

            # ---------------------------------------------------
            # 2) i端/j端分别形成力幅值（各自乘自己的shape）
            # i端shape = rj/ri, j端shape = ri/rj
            # ---------------------------------------------------
            s_i = 0.0
            if ri > eps:
                s_i = rj / ri

            s_j = 0.0
            if rj > eps:
                s_j = ri / rj

            dforce_i = area_edge[p] * base_i * s_i * csr[p]
            dforce_j = area_edge[p] * base_j * s_j * csr[p]

            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce
            Fi_z += n_z_edge[p] * dforce

        # ---------------------------------------------------
        # 3) 轴对称体力项 b_r（node-wise，只用i点参数和值）
        # 你原来是 -2*inv_ri*(...)：我保留不动
        # ---------------------------------------------------
        br_geom = -1.0 * inv_ri * (
            theta_i * (lam_i + mu_i) +
            (lam_i + 3.0 * mu_i) * Uri * inv_ri
            - kp_i * dTi * 1.25
        )

        inv_rho = 1.0 / (rho_node[i] + eps)
        Ar[i] = (Fi_r + br_geom + br[i]) * inv_rho
        Az[i] = (Fi_z + bz[i]) * inv_rho

    return Ar, Az


from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def compute_accel_osbpd_axisym_csr_inplace_mu(
    indptr, indices, area_edge,
    shape_edge,                 # (nnz,)  对 i 端：rj/ri
    dist_edge,                  # (nnz,)
    eij_edge,                   # (nnz,)
    n_r_edge, n_z_edge,         # (nnz,)
    crack, s0_edge,           # (nnz,)
    lamda_node, miu_node,       # (N,)
    r_node,                     # (N,)
    Ur,                         # (N,)
    dilation,                   # (N,)
    rho_node,                   # (N,)
    br, bz,                     # (N,)
    delta,                      # scalar
    cracking_act, T_m, Tpre_avg, kprime_node_m, csr,
    eps=1e-30
):
    N = Ur.size
    Ar = np.zeros(N, dtype=np.float64)
    Az = np.zeros(N, dtype=np.float64)

    coeff1 = 3.0 / (np.pi * (delta ** 3))
    coeff2 = 12.0 / (np.pi * (delta ** 3))

    for i in prange(N):
        ri = r_node[i]
        inv_ri = 1.0 / ri

        Fi_r = 0.0
        Fi_z = 0.0

        theta_i = dilation[i]
        Uri = Ur[i]
        dTi = T_m[i] - Tpre_avg

        # ✅ i端材料参数（全部用 i 的）
        lam_i = lamda_node[i]
        mu_i  = miu_node[i]
        kp_i  = kprime_node_m[i]

        a0 = indptr[i]
        b0 = indptr[i + 1]

        for p in range(a0, b0):
            j = indices[p]
            if j == i:
                continue
            if crack[p] == 0:
                continue

            stretch = eij_edge[p] / (dist_edge[p] + eps)


            # ---- j node fields ----
            rj = r_node[j]
            inv_rj = 1.0 / rj

            theta_j = dilation[j]
            Urj = Ur[j]
            dTj = T_m[j] - Tpre_avg

            if (stretch - 4e-6 * (dTi + dTj)/2 > s0_edge[p]) & cracking_act:
                crack[p] = 0
                continue

            # ✅ j端材料参数（全部用 j 的）
            lam_j = lamda_node[j]
            mu_j  = miu_node[j]
            kp_j  = kprime_node_m[j]

            # -------------------------------
            lam_avg = 0.5 * (lam_i + lam_j)
            mu_avg = 0.5 * (mu_i + mu_j)
            kp_avg = 0.5 * (kp_i + kp_j)

            ti = lam_avg * theta_i + (lam_avg + mu_avg) * (Uri * inv_ri) - kp_avg * dTi
            tj = lam_avg * theta_j + (lam_avg + mu_avg) * (Urj * inv_rj) - kp_avg * dTj

            base_i = coeff1 * ti + coeff2 * mu_avg * stretch
            base_j = coeff1 * tj + coeff2 * mu_avg * stretch

            # -------------------------------
            s_i = shape_edge[p]                       # rj/ri (for i-equation)
            s_j = 0.0
            if s_i > eps or s_i < -eps:
                s_j = 1.0 / s_i                       # ri/rj (for j-equation)

            dforce_i = area_edge[p] * base_i * s_i * csr[p] * crack[p]
            dforce_j = area_edge[p] * base_j * s_j * csr[p] * crack[p]

            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce * 0.5
            Fi_z += n_z_edge[p] * dforce * 0.5

        # node-wise geometric term（仍然是 i 的）
        br_geom = -inv_ri * (
            theta_i * (lam_i + mu_i) +
            (lam_i + 3.0 * mu_i) * Uri * inv_ri
            - kp_i * dTi
        )

        inv_rho = 1.0 / rho_node[i]
        Ar[i] = (Fi_r + br_geom + br[i]) * inv_rho
        Az[i] = (Fi_z + bz[i]) * inv_rho

    return Ar, Az

def compute_mechanical_step_csr(
    indptr, indices, dist, area,
    r_flat, z_flat, edge_i,
    shape_edge, coeff_dilation,
    lamda_node, miu_node,
    crack, s0_edge,
    rho_node,
    Ur, Uz, br, bz,               # 先保留接口（你后面要加热膨胀耦合时用）
    Fr_prev, Fz_prev,
    Vr_half, Vz_half,
    lambda_diag, dt_ADR,
    delta, cracking_act, T_m, T_prev, kprime_node_m, csr
):
    # 1) dilation + eij + n （一次 nnz 遍历）
    dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr(
        r_flat, z_flat, Ur, Uz,
        edge_i, indices,
        dist,
        area,
        shape_edge,
        coeff_dilation
    )

    # 2) 加速度（mu_edge 在内核里原地更新）
    Ar, Az = compute_accel_osbpd_axisym_csr_inplace_mu(
        indptr, indices, area,
        shape_edge,
        dist,
        eij_edge,
        n_r_edge, n_z_edge,
        crack, s0_edge,
        lamda_node, miu_node,
        r_flat,
        Ur,
        dilation,
        rho_node,
        br, bz,
        delta,
        cracking_act, T_m, T_prev, kprime_node_m, csr
    )

    # 3) 当前内力（与原 ADR 接口一致）
    Fr_curr = Ar * rho_node
    Fz_curr = Az * rho_node

    # 4) Numba 计算 cn（标量）
    cr_n = ADR.compute_local_damping_coefficient_numba(Fr_curr, Fr_prev, Vr_half, lambda_diag, Ur, dt_ADR)
    cz_n = ADR.compute_local_damping_coefficient_numba(Fz_curr, Fz_prev, Vz_half, lambda_diag, Uz, dt_ADR)

    # 5) 更新 F_prev
    Fr_prev = Fr_curr
    Fz_prev = Fz_curr

    # 6) ADR 更新半步速度与位移
    Vr_half, Ur = ADR.adr_update_velocity_displacement(Ur, Vr_half, Fr_curr, cr_n, lambda_diag, dt_ADR)
    Vz_half, Uz = ADR.adr_update_velocity_displacement(Uz, Vz_half, Fz_curr, cz_n, lambda_diag, dt_ADR)

    return Ur, Uz, Fr_prev, Fz_prev, Vr_half, Vz_half


