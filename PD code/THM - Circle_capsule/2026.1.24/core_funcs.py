

import numpy as np
from numba import njit, prange
def compute_shape_factor_data_csr(
    r_node: np.ndarray,          # (N,) 每个点的 r 坐标，例如 coords[:,0]
    indptr: np.ndarray,          # (N+1,)
    indices: np.ndarray,         # (nnz,)
    eps: float = 1e-15
) -> np.ndarray:

    N = indptr.size - 1
    nnz = indices.size
    out = np.empty(nnz, dtype=np.float64)

    # 每条边对应的行号 i
    edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))

    ri = r_node[edge_i]
    rj = r_node[indices]
    denom = ri + rj

    # 防止除零
    mask = np.abs(denom) > eps
    out[mask] = 1 #2.0 * rj[mask] / denom[mask]
    out[~mask] = 1.0
    return out

# Convert temperature to enthalpy

def get_enthalpy(
    Tarr,
    mask_core,
    mask_void,
    rho_s, rho_l, rho_shell,
    Cs, Cl,
    Cp_shell,          # ✅ 常数
    L, Ts, Tl,
    rho_void,     # ✅ 可选
    Cp_void       # ✅ 可选
):
    Tarr = np.asarray(Tarr)
    mask_core = np.asarray(mask_core, dtype=bool)
    mask_void = np.asarray(mask_void, dtype=bool)

    H = np.zeros_like(Tarr, dtype=np.float64)

    # ---------- Core region (with phase change) ----------
    if np.any(mask_core):
        T_core = Tarr[mask_core]
        Hc = np.zeros_like(T_core, dtype=np.float64)

        # Solid
        mask_solid = T_core < Ts
        Hc[mask_solid] = rho_s * Cs * T_core[mask_solid]

        # Phase change
        mask_phase = (T_core >= Ts) & (T_core <= Tl)
        if np.any(mask_phase):
            dT = (Tl - Ts)
            alpha = (T_core[mask_phase] - Ts) / dT
            rho_mix = (1.0 - alpha) * rho_s + alpha * rho_l
            Hc[mask_phase] = rho_mix * (Cs * Ts + alpha * L)

        # Liquid
        mask_liquid = T_core > Tl
        if np.any(mask_liquid):
            Hc[mask_liquid] = rho_l * (Cs * Ts + L + Cl * (T_core[mask_liquid] - Tl))

        H[mask_core] = Hc

        H[mask_void] = rho_void * Cp_void * Tarr[mask_void]

    # ---------- Shell region ----------
    # ✅ 关键修正：shell = 非core 且 非void
    mask_shell = ~(mask_core | mask_void)
    if np.any(mask_shell):
        H[mask_shell] = rho_shell * Cp_shell * Tarr[mask_shell]

    return H


def get_density(
    Tarr, mask_core, mask_void,
    rho_s, rho_l, Ts, Tl,
    rho_shell, rho_void,
    edge_i, edge_j,
    eps=1e-30
):
    """
    返回：
      rho_node: (N,) 每个节点密度
      rho_edge: (nnz,) 每条 CSR 边的平均密度（可选；传入 edge_i/edge_j 时返回，否则为 None）

    说明：
      - core：固/相变/液 相变区线性混合 rho_s 与 rho_l
      - void：rho_void
      - shell：rho_shell（注意：如果 void 不是 core 的子集，shell 需排除 void）
    """
    Tarr = np.asarray(Tarr)
    mask_core = np.asarray(mask_core, dtype=bool)
    mask_void = np.asarray(mask_void, dtype=bool)

    N = Tarr.size
    rho_node = np.empty(N, dtype=np.float64)

    # --- default: shell ---
    rho_node[:] = rho_shell

    # --- core overwrite (excluding void is optional; usually void is subset of core) ---
    if np.any(mask_core):
        Tc = Tarr[mask_core]
        rho_c = np.empty_like(Tc, dtype=np.float64)

        mask_solid = Tc < Ts
        mask_liquid = Tc > Tl
        mask_phase  = (~mask_solid) & (~mask_liquid)  # Ts <= T <= Tl

        rho_c[mask_solid] = rho_s
        rho_c[mask_liquid] = rho_l
        if np.any(mask_phase):
            dT = Tl - Ts
            if dT <= 0:
                # 兜底：不允许 Ts==Tl
                rho_c[mask_phase] = rho_s
            else:
                alpha = (Tc[mask_phase] - Ts) / dT
                rho_c[mask_phase] = (1.0 - alpha) * rho_s + alpha * rho_l

        rho_node[mask_core] = rho_c

    # --- void overwrite (highest priority) ---
    if np.any(mask_void):
        rho_node[mask_void] = rho_void

    # --- optional edge-averaged density ---
    rho_edge = None
    if edge_i is not None and edge_j is not None:
        edge_i = np.asarray(edge_i, dtype=np.int64)
        edge_j = np.asarray(edge_j, dtype=np.int64)
        rho_edge = 0.5 * (rho_node[edge_i] + rho_node[edge_j])
        rho_edge = np.maximum(rho_edge, eps)  # 防止除零（尤其 rho_air=0 时）

    return rho_node, rho_edge


def compute_dt_cr_th_solid_with_csr(
    rho_solid: float,
    c_solid: float,
    k_solid: float,
    indptr: np.ndarray,          # (N+1,) int32/int64
    dist: np.ndarray,            # (nnz,) float64
    area: np.ndarray,            # (nnz,) float64   (partial_area_flat)
    delta: float,
    i: int | None = None,
    eps: float = 1e-15
) -> float:
    """
    CSR 版本：计算某个粒子 i 的临界热时间步长 Δt_cr^TH

    对应原 dense 版：
      denominator = sum_j (k_eff / d_ij^2) * area_ij   (仅 horizon 内且排除 self)

    这里：
      d_ij -> dist[p]
      area_ij -> area[p]
      p ∈ [indptr[i], indptr[i+1])
    """
    N = indptr.size - 1
    if i is None:
        i = N // 2

    a = indptr[i]
    b = indptr[i + 1]
    if a == b:
        return np.inf

    dij = dist[a:b]
    aij = area[a:b]

    # 你的原式：k = k_solid * 4/(pi*delta^2)
    k_eff = k_solid * 4.0 / (np.pi * delta * delta)

    # 安全：排除 dij=0 或 area=0（area==0 可以不排除也没影响，但能少算点）
    mask = (dij > eps) & (aij != 0.0)
    if not np.any(mask):
        return np.inf

    denominator = np.sum((k_eff / (dij[mask] * dij[mask])) * aij[mask])
    if denominator == 0.0:
        return np.inf

    return (rho_solid * c_solid) / denominator

@njit(fastmath=True)
def _thermal_lambda_node(
    T, is_core, is_void,
    k_s, k_l, Ts, Tl,
    k_shell,           # k_all[i]
    delta,
    k_void
):
    # base_factor = 4/(pi*delta^2)
    base_factor = 4.0 / (np.pi * delta * delta)

    # void 优先级最高
    if is_void:
        return k_void * base_factor

    # core：相变插值
    if is_core:
        if T < Ts:
            return k_s * base_factor
        elif T > Tl:
            return k_l * base_factor
        else:
            alpha = (T - Ts) / (Tl - Ts)
            return ((1.0 - alpha) * k_s + alpha * k_l) * base_factor

    # shell
    return k_shell * base_factor


@njit(parallel=True, fastmath=True)
def compute_thermal_conductivity_data_csr_numba(
    Tarr, mask_core, mask_void,
    k_s, k_l, Ts, Tl,
    k_shell, delta,
    k_void,
    indptr, indices,
):
    """
    返回：
      lambda_node: (N,)
      cond_data:   (nnz,)  harmonic mean of lambda_node[i], lambda_node[j]
    """
    N = Tarr.size
    nnz = indices.size

    lambda_node = np.empty(N, dtype=np.float64)
    for i in prange(N):
        lambda_node[i] = _thermal_lambda_node(
            Tarr[i], mask_core[i], mask_void[i],
            k_s, k_l, Ts, Tl,
            k_shell, delta, k_void
        )

    cond_data = np.empty(nnz, dtype=np.float64)

    # 需要 edge_i
    edge_i = np.empty(nnz, dtype=np.int64)
    pos = 0
    for i in range(N):
        a = indptr[i]
        b = indptr[i + 1]
        for p in range(a, b):
            edge_i[pos] = i
            pos += 1

    for p in prange(nnz):
        i = edge_i[p]
        j = indices[p]
        li = lambda_node[i]
        lj = lambda_node[j]
        den = li + lj
        cond_data[p] = 2.0 * li * lj / den

    return cond_data

@njit(parallel=True, fastmath=True)
def build_Kdata_and_rowsum_csr_numba(
    Tarr, mask_core, mask_void,
    factor_data, area_data, shape_factor_data, dist,
    indptr, indices,
    k_s, k_l, Ts, Tl,
    k_shell, delta,
    k_void,
    dt,
):
    """
    返回：
      K_data:  (nnz,) 仅 off-diagonal 的正值 w_ij（按 CSR 顺序）
      row_sum: (N,)   对角项 Kii = -sum_{j!=i} w_ij （负数）
    """
    N = Tarr.size
    nnz = indices.size

    # cond_data[p] = harmonic mean of lambda_i, lambda_j
    cond_data = compute_thermal_conductivity_data_csr_numba(
        Tarr, mask_core, mask_void,
        k_s, k_l, Ts, Tl,
        k_shell, delta, k_void,
        indptr, indices,
    )

    K_data = np.zeros(nnz, dtype=np.float64)
    row_sum = np.zeros(N, dtype=np.float64)  # 存 Kii（负）

    for i in prange(N):
        a = indptr[i]
        b = indptr[i + 1]
        s_pos = 0.0  # sum of off-diagonal positive weights

        for p in range(a, b):
            j = indices[p]
            if j == i:
                continue
            dij = dist[p]
            aij = area_data[p]

            w = (factor_data[p] * aij * shape_factor_data[p] *
                 cond_data[p] / (dij * dij) * dt)

            K_data[p] = w
            s_pos += w

        # 对角项（负号），与 dense 版 fill_diagonal 一致
        row_sum[i] = -s_pos

    return K_data, row_sum

@njit(parallel=True, fastmath=True)
def apply_K_with_diag_csr_numba(indptr, indices, K_data, diag, Tarr):
    N = Tarr.size
    out = np.zeros(N, dtype=np.float64)
    for i in prange(N):
        s = diag[i] * Tarr[i]
        a = indptr[i]
        b = indptr[i+1]
        for p in range(a, b):
            j = indices[p]
            if j == i:
                continue
            s += K_data[p] * Tarr[j]
        out[i] = s
    return out


@njit(parallel=True, fastmath=True)
def temperature_from_enthalpy_numba(
    Harr,
    mask_core, mask_void,
    rho_s, rho_l, cs, cl, L, Ts, Tl,
    rho_shell, c_shell,
    rho_void, Cp_void,
    eps=1e-30
):
    """
    完全 numba 版 get_temperature（无 boolean slicing）
    """
    N = Harr.size
    T = np.empty(N, dtype=np.float64)

    H_solid_max = rho_s * cs * Ts
    H_liquid_min = H_solid_max + rho_s * L
    dT = Tl - Ts

    for i in prange(N):
        H = Harr[i]

        if mask_void[i]:
            den = rho_void * Cp_void
            if den < eps:
                T[i] = Ts  # 兜底
            else:
                T[i] = H / den
            continue

        if mask_core[i]:
            if H <= H_solid_max:
                T[i] = H / (rho_s * cs)
            elif H <= H_liquid_min:
                alpha = (H - H_solid_max) / (rho_s * L)
                T[i] = Ts + alpha * dT
            else:
                H_adj = H - (H_solid_max + rho_s * L)
                T[i] = H_adj / (rho_l * cl) + Tl
            continue

        # shell
        den = rho_shell * c_shell
        if den < eps:
            T[i] = Ts
        else:
            T[i] = H / den

    return T
