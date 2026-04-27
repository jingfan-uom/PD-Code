
import numpy as np
from numba import njit, prange

def compute_shape_factor_edge_csr(
    r_node: np.ndarray,      # (N,)
    edge_i: np.ndarray,      # (nnz,)
    indices: np.ndarray,     # (nnz,)
    mechanical: bool,
    eps: float = 1e-30
) -> np.ndarray:
    ri = r_node[edge_i].astype(np.float64)
    rj = r_node[indices].astype(np.float64)

    out = np.empty(indices.size, dtype=np.float64)

    if mechanical:
        # rj/ri
        out[:] = rj/ri
    else:
        # 2*rj/(ri+rj)
        out[:] = 2*rj/(ri+rj)

    return out

# Convert temperature to enthalpy

def get_enthalpy(
    Tarr,
    mask_core,

    rho_s, rho_l, rho_shell,
    Cs, Cl,
    Cp_shell,          # âœ… å¸¸æ•°
    L, Ts, Tl,
    rho_void,     # âœ… å¯é€‰
    Cp_void       # âœ… å¯é€‰
):
    Tarr = np.asarray(Tarr)
    mask_core = np.asarray(mask_core, dtype=bool)

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


    # ---------- Shell region ----------
    # âœ… å…³é”®ä¿®æ­£ï¼šshell = éžcore ä¸” éžvoid
    mask_shell = ~(mask_core)
    if np.any(mask_shell):
        H[mask_shell] = rho_shell * Cp_shell * Tarr[mask_shell]

    return H


def get_density(
    Tarr, mask_core,
    rho_s, rho_l, Ts, Tl,
    rho_shell, rho_void,
    edge_i, edge_j,
    eps=1e-30
):
    """
    è¿”å›žï¼š
      rho_node: (N,) æ¯ä¸ªèŠ‚ç‚¹å¯†åº¦
      rho_edge: (nnz,) æ¯æ¡ CSR è¾¹çš„å¹³å‡å¯†åº¦ï¼ˆå¯é€‰ï¼›ä¼ å…¥ edge_i/edge_j æ—¶è¿”å›žï¼Œå¦åˆ™ä¸º Noneï¼‰

    è¯´æ˜Žï¼š
      - coreï¼šå›º/ç›¸å˜/æ¶² ç›¸å˜åŒºçº¿æ€§æ··åˆ rho_s ä¸Ž rho_l
      - voidï¼šrho_void
      - shellï¼šrho_shellï¼ˆæ³¨æ„ï¼šå¦‚æžœ void ä¸æ˜¯ core çš„å­é›†ï¼Œshell éœ€æŽ’é™¤ voidï¼‰
    """
    Tarr = np.asarray(Tarr)
    mask_core = np.asarray(mask_core, dtype=bool)

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
                # å…œåº•ï¼šä¸å…è®¸ Ts==Tl
                rho_c[mask_phase] = rho_s
            else:
                alpha = (Tc[mask_phase] - Ts) / dT
                rho_c[mask_phase] = (1.0 - alpha) * rho_s + alpha * rho_l

        rho_node[mask_core] = rho_c

    # --- void overwrite (highest priority) ---

    rho_edge = 0.5 * (rho_node[edge_i] + rho_node[edge_j])

    return rho_node, rho_edge


def compute_dt_cr_th_solid_with_csr(
    rho_solid: float,
    c_solid: float,
    k_solid: float,
    indptr: np.ndarray,          # (N+1,) int64
    dist: np.ndarray,            # (nnz,) float64
    area: np.ndarray,            # (nnz,) float64   (partial_area_flat)
    delta: float,
    i: int | None = None,
    eps: float = 1e-15
) -> float:
    """
    CSR ç‰ˆæœ¬ï¼šè®¡ç®—æŸä¸ªç²’å­ i çš„ä¸´ç•Œçƒ­æ—¶é—´æ­¥é•¿ Î”t_cr^TH

    å¯¹åº”åŽŸ dense ç‰ˆï¼š
      denominator = sum_j (k_eff / d_ij^2) * area_ij   (ä»… horizon å†…ä¸”æŽ’é™¤ self)

    è¿™é‡Œï¼š
      d_ij -> dist[p]
      area_ij -> area[p]
      p âˆˆ [indptr[i], indptr[i+1])
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

    # ä½ çš„åŽŸå¼ï¼šk = k_solid * 4/(pi*delta^2)
    k_eff = k_solid * 4.0 / (np.pi * delta * delta)

    # å®‰å…¨ï¼šæŽ’é™¤ dij=0 æˆ– area=0ï¼ˆarea==0 å¯ä»¥ä¸æŽ’é™¤ä¹Ÿæ²¡å½±å“ï¼Œä½†èƒ½å°‘ç®—ç‚¹ï¼‰
    mask = (dij > eps) & (aij != 0.0)
    if not np.any(mask):
        return np.inf

    denominator = np.sum((k_eff / (dij[mask] * dij[mask])) * aij[mask])
    if denominator == 0.0:
        return np.inf

    return (rho_solid * c_solid) / denominator

@njit(fastmath=True)
def _thermal_lambda_node(
    T, is_core,
    k_s, k_l, Ts, Tl,
    k_shell,           # k_all[i]
    delta,
    k_void
):
    # base_factor = 4/(pi*delta^2)
    base_factor = 4.0 / (np.pi * delta * delta)



    # coreï¼šç›¸å˜æ’å€¼
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
    Tarr, mask_core,
    k_s, k_l, Ts, Tl,
    k_shell, delta,
    k_void,
    indptr, indices,
):
    """
    è¿”å›žï¼š
      lambda_node: (N,)
      cond_data:   (nnz,)  harmonic mean of lambda_node[i], lambda_node[j]
    """
    N = Tarr.size
    nnz = indices.size

    lambda_node = np.empty(N, dtype=np.float64)
    for i in prange(N):
        lambda_node[i] = _thermal_lambda_node(
            Tarr[i], mask_core[i],
            k_s, k_l, Ts, Tl,
            k_shell, delta, k_void
        )

    cond_data = np.empty(nnz, dtype=np.float64)

    # éœ€è¦ edge_i
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
    Tarr, mask_core,
    factor_data, area_data, shape_factor_data, dist,
    indptr, indices,
    k_s, k_l, Ts, Tl,
    k_shell, delta,
    k_void,
    dt,
):
    """
    è¿”å›žï¼š
      K_data:  (nnz,) ä»… off-diagonal çš„æ­£å€¼ w_ijï¼ˆæŒ‰ CSR é¡ºåºï¼‰
      row_sum: (N,)   å¯¹è§’é¡¹ Kii = -sum_{j!=i} w_ij ï¼ˆè´Ÿæ•°ï¼‰
    """
    N = Tarr.size
    nnz = indices.size

    # cond_data[p] = harmonic mean of lambda_i, lambda_j
    cond_data = compute_thermal_conductivity_data_csr_numba(
        Tarr, mask_core,
        k_s, k_l, Ts, Tl,
        k_shell, delta, k_void,
        indptr, indices,
    )

    K_data = np.zeros(nnz, dtype=np.float64)
    row_sum = np.zeros(N, dtype=np.float64)  # å­˜ Kiiï¼ˆè´Ÿï¼‰

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

        # å¯¹è§’é¡¹ï¼ˆè´Ÿå·ï¼‰ï¼Œä¸Ž dense ç‰ˆ fill_diagonal ä¸€è‡´
        row_sum[i] = -s_pos

    return K_data, row_sum

@njit(parallel=True, fastmath=True)
def apply_K_with_diag_csr_numba(
    indptr, indices, K_data, diag, Tarr,
    rho_void_old, Cp_void_old,
    rho_void_new, Cp_void_new
):
    N = Tarr.size
    out = np.zeros(N, dtype=np.float64)

    M_old = rho_void_old * Cp_void_old
    M_new = rho_void_new * Cp_void_new
    dM = M_new - M_old

    for i in prange(N):
        s = diag[i] * Tarr[i]
        a = indptr[i]
        b = indptr[i + 1]

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
    mask_core,
    rho_s, rho_l, cs, cl, L, Ts, Tl,
    rho_shell, c_shell,
    rho_void, Cp_void,
    eps=1e-30
):
    """
    å®Œå…¨ numba ç‰ˆ get_temperatureï¼ˆæ—  boolean slicingï¼‰
    """
    N = Harr.size
    T = np.empty(N, dtype=np.float64)

    H_solid_max = rho_s * cs * Ts
    H_liquid_min = H_solid_max + rho_s * L
    dT = Tl - Ts

    for i in prange(N):
        H = Harr[i]


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


