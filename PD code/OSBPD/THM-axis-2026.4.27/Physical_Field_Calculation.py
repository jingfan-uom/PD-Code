import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_dilation_axisym_csr(
    r_flat, z_flat, Ur, Uz,
    edge_i, indices,
    dist_edge,
    area_edge,
    shape_edge,
    coeff, csr,         # 标量
    eps=1e-30
):
    N = r_flat.shape[0]
    nnz = indices.shape[0]

    dilation = np.zeros(N, dtype=np.float64)
    eij_edge = np.empty(nnz, dtype=np.float64)
    n_r_edge = np.empty(nnz, dtype=np.float64)
    n_z_edge = np.empty(nnz, dtype=np.float64)

    for k in range(nnz):

        i = edge_i[k]
        j = indices[k]
        if j == i:
            eij_edge[k] = 0.0
            n_r_edge[k] = 0.0
            n_z_edge[k] = 0.0
            continue

        dx_r = r_flat[j] - r_flat[i]
        dx_z = z_flat[j] - z_flat[i]
        dist = dist_edge[k] + eps  # L0

        du_r = Ur[j] - Ur[i]
        du_z = Uz[j] - Uz[i]

        # exact bond strain: eij = (L1 - L0) / L0
        dx1_r = dx_r + du_r
        dx1_z = dx_z + du_z
        L1 = np.sqrt(dx1_r * dx1_r + dx1_z * dx1_z)

        eij = (L1 - dist)
        eij_edge[k] = eij / dist

        inv = 1.0 / dist
        n_r_edge[k] = dx1_r * inv
        n_z_edge[k] = dx1_z * inv

        dilation[i] += eij * area_edge[k] * coeff * shape_edge[k]

    return dilation, eij_edge, n_r_edge, n_z_edge


@njit(parallel=True, fastmath=True)
def compute_dilation_axisym_csr_weighted(
    r_flat, z_flat, Ur, Uz,
    edge_i, indices,
    dist_edge,
    area_edge,
    shape_edge,
    coeff,
    csr,
    use_csr_in_dilation,
    eps=1e-30
):
    N = r_flat.shape[0]
    nnz = indices.shape[0]

    dilation = np.zeros(N, dtype=np.float64)
    eij_edge = np.empty(nnz, dtype=np.float64)
    n_r_edge = np.empty(nnz, dtype=np.float64)
    n_z_edge = np.empty(nnz, dtype=np.float64)

    for k in range(nnz):
        i = edge_i[k]
        j = indices[k]
        if j == i:
            eij_edge[k] = 0.0
            n_r_edge[k] = 0.0
            n_z_edge[k] = 0.0
            continue

        dx_r = r_flat[j] - r_flat[i]
        dx_z = z_flat[j] - z_flat[i]
        dist = dist_edge[k] + eps

        du_r = Ur[j] - Ur[i]
        du_z = Uz[j] - Uz[i]

        dx1_r = dx_r + du_r
        dx1_z = dx_z + du_z
        L1 = np.sqrt(dx1_r * dx1_r + dx1_z * dx1_z)

        eij = (L1 - dist)
        eij_edge[k] = eij / dist

        inv = 1.0 / dist
        n_r_edge[k] = dx1_r * inv
        n_z_edge[k] = dx1_z * inv

        w = area_edge[k] * coeff * shape_edge[k]
        if use_csr_in_dilation:
            w *= csr[k]
        dilation[i] += eij * w

    return dilation, eij_edge, n_r_edge, n_z_edge


def compute_direction_edges_csr_numba(coords, edge_i, edge_j, dist, eps=1e-30):
    Nnz = dist.size
    dir_r = np.zeros(Nnz, dtype=np.float64)
    dir_z = np.zeros(Nnz, dtype=np.float64)

    x = coords[:, 0]
    z = coords[:, 1]

    for p in prange(Nnz):
        i = edge_i[p]
        j = edge_j[p]
        dij = dist[p]
        if dij > eps and i != j:
            dir_r[p] = (x[j] - x[i]) / dij
            dir_z[p] = (z[j] - z[i]) / dij
        else:
            dir_r[p] = 0.0
            dir_z[p] = 0.0

    return dir_r, dir_z


def compute_Tensor_product(x, y, horizon_mask):

    # Current relative positions after deformation: x '- x
    dx_eff = x[None, :] - x[:, None]
    dz_eff = y[None, :] - y[:, None]
    dist_eff = np.sqrt(dx_eff**2 + dz_eff**2)
    # Only compute values where horizon_mask is True; set others to zero
    n_x = np.zeros_like(dx_eff)
    n_z = np.zeros_like(dz_eff)

    n_x[horizon_mask] = dx_eff[horizon_mask] / dist_eff[horizon_mask]
    n_z[horizon_mask] = dz_eff[horizon_mask] / dist_eff[horizon_mask]

    N = n_x.shape[0]
    Cxx = np.zeros((N, N))
    Cxz = np.zeros((N, N))
    Czx = np.zeros((N, N))
    Czz = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # 只对有效（horizon_mask为True）的键计算
            if horizon_mask[i, j]:
                n_vec = np.array([n_x[i, j], n_z[i, j]])  # 组成二维单位向量
                n_outer = np.outer(n_vec, n_vec)  # 2x2张量积

                Cxx[i, j] = n_outer[0, 0]
                Cxz[i, j] = n_outer[0, 1]
                Czx[i, j] = n_outer[1, 0]
                Czz[i, j] = n_outer[1, 1]

    return Cxx, Cxz, Czx, Czz

def compute_s_matrix(coords, Ux, Uz, horizon_mask):

    """Compute elongation matrix s_matrix (N, N) using vectorized matrix operations.

    Parameters:
        coords: (N, 2) array of original coordinates (x, y)
        Ux, Uz: displacement arrays (N,)
        horizon_mask: boolean array of shape (N, N), True if bond (i, j) is valid

    Returns:
        s_matrix: elongation matrix (N, N)"""

    # Original coordinates
    x_flat = coords[:, 0]
    y_flat = coords[:, 1]

    # Deformed coordinates
    x_def = x_flat + Ux
    y_def = y_flat + Uz

    # Initial lengths L0
    dx0 = x_flat[None, :] - x_flat[:, None]
    dz0 = y_flat[None, :] - y_flat[:, None]
    L0 = np.sqrt(dx0 ** 2 + dz0 ** 2)

    # Deformed lengths L1
    dx1 = x_def[None, :] - x_def[:, None]
    dz1 = y_def[None, :] - y_def[:, None]
    L1 = np.sqrt(dx1 ** 2 + dz1 ** 2)

    # Elongation computation (vectorized)
    s_matrix = np.zeros_like(L0)
    mask = horizon_mask & (L0 > 0)
    s_matrix[mask] = (L1[mask] - L0[mask]) / L0[mask]

    return s_matrix

def compute_delta_temperature(T_grid, Tpre_avg):

    """Compute the average temperature matrix and the difference from the previous step.

    Parameters:
        T_grid: current temperature field (1D or flattened)
        Tpre_avg: average temperature matrix from previous time step, shape (N, N)

    Returns:
        T_delta: difference between current and previous average temperature matrices
    """
    T_i = T_grid[:, np.newaxis]  # shape (N, 1)
    T_j = T_grid[np.newaxis, :]  # shape (1, N)
    Tcurr_avg = 0.5 * (T_i + T_j)- Tpre_avg  # shape (N, N)

    return Tcurr_avg


def shrink_Tth_by_matching_coords(coords_m, coords_t):

    coords_m = np.asarray(coords_m)
    coords_t = np.asarray(coords_t)

    # If there are three columns, take the first two columns to match
    if coords_m.shape[1] >= 3:
        coords_m = coords_m[:, :2]
    if coords_t.shape[1] >= 3:
        coords_t = coords_t[:, :2]

    # Nearest Neighbor Matching
    tree = cKDTree(coords_t)
    _, indices = tree.query(coords_m, k=1)

    return indices.astype(np.int64)



def filter_array_by_indices_keep_only(Tarr, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return np.ascontiguousarray(Tarr[indices])



def update_mu_by_failure(mu, Relative_elongation, s0):

    failure_mask = (mu == 1) & (Relative_elongation >= s0)
    mu_new = mu.copy()
    mu_new[failure_mask] = 0

    return mu_new

def find_inner_surface_layer(coords_phys_m, r, dshell, dr, r_center=0.0):
    coords = np.asarray(coords_phys_m)

    x_arr = coords[:, 0]
    z_arr = coords[:, 1]

    xc, zc = float(r_center), float(r)
    dist = np.sqrt((x_arr - xc)**2 + (z_arr - zc)**2)

    r_inner = float(r) - float(dshell)

    mask = (dist >= r_inner) & (dist <= r_inner + 3 * dr / 4)
    idx_layer = np.where(mask)[0]

    # 这里给的是“从圆心指向该点”的外法向
    vx = x_arr[idx_layer] - xc
    vz = z_arr[idx_layer] - zc
    norm = np.hypot(vx, vz)

    unit_outward = np.zeros((idx_layer.size, 2), dtype=coords.dtype)
    nz = norm > 1e-12
    unit_outward[nz, 0] = vx[nz] / norm[nz]
    unit_outward[nz, 1] = vz[nz] / norm[nz]

    return {
        "indices": idx_layer,
        "unit_outward": unit_outward
    }

def compute_melt_and_thermal_expansion(
    T,
    mask_core,
    rho_s, rho_l,
    Ts, Tl,
    cell_volume,     # 标量 或 (N,) 或 (N_core,)
    beta_s,
    beta_l,
    T_ref,
):
    T = np.asarray(T, dtype=float)
    core = np.asarray(mask_core, dtype=bool)

    T_core = T[core]

    # 把 cell_volume 对齐到 core 点
    cv = np.asarray(cell_volume, dtype=float)
    if cv.shape[0] == T.shape[0]:
        cv_core = cv[core]
    else:
        cv_core = cv  # 认为已经是 core 对齐的

    dT_phase = Tl - Ts
    alpha = (T_core - Ts) / dT_phase
    alpha = np.clip(alpha, 0.0, 1.0)

    # 等效已熔化体积
    V_melt_equiv = float(np.sum(alpha * cv_core))

    kappa = (rho_s / rho_l) - 1.0
    deltaV_phase = V_melt_equiv * kappa

    dT_vec = T_core - T_ref
    beta_eff = beta_s * (1.0 - alpha) + beta_l * alpha

    deltaV_thermal = float(np.sum(beta_eff * dT_vec * cv_core))

    return float(deltaV_phase), float(deltaV_thermal)



def void_z_3d(f_void: float, r: float, tol: float = 1e-12) -> float:
    """
    3D: 球体(半径 r)中，上部球冠体积分数 = f_void。
    坐标 z 从 0(底部) 到 2r(顶部)，返回界面高度 z0：
      - f_void = 0 -> z0 = 2r（没有空腔）
      - f_void = 1 -> z0 = 0  （空腔占满球体）
    """
    if f_void <= 0.0:
        return 2.0 * r
    if f_void >= 1.0:
        return 0.0

    # 解 x^2 (3 - x) = 4 f, 其中 x = h/r, h = 2r - z0
    target = 4.0 * f_void

    def G(x: float) -> float:
        return x * x * (3.0 - x) - target

    lo, hi = 0.0, 2.0  # x in [0,2]
    glo, ghi = G(lo), G(hi)
    # 理论上 glo=-target<0, ghi=4-target>0 一定异号
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gmid = G(mid)
        if (hi - lo) < tol:
            x = mid
            break
        if glo * gmid <= 0.0:
            hi = mid
            ghi = gmid
        else:
            lo = mid
            glo = gmid
    else:
        x = 0.5 * (lo + hi)

    h = x * r
    z0 = 2.0 * r - h
    return z0


def find_region_and_index(phys_coords_list_t, target=(0.0, 20e-6)):
    """
    在每个区域的物理点列表中找到与 target 最近的点。
    phys_coords_list_t[i] 形如 [r, z, region_id]。
    Returns: (region_id, index_in_region)
    """
    best_region, best_idx, best_d2 = None, None, np.inf
    tgt = np.asarray(target, dtype=float)

    for region_id, arr in enumerate(phys_coords_list_t):
        if arr.size == 0:
            continue
        coords = arr[:, :2]          # 只取 r,z
        d2 = np.sum((coords - tgt)**2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] < best_d2:
            best_region, best_idx, best_d2 = region_id, idx, float(d2[idx])

    if best_region is None:
        raise ValueError("phys_coords_list_t 为空或无点。")

    return best_region, best_idx


def map_Tth_to_mech(coords_m, coords_th, T_th, tol):
    pts_m  = np.ascontiguousarray(coords_m[:, :2])
    pts_th = np.ascontiguousarray(coords_th[:, :2])

    tree = cKDTree(pts_th)
    dist, idx = tree.query(pts_m, k=1)

    # 强制容差，否则会“乱配”
    bad = dist > tol
    if np.any(bad):
        raise ValueError(
            f"KDTree mapping failed: {bad.sum()} / {len(dist)} points exceed tol. "
            f"max_dist={dist.max():.3e}, tol={tol:.3e}"
        )

    # 关键：按 mech 顺序直接重排
    return T_th[idx]


def temperature_sphere_dirichlet(rho, t, a, kappa, Tinit, Tsurr, n_terms=200):
    """
    3D 实心球，初始均匀温度 Tinit，表面恒温 Tsurr
    返回温度理论解 T(rho, t)
    rho 可为标量或数组
    """
    rho = np.asarray(rho, dtype=float)
    T = np.empty_like(rho)

    n = np.arange(1, n_terms + 1, dtype=float)
    decay = np.exp(-(n**2) * (np.pi**2) * kappa * t / (a**2))

    mask = rho > 1e-14

    if np.any(mask):
        rr = rho[mask][:, None]
        series = np.sum(
            ((-1.0)**(n + 1) / n) * np.sin(n * np.pi * rr / a) * decay,
            axis=1
        )
        ratio = (2.0 * a / (np.pi * rho[mask])) * series
        T[mask] = Tsurr + (Tinit - Tsurr) * ratio

    # rho = 0 处取极限
    if np.any(~mask):
        series0 = np.sum(((-1.0)**(n + 1)) * decay)
        ratio0 = 2.0 * series0
        T[~mask] = Tsurr + (Tinit - Tsurr) * ratio0

    return T

def radial_disp_theory_sphere(rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
                              n_terms=200, n_r_int=2000):
    """
    3D 实心球在自由边界条件下的球对称径向位移理论解 u(rho_p, t)
    """
    if rho_p < 1e-14:
        return 0.0

    rho_grid = np.linspace(0.0, a, n_r_int)
    T_grid = temperature_sphere_dirichlet(
        rho_grid, t, a, kappa, Tinit, Tsurr, n_terms=n_terms
    )
    dT_grid = T_grid - Tinit

    Ia = np.trapz(dT_grid * rho_grid ** 2, rho_grid)


    mask = rho_grid <= rho_p
    rho_sub = rho_grid[mask]
    dT_sub = dT_grid[mask]
    Ir = np.trapz(dT_sub * rho_sub ** 2, rho_sub)

    u = (alpha / (1.0 - nu)) * (
        (1.0 + nu) * Ir / (rho_p**2)
        + 2.0 * (1.0 - 2.0 * nu) * rho_p * Ia / (a**3)
    )
    return u

def horizontal_disp_theory_at_point(r_pt, z_pt, t, a, center_r, center_z,
                                    kappa, alpha, nu, Tinit, Tsurr,
                                    n_terms=200, n_r_int=2000):
    """
    给定截面点 (r_pt, z_pt)，返回理论水平位移 u_r
    """
    dr = r_pt - center_r
    dz = z_pt - center_z
    rho_p = np.sqrt(dr**2 + dz**2)

    if rho_p < 1e-14:
        return 0.0

    if rho_p > a + 1e-12:
        raise ValueError(f"Point outside sphere: rho_p={rho_p:.6e}, a={a:.6e}")

    u_radial = radial_disp_theory_sphere(
        rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
        n_terms=n_terms, n_r_int=n_r_int
    )

    # 投影到水平 r 方向
    return u_radial * (dr / rho_p)


from numba import njit
import numpy as np
@njit(fastmath=True)
def segment_lengths_in_core_shell_axisym(p1, p2, r_core, r_center, z_center, tol=1e-14):
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    L = np.sqrt(dx * dx + dz * dz)

    if L < tol:
        return 0.0, 0.0, 0.0

    x1 = p1[0] - r_center
    z1 = p1[1] - z_center
    x2 = p2[0] - r_center
    z2 = p2[1] - z_center

    in1 = (x1 * x1 + z1 * z1) <= r_core * r_core + tol
    in2 = (x2 * x2 + z2 * z2) <= r_core * r_core + tol

    if in1 and in2:
        return L, 0.0, L

    a = dx * dx + dz * dz
    b = 2.0 * (x1 * dx + z1 * dz)
    c = x1 * x1 + z1 * z1 - r_core * r_core

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t1 > t2:
        tmp = t1
        t1 = t2
        t2 = tmp

    has_t1 = (-tol <= t1 <= 1.0 + tol)
    has_t2 = (-tol <= t2 <= 1.0 + tol)

    if in1 != in2:
        t_hit = t1 if has_t1 else t2

        if t_hit < 0.0:
            t_hit = 0.0
        elif t_hit > 1.0:
            t_hit = 1.0

        if in1:
            L_core = t_hit * L
        else:
            L_core = (1.0 - t_hit) * L

        L_shell = L - L_core
        return L_core, L_shell, L

    if has_t1 and has_t2:
        t_in = t1
        t_out = t2

        if t_in < 0.0:
            t_in = 0.0
        if t_out > 1.0:
            t_out = 1.0

        if t_out > t_in + tol:
            L_core = (t_out - t_in) * L
            L_shell = L - L_core
            return L_core, L_shell, L

    return 0.0, L, L
@njit(parallel=True, fastmath=True)
def precompute_edge_core_shell_lengths(coords_all, edge_i, edge_j, r_core, r_center, z_center, tol=1e-14):
    nnz = edge_i.shape[0]

    L_core_edge = np.empty(nnz, dtype=np.float64)
    L_shell_edge = np.empty(nnz, dtype=np.float64)
    L_total_edge = np.empty(nnz, dtype=np.float64)

    for p in prange(nnz):
        i = edge_i[p]
        j = edge_j[p]

        p1 = coords_all[i]
        p2 = coords_all[j]

        Lc, Ls, Lt = segment_lengths_in_core_shell_axisym(
            p1, p2, r_core, r_center, z_center, tol
        )

        L_core_edge[p] = Lc
        L_shell_edge[p] = Ls
        L_total_edge[p] = Lt

    return L_core_edge, L_shell_edge, L_total_edge


@njit(parallel=True, fastmath=True)
def build_edge_property_harmonic_from_lengths(
    prop_node, is_core_node, edge_i, edge_j,
    L_core_edge, L_shell_edge, L_total_edge,
    r_node=None,
    mode="harm",
    eps=1e-30
):
    nnz = edge_i.shape[0]
    out = np.empty(nnz, dtype=np.float64)


    for p in prange(nnz):
        i = edge_i[p]
        j = edge_j[p]

        pi = prop_node[i]
        pj = prop_node[j]

        Lt = L_total_edge[p]
        Lc = L_core_edge[p]
        Ls = L_shell_edge[p]
        in_i = is_core_node[i]
        in_j = is_core_node[j]

        if Lt < eps:

            if mode == "mean":
                out[p] = 0.5 * (pi + pj)
            else:
                out[p] = pi
            continue

        # 同材料键
        if in_i == in_j:
            out[p] = pi
            continue

        # 跨界面键
        if in_i:
            p_core = pi
            p_shell = pj
        else:
            p_core = pj
            p_shell = pi


        # Default: true piecewise averaging based on the actual segment lengths.
        Wc = Lc
        Ws = Ls

        if r_node is not None:
            ri = r_node[i]
            rj = r_node[j]

            if in_i:
                Wc = Lc * ri
                Ws = Ls * rj
            else:
                Wc = Lc * rj
                Ws = Ls * ri

            if Wc + Ws < eps:
                Wc = Lc
                Ws = Ls

        Wt = Wc + Ws

        if mode == "harm":
            denom = Wc / (p_core + eps) + Ws / (p_shell + eps)
            out[p] = Wt / (denom + eps)
            continue

        if mode == "mean":
            out[p] = (Wc * p_core + Ws * p_shell) / (Wt + eps)
            continue

        out[p] = pi

    return out


def temperature_sphere_dirichlet(rho, t, a, kappa, Tinit, Tsurr, n_terms=200):
    """
    3D 实心球，初始均匀温度 Tinit，表面恒温 Tsurr
    返回温度理论解 T(rho, t)
    rho 可为标量或数组
    """
    rho = np.asarray(rho, dtype=float)
    T = np.empty_like(rho)

    n = np.arange(1, n_terms + 1, dtype=float)
    decay = np.exp(-(n**2) * (np.pi**2) * kappa * t / (a**2))

    mask = rho > 1e-14

    if np.any(mask):
        rr = rho[mask][:, None]
        series = np.sum(
            ((-1.0)**(n + 1) / n) * np.sin(n * np.pi * rr / a) * decay,
            axis=1
        )
        ratio = (2.0 * a / (np.pi * rho[mask])) * series
        T[mask] = Tsurr + (Tinit - Tsurr) * ratio

    # rho = 0 处取极限
    if np.any(~mask):
        series0 = np.sum(((-1.0)**(n + 1)) * decay)
        ratio0 = 2.0 * series0
        T[~mask] = Tsurr + (Tinit - Tsurr) * ratio0

    return T

def radial_disp_theory_sphere(rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
                              n_terms=100, n_r_int=1000):
    """
    3D 实心球在自由边界条件下的球对称径向位移理论解 u(rho_p, t)
    """
    if rho_p < 1e-14:
        return 0.0

    rho_grid = np.linspace(0.0, a, n_r_int)
    T_grid = temperature_sphere_dirichlet(
        rho_grid, t, a, kappa, Tinit, Tsurr, n_terms=n_terms
    )
    dT_grid = T_grid - Tinit

    Ia = np.trapz(dT_grid * rho_grid ** 2, rho_grid)


    mask = rho_grid <= rho_p
    rho_sub = rho_grid[mask]
    dT_sub = dT_grid[mask]
    Ir = np.trapz(dT_sub * rho_sub ** 2, rho_sub)

    u = (alpha / (1.0 - nu)) * (
        (1.0 + nu) * Ir / (rho_p**2)
        + 2.0 * (1.0 - 2.0 * nu) * rho_p * Ia / (a**3)
    )
    return u

def horizontal_disp_theory_at_point(r_pt, z_pt, t, a, center_r, center_z,
                                    kappa, alpha, nu, Tinit, Tsurr,
                                    n_terms=100, n_r_int=1000):
    """
    给定截面点 (r_pt, z_pt)，返回理论水平位移 u_r
    """
    dr = r_pt - center_r
    dz = z_pt - center_z
    rho_p = np.sqrt(dr**2 + dz**2)

    if rho_p < 1e-14:
        return 0.0

    if rho_p > a + 1e-12:
        raise ValueError(f"Point outside sphere: rho_p={rho_p:.6e}, a={a:.6e}")

    u_radial = radial_disp_theory_sphere(
        rho_p, t, a, kappa, alpha, nu, Tinit, Tsurr,
        n_terms=n_terms, n_r_int=n_r_int
    )

    # 投影到水平 r 方向
    return u_radial * (dr / rho_p)
