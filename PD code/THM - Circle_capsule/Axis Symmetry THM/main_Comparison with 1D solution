import generate_coordinates as gc
import ADR
import time
import Multiprocess_task_function as mt
import bc_funcs as bc
import core_funcs as cf
import Physical_Field_Calculation as pfc
import plot_utils as plot
import numpy as np

rho_s, cs, ks = 6800.0, 257.0, 31
rho_l, cl, kl = 6800.0, 257.0, 31
rho_air, cair, kair = 1, 1010, 0.0263
rho_shell, c_shell, k_shell = 7020.0, 348.95, 40
nu_core, nu_shell = 0.33, 0.284  # Poisson's ratio
E_core, alpha_core = 43.3e9, 18e-6
E_shell, alpha_shell = 222.72e9, 4e-6
sigmat = 803e6
G0 = 0.2  # J/m²
G1 = 10000  # J/m²

Ts = 598.65
Tl = 599.65
L = 60.627
number_pressure = 1
h = 1.
tolerance = 1e-14
Tsurr = 573.15
Tinit = 303.15
Tpre_avg = 303.15

""" Initialization of coarse regions and temperatures """
# dr1, dr2, dr3 are used to generate regional particle density by calling functions to create coordinates.
size = 1e-6
r = 24 * 1e-6  # Domain size in r and z directions (meters)
dshell = 0 * 1e-6
r_core = r - dshell
V0_all = np.pi * r_core ** 2 / 2

dr1 = 0.8 * size
dr2 = 0.8 * size
dr3 = 1 * dr2
dr_l = 0.8 * size
len1 = 8 * size
len2 = 8 * size
len3 = 8 * size

ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction
n_slices = 6

"""Note that n_slices here refers to the minimum number of layers. 
Subsequent checks and modifications will be performed 
in the gc.compute_layer_dr_r_nr function to ensure that the granularity of each layer is reasonable."""
dt_ADR = 1
total_time = 1e-5
rms = 5e-12
f_void = 0.
enable_void = (f_void > 1e-5)

graph_inner_shell = False
graph_point_t = False
graph_point_m = False

timemagnify = 0.5

# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
if __name__ == "__main__":
    # Shared zoning
    z0 = pfc.void_z_3d(f_void, r)
    start_time2 = time.time()
    """1. Definition of regional coordinates and area matrix (shared for T and M)"""
    zones, n_slices = gc.compute_layer_dr_r_nr(
        r, n_slices, dr1, dr2, dr3, dr_l, len1, len2, len3, size, ghost_nodes_r
    )
    for z in zones:
        print(f"Layer {z['layer']:2d}: dr = {z['dr']:.2e}, delta = {z['delta']:.2e}, "
              f"length = {z['length']:.2e}, Nr = {z['Nr']}")

    # ---------------- Temperature field ----------------
    phys_coords_list_t = []
    ghost_coords_list_t = []
    ghost_dict_list_t = []

    for i in range(n_slices):
        zone = zones[i]  # shared zone info
        dr_i = zone["dr"]
        Nr_i = zone["Nr"]
        delta_i = zone["delta"]

        coords_phys_t, ghost_coords_t, n_points_t, ghost_dict_t = gc.generate_one_slice_coordinates(
            r, Nr_i, ghost_nodes_r,
            zones,
            r_ghost_left=False,
            r_ghost_right=True,
            r_ghost_top=True,
            r_ghost_bot=True,
            n_slices=n_slices,
            slice_id=i,
            graph=graph_point_t
        )
        print(f"[Temperature] Number of particles in region {i}: {n_points_t}")

        phys_labeled_t = np.hstack([coords_phys_t, np.full((coords_phys_t.shape[0], 1), i)])
        ghost_labeled_t = np.hstack([ghost_coords_t, np.full((ghost_coords_t.shape[0], 1), i)])
        phys_coords_list_t.append(phys_labeled_t)
        ghost_coords_list_t.append(ghost_labeled_t)
        ghost_dict_list_t.append(ghost_dict_t)

    # ---------------- Mechanical field ----------------
    phys_coords_list_m = []
    ghost_coords_list_m = []
    ghost_dict_list_m = []

    for i in range(n_slices):
        zone = zones[i]  # same shared zoning
        dr_i = zone["dr"]
        Nr_i = zone["Nr"]

        coords_phys_m, ghost_coords_m, n_points_m, ghost_dict_m = gc.generate_one_slice_coordinates(
            r, Nr_i, ghost_nodes_r,
            zones,
            r_ghost_left=False,
            r_ghost_right=False,
            r_ghost_top=True,
            r_ghost_bot=True,
            n_slices=n_slices,
            slice_id=i,
            graph=graph_point_m
        )
        print(f"[Mechanical] Number of particles in region {i}: {n_points_m}")

        phys_labeled_m = np.hstack([coords_phys_m, np.full((coords_phys_m.shape[0], 1), i)])
        ghost_labeled_m = np.hstack([ghost_coords_m, np.full((ghost_coords_m.shape[0], 1), i)])
        phys_coords_list_m.append(phys_labeled_m)
        ghost_coords_list_m.append(ghost_labeled_m)
        ghost_dict_list_m.append(ghost_dict_m)

    # -------- Temperature field --------
    start_time_t = time.time()

    task_args_t = []
    coords_all_t = {}

    mask_core_regions_t = {}
    mask_core_regions_phy_t = {}
    mask_void_regions_t = {}

    for i in range(n_slices):
        coords_t = np.vstack([phys_coords_list_t[i], ghost_coords_list_t[i]])
        coords_all_t[i] = coords_t
        coords_phy_t = np.vstack(phys_coords_list_t[i])

        dr = zones[i]["dr"]  # shared
        delta = zones[i]["delta"]  # shared
        slice_id = zones[i]["layer"]  # shared
        task_args_t.append((coords_t, dr, delta, tolerance, slice_id))
        x_t = coords_t[:, 0]
        z_t = coords_t[:, 1]
        x_phy_t = coords_phy_t[:, 0]
        z_phy_t = coords_phy_t[:, 1]
        mask_phy_core = (x_phy_t ** 2 + (z_phy_t - r) ** 2) < r_core ** 2
        mask_core = (x_t ** 2 + (z_t - r) ** 2) < r_core ** 2
        mask_core_regions_t[i] = mask_core
        mask_core_regions_phy_t[i] = mask_phy_core

        if not enable_void:
            mask_void = np.zeros_like(mask_core, dtype=bool)
        else:
            mask_void = mask_core & (z_t >= (z0 - 1e-12))
        mask_void_regions_t[i] = mask_void

    results_t = [mt.compute_region_matrices(args) for args in task_args_t]
    csr_indptr_t, csr_indices_t, csr_dist_t, csr_area_t = [], [], [], []
    for indptr, indices, dist, area in results_t:
        csr_indptr_t.append(indptr)
        csr_indices_t.append(indices)
        csr_dist_t.append(dist)
        csr_area_t.append(area)

    end_time_t = time.time()
    print(
        f"[Temperature] Calculation of partial_area_matrices finished, elapsed real time = {end_time_t - start_time_t:.2f}s")

    # -------- Mechanical field --------
    start_time_m = time.time()

    task_args_m = []
    mask_core_regions_m = {}
    mask_void_regions_m = {}
    coords_all_m = {}

    for i in range(n_slices):
        coords_m = np.vstack([phys_coords_list_m[i], ghost_coords_list_m[i]])

        dr = zones[i]["dr"]
        delta = zones[i]["delta"]
        slice_id = zones[i]["layer"]

        task_args_m.append((coords_m, dr, delta, tolerance, slice_id))
        x_m, z_m = coords_m[:, 0], coords_m[:, 1]
        mask_core = (x_m ** 2 + (z_m - r) ** 2) < r_core ** 2
        mask_core_regions_m[i] = mask_core

        if not enable_void:
            mask_void_regions_m[i] = np.zeros_like(mask_core, dtype=bool)
        else:
            mask_void_regions_m[i] = mask_core & (z_m >= (z0 - 1e-12))
    # ---- CSR containers ----
    csr_indptr_m = []
    csr_indices_m = []
    csr_dist_m = []
    csr_area_m = []
    materials_m = {}
    edge_i_m = {}
    c_m = {}
    csr = {}
    nu_m = {}
    row_sum_area_m = {}

    coords_all_m_list = [np.vstack([phys_coords_list_m[i], ghost_coords_list_m[i]]) for i in range(n_slices)]
    coords_all_t_list = [np.vstack([phys_coords_list_t[i], ghost_coords_list_t[i]]) for i in range(n_slices)]
    results_m = [mt.compute_region_matrices(args) for args in task_args_m]

    miu_m = [None] * n_slices
    lamda_m = [None] * n_slices
    lamda_node_m = [None] * n_slices
    miu_node_m = [None] * n_slices  # ✅ 新增：存 μ_node
    alpha_m = [None] * n_slices  # 可选
    a_node_m = [None] * n_slices
    gama_node_m = [None] * n_slices
    kprime_m = [None] * n_slices
    kprime_node_m = [None] * n_slices

    for i, (indptr, indices, dist, area) in enumerate(results_m):
        csr_indptr_m.append(indptr)
        csr_indices_m.append(indices)
        csr_dist_m.append(dist)
        csr_area_m.append(area)

        coords_all_m = coords_all_m_list[i]
        coords_all_t = coords_all_t_list[i]

        mask_core_i = mask_core_regions_m[i]
        mask_void_i = mask_void_regions_m[i]

        # === Calculation of non-local mechanical parameters ===
        # --- 2) CSR（必须用机械场的 CSR，不要用 _t）---
        indptr = csr_indptr_m[i]
        indices = csr_indices_m[i]
        area = csr_area_m[i]  # (nnz,)
        N = len(indptr) - 1
        nnz = len(indices)

        edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
        edge_i_m[i] = edge_i

        # 行面积和：row_sum_area[i] = sum_j area_ij
        nu_node = np.where(mask_core_i, nu_core, nu_shell)  # (N,)
        nu_edge = 0.5 * (nu_node[edge_i] + nu_node[indices])  # (nnz,)
        nu_m[i] = nu_edge
        # --- 6) 节点 E（常数分区即可；void 可以置 0）--- (注意，在空腔被填满前，核内物理场为0！，因此不需要空腔的E)
        E_node = np.where(mask_core_i, E_core, E_shell).astype(np.float64)
        alpha_node = np.where(mask_core_i, alpha_core, alpha_shell).astype(np.float64)  # (N,)

        alpha_edge = 0.5 * (alpha_node[edge_i] + alpha_node[indices])  # (nnz,)
        E_edge = 0.5 * (E_node[edge_i] + E_node[indices])  # (nnz,)
        eps = 1e-30

        # --- node-wise Lamé parameters (for br_geom term) ---
        lamda0_node = (E_node * nu_node) / ((1.0 - 2.0 * nu_node) * (1.0 + nu_node) + eps)
        miu_node = E_node / (2.0 * (1.0 + nu_node) + eps)
        lamda_node = lamda0_node - miu_node

        lamda_m[i] = 0.5 * (lamda_node[edge_i] + lamda_node[indices])
        miu_m[i] = 0.5 * (miu_node[edge_i] + miu_node[indices])
        miu_node_m[i] = miu_node
        lamda_node_m[i] = lamda_node

        delta_i = zones[i]["delta"]
        c_micro_node = 6.0 * E_node / (np.pi * delta_i ** 3 * h * (1.0 - 2.0 * nu_node) * (1.0 + nu_node))
        c_micro_edge = 0.5 * (c_micro_node[edge_i] + c_micro_node[indices])  # (nnz,)
        row_sum_area = np.bincount(edge_i, weights=area, minlength=N).astype(np.float64)
        denom = row_sum_area[edge_i] + row_sum_area[indices]
        csr[i] = (2.0 * np.pi * delta_i ** 2 / denom)
        c_m[i] = (2.0 * np.pi * delta_i ** 2 / denom) * c_micro_edge
        # ---- c_edge 已经算完：现在把 self-area 置 0（仅影响后续使用 area 的地方）----
        mask_self = (indices == edge_i)
        if np.any(mask_self):
            area = area.copy()
            area[mask_self] = 0.0
            csr_area_m[i] = area  # 更新列表里保存的 area（后续都用“self=0”的版本）
        # ==========================
        kprime_node = (E_node / (1.0 - 2.0 * nu_node)) * alpha_node  # (N,)
        kprime_node_m[i] = kprime_node
        kprime_m[i] = 0.5 * (kprime_node[edge_i] + kprime_node[indices])  # (nnz,)

    end_time_m = time.time()
    print(f"[Mechanical] CSR matrices finished, elapsed real time = {end_time_m - start_time_m:.2f}s")

    """2. Definition of particle information at the interface between regions """
    # Initialize structured dictionary
    # Temperature field boundary mapping
    ghost_top_t = {}
    ghost_bot_t = {}
    ghost_left_t = {}
    ghost_right_t = {}
    phys_t = {}

    # Save the matching relationship between each pair of adjacent regions
    boundary_neighbors_t = {}
    boundary_distances_t = {}
    for i in range(n_slices):  # Region i is adjacent to region i+1 (vertical direction)

        ghost_top_t[i] = ghost_dict_list_t[i]["top"]
        ghost_bot_t[i] = ghost_dict_list_t[i]["bot"]
        ghost_left_t[i] = ghost_dict_list_t[i]["left"]
        ghost_right_t[i] = ghost_dict_list_t[i]["right"]
        phys_t[i] = phys_coords_list_t[i]

    for i in range(n_slices):  # Region i is adjacent to region i+1 (vertical direction)
        # The lower boundary ghost point of region i corresponds to the physical point of region i+1
        if i <= n_slices - 2:
            dr_i = zones[i]["dr"]
            dr_ip1 = zones[i + 1]["dr"]
            ghost_coords_bot_t = ghost_bot_t[i][:, :2]
            phys_coords_ip1_t = phys_t[i + 1][:, :2]

            if abs(dr_i - dr_ip1) < tolerance:
                ghost_idx_bot_t, phys_idx_bot_t = bc.get_same_neighbor_points(
                    ghost_coords_bot_t, phys_coords_ip1_t, tol=tolerance)
            elif dr_i < dr_ip1:
                ghost_idx_bot_t, phys_idx_bot_t, dist_bot_t = bc.get_coarse_neighbor_points(
                    ghost_coords_bot_t, phys_coords_ip1_t, dr_fine=dr_i, tol=tolerance
                )
                boundary_distances_t[(i, 'bot')] = dist_bot_t
            else:  # dr_i > dr_ip1
                ghost_idx_bot_t, phys_idx_bot_t = bc.get_fine_neighbor_points(
                    ghost_coords_bot_t, phys_coords_ip1_t, dr_fine=dr_ip1, tol=tolerance)

            boundary_neighbors_t[(i, 'bot')] = {
                "ghost_indices": ghost_idx_bot_t,
                "phys_indices": phys_idx_bot_t,
                "target_region": i + 1
            }

            # Upper boundary of region i+1
            ghost_coords_top_t = ghost_top_t[i + 1][:, :2]
            phys_coords_i_t = phys_t[i][:, :2]
            if abs(dr_i - dr_ip1) < tolerance:
                ghost_idx_top_t, phys_idx_top_t = bc.get_same_neighbor_points(
                    ghost_coords_top_t, phys_coords_i_t, tol=tolerance)
            elif dr_ip1 < dr_i:
                ghost_idx_top_t, phys_idx_top_t, dist_top_t = bc.get_coarse_neighbor_points(
                    ghost_coords_top_t, phys_coords_i_t, dr_fine=dr_ip1, tol=tolerance
                )
                boundary_distances_t[(i + 1, 'top')] = dist_top_t
            else:
                ghost_idx_top_t, phys_idx_top_t = bc.get_fine_neighbor_points(
                    ghost_coords_top_t, phys_coords_i_t, dr_fine=dr_i, tol=tolerance)

            boundary_neighbors_t[(i + 1, 'top')] = {
                "ghost_indices": ghost_idx_top_t,
                "phys_indices": phys_idx_top_t,
                "target_region": i
            }

        coords_ghost_left_t = ghost_left_t[i][:, :2]
        coords_ghost_right_t = ghost_right_t[i][:, :2]
        coords_phys_t = phys_t[i][:, :2]
        dr = zones[i]["dr"]

        # Right boundary

        parts = []
        for a in (phys_coords_list_t[i], ghost_left_t[i], ghost_top_t[i], ghost_bot_t[i]):
            A = np.asarray(a)
            A = A[:, :2]  # 统一只取前两列
            parts.append(A)

        coords_t_pltb = np.vstack(parts) if parts else np.empty((0, 2), float)

        parts2 = []
        for a in (phys_coords_list_t[i], ghost_left_t[i], ghost_right_t[i], ghost_top_t[i], ghost_bot_t[i]):
            A = np.asarray(a)
            A = A[:, :2]  # 统一只取前两列
            parts2.append(A)
        coords_t2 = np.vstack(parts2) if parts2 else np.empty((0, 2), float)

        ghost_idx_left_t, phys_idx_left_t = bc.find_mirror_pairs(
            coords_ghost_left_t, coords_t2, tolerance
        )
        boundary_neighbors_t[(i, 'left')] = {
            "ghost_indices": ghost_idx_left_t,
            "phys_indices": phys_idx_left_t,
            "target_region": i
        }

        # Right boundary
        ghost_idx_right_t, phys_idx_right_t = bc.find_circle_mirror_pairs_multilayer(
            coords_ghost_right_t, coords_t_pltb, dr, r
        )
        boundary_neighbors_t[(i, 'right')] = {
            "ghost_indices": ghost_idx_right_t,
            "phys_indices": phys_idx_right_t,
            "target_region": i
        }

    # Mechanical field boundary mapping
    ghost_top_m = {}
    ghost_right_m = {}
    ghost_bot_m = {}
    ghost_left_m = {}
    phys_m = {}

    for i in range(n_slices):
        ghost_top_m[i] = ghost_dict_list_m[i]["top"]
        ghost_bot_m[i] = ghost_dict_list_m[i]["bot"]
        ghost_left_m[i] = ghost_dict_list_m[i]["left"]
        ghost_right_m[i] = ghost_dict_list_m[i]["right"]
        phys_m[i] = phys_coords_list_m[i]

    # Save the matching relationship between each pair of adjacent regions
    boundary_neighbors_m = {}
    boundary_distances_m = {}

    for i in range(n_slices):  # Region i is adjacent to region i+1 (vertical direction)
        # The lower boundary ghost point of region i corresponds to the physical point of region i+1
        if i <= n_slices - 2:
            dr_i = zones[i]["dr"]
            dr_ip1 = zones[i + 1]["dr"]
            ghost_coords_bot_m = ghost_bot_m[i][:, :2]
            phys_coords_ip1_m = phys_m[i + 1][:, :2]

            if abs(dr_i - dr_ip1) < tolerance:
                ghost_idx_bot_m, phys_idx_bot_m = bc.get_same_neighbor_points(
                    ghost_coords_bot_m, phys_coords_ip1_m, tol=tolerance)
            elif dr_i < dr_ip1:
                ghost_idx_bot_m, phys_idx_bot_m, dist_bot_m = bc.get_coarse_neighbor_points(
                    ghost_coords_bot_m, phys_coords_ip1_m, dr_fine=dr_i, tol=tolerance
                )
                boundary_distances_m[(i, 'bot')] = dist_bot_m
            else:  # dr_i > dr_ip1
                ghost_idx_bot_m, phys_idx_bot_m = bc.get_fine_neighbor_points(
                    ghost_coords_bot_m, phys_coords_ip1_m, dr_fine=dr_ip1, tol=tolerance)

            boundary_neighbors_m[(i, 'bot')] = {
                "ghost_indices": ghost_idx_bot_m,
                "phys_indices": phys_idx_bot_m,
                "target_region": i + 1
            }

            # Upper boundary of region i+1
            ghost_coords_top_m = ghost_top_m[i + 1][:, :2]
            phys_coords_i_m = phys_m[i][:, :2]
            if abs(dr_i - dr_ip1) < tolerance:
                ghost_idx_top_m, phys_idx_top_m = bc.get_same_neighbor_points(
                    ghost_coords_top_m, phys_coords_i_m, tol=tolerance)
            elif dr_ip1 < dr_i:
                ghost_idx_top_m, phys_idx_top_m, dist_top_m = bc.get_coarse_neighbor_points(
                    ghost_coords_top_m, phys_coords_i_m, dr_fine=dr_ip1, tol=tolerance
                )
                boundary_distances_m[(i + 1, 'top')] = dist_top_m
            else:
                ghost_idx_top_m, phys_idx_top_m = bc.get_fine_neighbor_points(
                    ghost_coords_top_m, phys_coords_i_m, dr_fine=dr_i, tol=tolerance)

            boundary_neighbors_m[(i + 1, 'top')] = {
                "ghost_indices": ghost_idx_top_m,
                "phys_indices": phys_idx_top_m,
                "target_region": i
            }

    # Note: No right boundary for mechanical field

    """3. Definition of temperature field"""
    factor_data_t = {}  # 或者 list，随你
    T_phys, T_left, T_right, T_top, T_bot = {}, {}, {}, {}, {}
    dt_th = cf.compute_dt_cr_th_solid_with_csr(
        rho_s, cs, ks,
        csr_indptr_t[0],
        csr_dist_t[0],
        csr_area_t[0],
        zones[0]["delta"]
    ) * timemagnify

    nsteps_th = int(total_time / dt_th)
    T_increment = Tsurr  # Tinit + (Tsurr - Tinit) / nsteps_th * 5

    for i in range(n_slices):
        threshold_distance = np.sqrt(2) * zones[i]["dr"] + tolerance
        factor_data_t[i] = np.where(
            csr_dist_t[i] <= threshold_distance,
            1.125,
            1.0
        ).astype(np.float64)
        T_phys[i] = np.full(phys_t[i].shape[0], Tinit)
        T_left[i] = np.full(ghost_left_t[i].shape[0], Tinit)
        T_right[i] = np.full(ghost_right_t[i].shape[0], T_increment)
        T_top[i] = np.full(ghost_top_t[i].shape[0], Tinit)
        T_bot[i] = np.full(ghost_bot_t[i].shape[0], Tinit)
    # Boundary temperature assignment
    T = {}
    for (region_id, direction), neighbor_data in boundary_neighbors_t.items():
        if direction == 'right':
            T1 = np.concatenate(
                [T_phys[region_id], T_left[region_id], T_top[region_id], T_bot[region_id]])
            ghost_indices = neighbor_data["ghost_indices"]
            phys_indices = neighbor_data["phys_indices"]  # 这些索引是相对于 phys_coords 的
            T_right[region_id][ghost_indices] = 2 * Tsurr - T1[phys_indices]  # 你现在用的是 Dirichlet
            T[region_id] = np.concatenate(
                [T_phys[region_id], T_left[region_id], T_right[region_id], T_top[region_id], T_bot[region_id]])
            # 如果你以后要用镜像/反射形式，就写：
            # T_right[region_id][ghost_indices] = 2*T_increment - T_phys[region_id][phys_indices]

    """4. Definition of enthalpy for temperature field"""
    H = {}
    K = {}
    segment_slices_t = {}

    for i in range(n_slices):
        segments = {
            "phys": T_phys[i],
            "left": T_left[i],
            "right": T_right[i],
            "top": T_top[i],
            "bot": T_bot[i],
        }
        # build slices
        s = 0
        segment_slices_t[i] = {}
        for name, arr in segments.items():
            n = len(arr)
            segment_slices_t[i][name] = slice(s, s + n)
            s += n
        segment_slices_t[i]["total_len"] = s

    deltaV_total_all = 0.0  # Initialize total expansion amount

    for i in range(n_slices):
        dr_i = zones[i]["dr"]
        dA = dr_i * dr_i
        r_phys = phys_coords_list_t[i][:, 0]  # 或 phys_coords_list_t[i][:,0]，用物理点半径更合适
        cell_volume_node = 2.0 * np.pi * r_phys * dA  # (N_phys,)

        deltaV_phase, deltaV_thermal = pfc.compute_melt_and_thermal_expansion(
            T_phys[i], mask_core_regions_phy_t[i], rho_s, rho_l, Ts, Tl, cell_volume_node,
            beta_s=0,
            beta_l=alpha_core,
            T_ref=Tpre_avg
        )
        deltaV_total_all += deltaV_phase + deltaV_thermal

    if f_void <= 1e-10:
        phi_global = 1.0
        cavity_vol = 0.0
    else:
        cavity_vol = f_void * (4.0 / 3.0) * np.pi * r_core ** 3
        phi_global = min(1.0, deltaV_total_all / cavity_vol)

    alpha_core_cal = 0.0 if (f_void > 1e-10 and deltaV_total_all <= cavity_vol) else alpha_core

    """4.1 空腔属性的定义"""
    rho_void = phi_global * rho_l + (1.0 - phi_global) * rho_air
    Cp_void = phi_global * cl + (1.0 - phi_global) * cair
    k_void = phi_global * kl + (1.0 - phi_global) * kair

    """4.2 焓及轴对称因子的初始定义"""
    shape_factor_t = {}
    for i in range(n_slices):
        H[i] = cf.get_enthalpy(
            T[i],
            mask_core_regions_t[i],
            mask_void_regions_t[i],
            rho_s, rho_l, rho_shell,
            cs, cl,
            c_shell,  # ✅ 壳层 Cp 常数
            L, Ts, Tl,
            rho_void,
            Cp_void
        )
        r_node = coords_all_t_list[i][:, 0]
        N_t = csr_indptr_t[i].size - 1
        edge_i_t = np.repeat(np.arange(N_t, dtype=np.int64), np.diff(csr_indptr_t[i]))
        shape_factor_t[i] = cf.compute_shape_factor_edge_csr(
            r_node,
            edge_i_t,
            csr_indices_t[i],
            mechanical=False
        )

    """5. Definition of Mechanical field"""
    # Mechanical field initialization
    # ---------- 1) Initialize displacement/velocity/acceleration ----------
    Ur, Uz = {}, {}
    Ar, Az = {}, {}
    br, bz = {}, {}

    segment_slices_m = {}
    region_lengths_m = {}
    CorrList_T = {}
    T_m = {}
    dir_r_m, dir_z_m = {}, {}
    inner_surface_info_list = []
    dx0_edge_m = []
    dz0_edge_m = []
    pre_m = [None] * n_slices

    for i in range(n_slices):
        coords_all_m = coords_all_m_list[i]
        coords_all_t = coords_all_t_list[i]

        CorrList_T[i] = pfc.shrink_Tth_by_matching_coords(coords_all_m, coords_all_t)
        T_m = pfc.filter_array_by_indices_keep_only(T[i], CorrList_T[i])
        # 或者直接 materials_m[i]["T_m"] = ...

        mask_core_i = mask_core_regions_m[i]
        mask_void_i = mask_void_regions_m[i]

        # === Calculation of non-local mechanical parameters ===
        # --- 2) CSR（必须用机械场的 CSR，不要用 _t）---
        indptr = csr_indptr_m[i]
        indices = csr_indices_m[i]
        dist = csr_dist_m[i]  # (nnz,)
        area = csr_area_m[i]  # (nnz,)
        N = len(indptr) - 1
        nnz = len(indices)

        edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
        edge_j = csr_indices_m[i]  # (nnz,)
        mu_edge = np.ones(nnz, dtype=np.int8)

        # --- alpha ---
        alpha_node = np.where(mask_core_i, alpha_core_cal, alpha_shell).astype(np.float64)
        alpha_edge = 0.5 * (alpha_node[edge_i] + alpha_node[indices])  # (nnz,)
        # --- s0 ---
        delta_i = zones[i]["delta"]
        crack = np.ones(nnz, dtype=np.int8)
        s0_core = 10
        s0_shell = sigmat / E_shell
        s0_node = np.where(mask_core_i, s0_core, s0_shell).astype(np.float64)
        s0_edge = 0.5 * (s0_node[edge_i] + s0_node[indices])  # (nnz,)

        rho_node, rho_edge = cf.get_density(
            CorrList_T[i], mask_core_i, mask_void_i,
            rho_s, rho_l, Ts, Tl,
            rho_shell, rho_void,
            edge_i, indices
        )

        dir_r_m, dir_z_m = pfc.compute_direction_edges_csr_numba(
            coords_all_m_list[i], edge_i, edge_j, dist
        )
        dx0_edge_m.append(dir_r_m * dist)
        dz0_edge_m.append(dir_z_m * dist)

        r_node = coords_all_m_list[i][:, 0]
        w_maxe = np.maximum(np.abs(dir_r_m), np.abs(dir_z_m))

        # --- storage ---
        materials_m[i] = {
            "T_m": T_m,  # (N,)
            "c_edge": c_m[i],  # (N,)
            "csr": csr[i],
            "nu_edge": nu_m[i],  # (N,)
            "edge_i": edge_i_m[i],  # (N,)
            "alpha_edge": alpha_edge,  # (nnz,)
            "s0_edge": s0_edge,  # (nnz,)
            "crack": crack,  # (nnz,)
            "rho_node": rho_node,
            "rho_edge": rho_edge,
            "dir_r_edge": dir_r_m,
            "dir_z_edge": dir_z_m,
        }

        pre_m[i] = {
            "mask_core": mask_core_i,
            "mask_void": mask_void_i,
            "edge_i": edge_i,
            "edge_j": edge_j,
            "alpha_edge": alpha_edge,
            "s0_edge": s0_edge,
            "crack": crack,
            "dir_r_edge": dir_r_m,
            "dir_z_edge": dir_z_m,
        }

        dr_i = zones[i]["dr"]
        surface_info = pfc.find_inner_surface_layer(
            coords_all_m,
            r, dshell,
            dr_i,
        )
        inner_surface_info_list.append(surface_info)
        region_lengths_m[i] = {
            'n_phys': phys_coords_list_m[i].shape[0],
            'n_left': ghost_left_m[i].shape[0],
            'n_right': ghost_right_m[i].shape[0],
            'n_top': ghost_top_m[i].shape[0],
            'n_bot': ghost_bot_m[i].shape[0]
        }

        s0 = 0
        s1 = s0 + phys_coords_list_m[i].shape[0]
        s2 = s1 + ghost_left_m[i].shape[0]
        s3 = s2 + ghost_right_m[i].shape[0]
        s4 = s3 + ghost_top_m[i].shape[0]
        s5 = s4 + ghost_bot_m[i].shape[0]

        segment_slices_m[i] = {
            "phys": slice(s0, s1),
            "left": slice(s1, s2),
            "right": slice(s2, s3),
            "top": slice(s3, s4),
            "bot": slice(s4, s5),
            "total_len": s5
        }

        total_points_region = (
                phys_coords_list_m[i].shape[0]
                + ghost_top_m[i].shape[0]
                + ghost_bot_m[i].shape[0]
                + ghost_left_m[i].shape[0]
                + ghost_right_m[i].shape[0]
        )

        Ur[i] = np.zeros(total_points_region)  # radial disp
        Uz[i] = np.zeros(total_points_region)  # axial disp
        Ar[i] = np.zeros(total_points_region)  # radial acc
        Az[i] = np.zeros(total_points_region)  # axial acc
        br[i] = np.zeros(total_points_region)
        bz[i] = np.zeros(total_points_region)

    # 循环外先定义
    axisym_cache_m = [None] * n_slices
    axis_mask_m = []
    # 定义左边界
    for i in range(n_slices):
        coords_all_m = coords_all_m_list[i]
        r_all = coords_all_m[:, 0]
        dr = zones[i]["dr"]
        eps = 1e-20
        axis_mask = (r_all > eps) & (r_all < 1 * dr - eps)  # 0<r<dr
        axis_mask_m.append(axis_mask)

        # 当前 slice 的 r,z（参考构形，固定）
        r_flat = coords_all_m[:, 0].astype(np.float64)
        z_flat = coords_all_m[:, 1].astype(np.float64)

        indices = csr_indices_m[i]  # (nnz,)
        dist_e = csr_dist_m[i]  # (nnz,) 参考键长
        area_e = csr_area_m[i]  # (nnz,) partial_area_flat_csr

        N = coords_all_m.shape[0]
        edge_i = edge_i_m[i]  # <- 推荐：你既然已 CSR 化，应该已经缓存过它

        shape_edge_m = cf.compute_shape_factor_edge_csr(r_flat, edge_i_m[i], csr_indices_m[i], mechanical=True)

        # coeff：你说已经把 2*pi*r 融进 coeff 了，并且是标量
        # 这里按你的定义传进去即可（示例：）
        delta_i = zones[i]["delta"]
        coeff = 3.0 / (np.pi * delta_i ** 3)  # <- 你如果已经改成包含2πr的版本，就用你的coeff

        axisym_cache_m[i] = {
            "r_flat": r_flat,
            "z_flat": z_flat,
            "edge_i": edge_i,
            "indices": indices,
            "dist_e": dist_e,
            "area_e": area_e,
            "shape_e": shape_edge_m,
            "coeff": float(coeff),
        }

    Fr = {}
    Fz = {}
    cr_n = {}
    cz_n = {}
    lambda_diag_matrix = {}
    lambda_diag_matrix1 = {}
    Fr_0 = [np.zeros_like(Ur[i]) for i in range(n_slices)]
    Fz_0 = [np.zeros_like(Uz[i]) for i in range(n_slices)]
    Vr_half = [np.zeros_like(Ur[i]) for i in range(n_slices)]
    Vz_half = [np.zeros_like(Uz[i]) for i in range(n_slices)]

    for i in range(n_slices):
        '''lambda_diag_matrix[i] = ADR.compute_lambda_diag_matrix(
            partial_area_flat=csr_area_m[i],
            dist=csr_dist_m[i],
            indptr=csr_indptr_m[i],
            c_edge=materials_m[i]["c_edge"],  # (nnz,)
            dt=dt_ADR,
            edge_i=edge_i_m[i]
        )'''

        indptr = csr_indptr_m[i]
        indices = csr_indices_m[i]
        N = len(indptr) - 1
        nnz = len(indices)
        dist = csr_dist_m[i]  # (nnz,)
        area = csr_area_m[i]  # (nnz,)
        r_flat = coords_all_m_list[i][:, 0]

        edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
        edge_j = csr_indices_m[i]  # (nnz,)
        dir_r_m, dir_z_m = pfc.compute_direction_edges_csr_numba(
            coords_all_m_list[i], edge_i, edge_j, dist
        )
        w_maxe_edge = np.maximum(np.abs(dir_r_m), np.abs(dir_z_m)).astype(np.float64)
        shape_edge = axisym_cache_m[i]["shape_e"]  # 暂时不用 shape 就传 1

        delta = zones[i]["delta"]
        q = 4 / 3 * np.pi ** 2 * delta ** 4
        lambda_diag_matrix[i] = ADR.compute_lambda_diag_matrix_axsy(
            indptr, indices,
            area, dist,
            w_maxe_edge,
            shape_edge,
            lamda_node_m[i], miu_node_m[i],
            r_flat,
            delta, q
        )

    # ------------------------
    # Time-stepping loop
    # ------------------------
    p = 0
    print_interval = int(10 / dt_th)  # Print progress every 10 simulated seconds
    print(f"Total steps: {nsteps_th}")
    print(f"thermal steps: {dt_th}")
    start_time = time.time()
    has_frac = False
    nsteps_m = int(10000)
    if graph_inner_shell:
        plot.plot_inner_shell_points(inner_surface_info_list, coords_all_m_list, show_normals=False,
                                     title="Inner surface points (all regions)")


    def find_closest_point_in_phys(phys_coords_list_m, target_r, target_z, n_slices):
        best_slice = None
        best_local_idx = None
        best_coord = None
        best_dist2 = np.inf

        for i in range(n_slices):
            coords = phys_coords_list_m[i]

            r_coords = coords[:, 0]
            z_coords = coords[:, 1]

            dist2 = (r_coords - target_r) ** 2 + (z_coords - target_z) ** 2
            local_idx = np.argmin(dist2)

            if dist2[local_idx] < best_dist2:
                best_dist2 = dist2[local_idx]
                best_slice = i
                best_local_idx = local_idx
                best_coord = coords[local_idx].copy()

        return best_slice, best_local_idx, best_coord


    target_points = [
        ("(r/2, r)", r/2, r),
        ("(r, r)", r, r),
    ]

    tracked_points = []

    for label, target_r, target_z in target_points:
        best_slice, best_local_idx, best_coord = find_closest_point_in_phys(
            phys_coords_list_m, target_r, target_z, n_slices
        )

        tracked_points.append({
            "label": label,
            "target_r": target_r,
            "target_z": target_z,
            "slice": best_slice,
            "local_idx": best_local_idx,
            "coord": best_coord
        })

        print(label)
        print("  best_slice =", best_slice)
        print("  best_local_idx =", best_local_idx)
        print("  best_coord =", best_coord)

    time_history = []

    ur_histories = {pt["label"]: [] for pt in tracked_points}
    T_histories = {pt["label"]: [] for pt in tracked_points}
    rho_void_old, Cp_void_old = rho_air, cair
    rho_void_new, Cp_void_new = rho_air, cair
    # 在每个时间步里

    for step1 in range(nsteps_th):

        current_time = (step1 + 1) * dt_th
        time_history.append(current_time)

        for pt in tracked_points:
            i_slice = pt["slice"]
            i_local = pt["local_idx"]

            current_ur = Ur[i_slice][i_local]
            current_T = T[i_slice][i_local]

            ur_histories[pt["label"]].append(current_ur)
            T_histories[pt["label"]].append(current_T)

        for i in range(n_slices):
            K_data, diag = cf.build_Kdata_and_rowsum_csr_numba(
                T[i], mask_core_regions_t[i], mask_void_regions_t[i],
                factor_data_t[i], csr_area_t[i], shape_factor_t[i], csr_dist_t[i],
                csr_indptr_t[i], csr_indices_t[i],
                ks, kl, Ts, Tl,
                k_shell, zones[i]["delta"],
                k_void,
                dt_th
            )

            dH = cf.apply_K_with_diag_csr_numba(
                csr_indptr_t[i], csr_indices_t[i], K_data, diag, T[i],
                mask_void_regions_t[i],
                rho_void_old, Cp_void_old,
                rho_void_new, Cp_void_new
            )
            H[i] = H[i] + dH
            # ② 直接调用：H -> T
            T[i] = cf.temperature_from_enthalpy_numba(
                H[i],
                mask_core_regions_t[i], mask_void_regions_t[i],
                rho_s, rho_l, cs, cl, L, Ts, Tl,
                rho_shell, c_shell,
                rho_void, Cp_void
            )
        # ✅ Split results
        for i in range(n_slices):
            sl = segment_slices_t[i]
            T_phys[i] = T[i][sl["phys"]]
            T_left[i] = T[i][sl["left"]]
            T_right[i] = T[i][sl["right"]]
            T_top[i] = T[i][sl["top"]]
            T_bot[i] = T[i][sl["bot"]]

        # ✅ Unified handling of boundary conditions
        # T_increment = Tsurr#min(Tinit + (Tsurr - Tinit) * (step1 + 2) / nsteps_th * 5,Tsurr)
        for (region_id, direction), neighbor_data in boundary_neighbors_t.items():

            ghost_indices = neighbor_data["ghost_indices"]
            phys_indices = neighbor_data["phys_indices"]
            target_region = neighbor_data["target_region"]

            dr1 = zones[region_id]["dr"]
            dr2 = zones[target_region]["dr"]

            if direction == "top":
                if abs(dr1 - dr2) < tolerance:
                    T_top[region_id] = bc.interpolate_temperature_for_same(
                        T_top[region_id], T_phys[target_region], ghost_indices, phys_indices
                    )
                elif dr1 < dr2:
                    T_top[region_id] = bc.interpolate_temperature_for_coarse(
                        T_top[region_id], T_phys[target_region], ghost_indices, phys_indices,
                        boundary_distances_t[(region_id, "top")]
                    )
                else:
                    T_top[region_id] = bc.interpolate_temperature_for_fine(T_top[region_id], T_phys[target_region],
                                                                           ghost_indices, phys_indices, )
            elif direction == "bot":
                if abs(dr1 - dr2) < tolerance:
                    T_bot[region_id] = bc.interpolate_temperature_for_same(
                        T_bot[region_id], T_phys[target_region], ghost_indices, phys_indices)
                elif dr1 < dr2:
                    T_bot[region_id] = bc.interpolate_temperature_for_coarse(
                        T_bot[region_id], T_phys[target_region], ghost_indices, phys_indices,
                        boundary_distances_t[(region_id, "bot")])
                else:
                    T_bot[region_id] = bc.interpolate_temperature_for_fine(
                        T_bot[region_id], T_phys[target_region], ghost_indices, phys_indices)

        for (region_id, direction), neighbor_data in boundary_neighbors_t.items():
            if direction == 'right':
                T1 = np.concatenate(
                    [T_phys[region_id], T_left[region_id], T_top[region_id], T_bot[region_id]])
                ghost_indices = neighbor_data["ghost_indices"]
                phys_indices = neighbor_data["phys_indices"]
                T_right[region_id][ghost_indices] = 2 * Tsurr - T1[phys_indices]
                T[region_id] = np.concatenate(
                    [T_phys[region_id], T_left[region_id], T_right[region_id], T_top[region_id], T_bot[region_id]])

        """for i in range(n_slices):
            coords = coords_all_t_list[i]  # (Ni,2)
            dr_i = zones[i]["dr"]  # 你的 dr
            idx_in, idx_out = bc.build_axis_firstlayer_pair_mask(coords, dr_i)  # 如果你有dz更好

            T[i] = bc.apply_T_copy_with_mask(T[i], idx_in, idx_out)"""

        print(f"✅ Completed {step1 + 1}/{nsteps_th} steps of calculation")
        deltaV_total_all = 0
        for i in range(n_slices):
            dr_i = zones[i]["dr"]
            dA = dr_i * dr_i
            r_phys = phys_coords_list_t[i][:, 0]  # 或 phys_coords_list_t[i][:,0]，用物理点半径更合适
            cell_volume_node = 2.0 * np.pi * r_phys * dA  # (N_phys,)

            deltaV_phase, deltaV_thermal = pfc.compute_melt_and_thermal_expansion(
                T_phys[i], mask_core_regions_phy_t[i], rho_s, rho_l, Ts, Tl, cell_volume_node,
                beta_s=0,
                beta_l=alpha_core,
                T_ref=Tpre_avg
            )
            deltaV_total_all += deltaV_phase + deltaV_thermal

        if f_void <= 1e-10:
            phi_global = 1.0
            cavity_vol = 0.0
        else:
            cavity_vol = f_void * (4.0 / 3.0) * np.pi * r_core ** 3
            phi_global = min(1.0, deltaV_total_all / cavity_vol)

        alpha_core_cal = 0.0 if (f_void > 1e-10 and deltaV_total_all <= cavity_vol) else alpha_core

        if k_void > kl - 1:
            aa = 1

        """ 空腔属性的update"""
        rho_void = phi_global * rho_l + (1.0 - phi_global) * rho_air
        Cp_void = phi_global * cl + (1.0 - phi_global) * cair
        k_void = phi_global * kl + (1.0 - phi_global) * kair
        cracking_act = False
        for i in range(n_slices):
            # --- 1) 更新 T_m ---
            Corr = CorrList_T[i]  # 初始化算过，直接用
            T_m = pfc.filter_array_by_indices_keep_only(T[i], Corr)
            materials_m[i]["T_m"] = T_m

            # 取初始化预存的常量/拓扑
            edge_i = pre_m[i]["edge_i"]
            edge_j = pre_m[i]["edge_j"]
            mask_core_i = pre_m[i]["mask_core"]
            mask_void_i = pre_m[i]["mask_void"]

            # --- 2) 重新计算 alpha_edge（按你原来的公式）---
            # 如果 alpha_core_cal/alpha_shell 未来要做成温度函数，也可以在这里用 T_m 来更新它们
            alpha_node = np.where(mask_core_i, alpha_core_cal, alpha_shell).astype(np.float64)
            alpha_edge = 0.5 * (alpha_node[edge_i] + alpha_node[edge_j])
            materials_m[i]["alpha_edge"] = alpha_edge

            # --- 3) 重新计算 rho_node / rho_edge ---
            rho_node, rho_edge = cf.get_density(
                T_m, mask_core_i, mask_void_i,
                rho_s, rho_l, Ts, Tl,
                rho_shell, rho_void,
                edge_i, edge_j
            )
            materials_m[i]["rho_node"] = rho_node
            materials_m[i]["rho_edge"] = rho_edge
        """6. Definition of physical strength: Note that the average value of the elastic modulus is taken here."""

        phi_regions = {}
        p_pre = p
        if deltaV_total_all <= V0_all * f_void:
            p = 0
        else:
            p = E_shell * ((deltaV_total_all - 0.5 * f_void * np.pi * r ** 2) / V0_all) / np.pi / r

        print(f"[Step {step1 + 1}] Time = {step1 * dt_th:.3e} s | Pressure p = {p:.3e} Pa")
        ninc = number_pressure
        delta_p = p - p_pre  # 本热步相对上一热步的新增压力
        inc_p = delta_p / ninc  # 每个增量的压力
        if step1 == 0:
            for i in range(n_slices):
                idx_local = inner_surface_info_list[i]["indices"]
                unit_vec = inner_surface_info_list[i]["unit_to_center"]
                dr_i = zones[i]["dr"]
                br[i][idx_local] += 0
                bz[i][idx_local] += 0

            """     7. Displacement initialization      """

            for i in range(n_slices):
                # 1) 当前构形下的 dilation / eij / n
                cache = axisym_cache_m[i]
                # 1) dilation + eij + n  (一次遍历 nnz 得到)
                dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr(
                    cache["r_flat"], cache["z_flat"],
                    Ur[i], Uz[i],
                    cache["edge_i"],
                    cache["indices"],
                    cache["dist_e"],
                    cache["area_e"],
                    cache["shape_e"],  # 这里用你 dilation 对应的 shape（你自己定）
                    cache["coeff"],  # 标量
                )

                # 2) accel (按你最新签名)
                Ar[i], Az[i] = mt.compute_accel_osbpd_axisym_csr_numba(
                    csr_indptr_m[i], csr_indices_m[i], csr_area_m[i],
                    csr_dist_m[i],
                    eij_edge,
                    n_r_edge, n_z_edge,
                    lamda_node_m[i],  # (N,)
                    miu_node_m[i],  # (N,)
                    coords_all_m_list[i][:, 0],  # r_node (N,)
                    Ur[i],
                    dilation,
                    materials_m[i]["rho_node"],  # (N,)
                    br[i], bz[i],  # (N,) 没有就传全0数组
                    zones[i]["delta"],
                    materials_m[i]["T_m"], Tpre_avg, kprime_node_m[i], materials_m[i]["csr"]
                )

            # ---------- 4) Calculate Fr_0 / Fz_0, λ, and half-step velocity based on acceleration ----------
            for i in range(n_slices):
                rho_node = materials_m[i]["rho_node"]  # (N,)
                Fr_0[i] = Ar[i] * rho_node
                Fz_0[i] = Az[i] * rho_node

                Vr_half[i] = 0.5 * (Fr_0[i] / lambda_diag_matrix[i])
                Vz_half[i] = 0.5 * (Fz_0[i] / lambda_diag_matrix[i])

                Ur[i] = Ur[i] + Vr_half[i] * dt_ADR
                Uz[i] = Uz[i] + Vz_half[i] * dt_ADR
                m = axis_mask_m[i]
                Ur[i][m] = 0.0

                # Ur[i][segment_slices_m[i]['left']] = 0.0

            for (region_id, direction), neighbor_data in boundary_neighbors_m.items():
                if direction not in ('top', 'bot'):
                    continue

                ghost_indices = neighbor_data["ghost_indices"]
                phys_indices = neighbor_data["phys_indices"]
                target_region = neighbor_data["target_region"]
                dists_list = boundary_distances_m.get((region_id, direction), None)

                dr1 = zones[region_id]["dr"]
                dr2 = zones[target_region]["dr"]

                # 片段
                sl_ghost = segment_slices_m[region_id][direction]  # ghost segment slice
                sl_phys = segment_slices_m[target_region]['phys']  # target phys slice

                # 位移片段
                Ur_g, Uz_g = Ur[region_id][sl_ghost], Uz[region_id][sl_ghost]
                Ur_p, Uz_p = Ur[target_region][sl_phys], Uz[target_region][sl_phys]
                # 速度片段
                Vr_half_g, Vz_half_g = Vr_half[region_id][sl_ghost], Vz_half[region_id][sl_ghost]
                Vr_half_p, Vz_half_p = Vr_half[target_region][sl_phys], Vz_half[target_region][sl_phys]

                if abs(dr1 - dr2) < tolerance:
                    # === same grid ===
                    Ur_new = bc.interpolate_temperature_for_same(Ur_g, Ur_p, ghost_indices, phys_indices)
                    Uz_new = bc.interpolate_temperature_for_same(Uz_g, Uz_p, ghost_indices, phys_indices)
                    Vr_half_new = bc.interpolate_temperature_for_same(Vr_half_g, Vr_half_p, ghost_indices, phys_indices)
                    Vz_half_new = bc.interpolate_temperature_for_same(Vz_half_g, Vz_half_p, ghost_indices, phys_indices)
                elif dr1 < dr2:
                    # region_id 更细（ghost 在细网格）→ 从粗网格插到细网格
                    Ur_new = bc.interpolate_temperature_for_coarse(Ur_g, Ur_p, ghost_indices, phys_indices,
                                                                   dists_list)
                    Uz_new = bc.interpolate_temperature_for_coarse(Uz_g, Uz_p, ghost_indices, phys_indices,
                                                                   dists_list)
                    Vr_half_new = bc.interpolate_temperature_for_coarse(Vr_half_g, Vr_half_p, ghost_indices,
                                                                        phys_indices,
                                                                        dists_list)
                    Vz_half_new = bc.interpolate_temperature_for_coarse(Vz_half_g, Vz_half_p, ghost_indices,
                                                                        phys_indices,
                                                                        dists_list)
                else:
                    # region_id 更粗（ghost 在粗网格）→ 从细网格插到粗网格
                    Ur_new = bc.interpolate_temperature_for_fine(Ur_g, Ur_p, ghost_indices, phys_indices)
                    Uz_new = bc.interpolate_temperature_for_fine(Uz_g, Uz_p, ghost_indices, phys_indices)
                    Vr_half_new = bc.interpolate_temperature_for_fine(Vr_half_g, Vr_half_p, ghost_indices, phys_indices)
                    Vz_half_new = bc.interpolate_temperature_for_fine(Vz_half_g, Vz_half_p, ghost_indices, phys_indices)

                Ur[region_id][sl_ghost] = Ur_new
                Uz[region_id][sl_ghost] = Uz_new
                Vr_half[region_id][sl_ghost] = Vr_half_new
                Vz_half[region_id][sl_ghost] = Vz_half_new

        else:
            # ✅ Reassemble after updating the boundaries.
            # --- 分段加载压力并强制每份收敛 ---
            if p > 0:
                ninc = number_pressure
            else:
                ninc = 1
                inc_p = 0.0

            for s in range(1, ninc + 1):
                # 本份累计到位后的目标压力
                p_s = p_pre + s * inc_p
                # 更新内壁面压力（只对各 slice 的内表面点施加）
                for i in range(n_slices):
                    idx_local = inner_surface_info_list[i]["indices"]
                    unit_vec = inner_surface_info_list[i]["unit_to_center"]  # 指向圆心的单位向量 [ur, uz]
                    # 注意：用“当前份的累计压力 p_s”来计算
                    br[i][idx_local] = p_s * unit_vec[:, 0]
                    bz[i][idx_local] = p_s * unit_vec[:, 1]

                # ---- 本份的非线性/迭代，必须收敛才允许进入下一份 ----
                converged_this_inc = False
                for step in range(nsteps_m):

                    # 记录上一步位移（用于收敛判据）
                    Ur_all_prev = np.concatenate([Ur[j] for j in range(n_slices)])
                    Uz_all_prev = np.concatenate([Uz[j] for j in range(n_slices)])

                    for i in range(n_slices):
                        cache = axisym_cache_m[i]  # 里面至少要有 r_flat,z_flat,edge_i,indices,dist_e,area_e,shape_e,coeff

                        Ur[i], Uz[i], Fr[i], Fz[i], Vr_half[i], Vz_half[i] = mt.compute_mechanical_step_csr(
                            csr_indptr_m[i], csr_indices_m[i], csr_dist_m[i], csr_area_m[i],
                            cache["r_flat"], cache["z_flat"], cache["edge_i"],
                            cache["shape_e"], cache["coeff"],  # shape_e 要用“机械场 rj/ri”那套
                            lamda_node_m[i], miu_node_m[i],
                            materials_m[i]["crack"], materials_m[i]["s0_edge"],
                            materials_m[i]["rho_node"],  # ✅ 按点密度数组
                            Ur[i], Uz[i], br[i], bz[i],
                            Fr_0[i], Fz_0[i],
                            Vr_half[i], Vz_half[i],
                            lambda_diag_matrix[i], dt_ADR,
                            zones[i]["delta"], cracking_act, materials_m[i]["T_m"], Tpre_avg, kprime_node_m[i],
                            materials_m[i]["csr"]
                        )

                        m = axis_mask_m[i]
                        Ur[i][m] = 0.0

                        # Ur[i][segment_slices_m[i]['left']] = 0.0

                    # 跨区域边界（仅顶/底）插值同步
                    for (region_id, direction), neighbor_data in boundary_neighbors_m.items():
                        if direction not in ('top', 'bot'):
                            continue

                        ghost_indices = neighbor_data["ghost_indices"]
                        phys_indices = neighbor_data["phys_indices"]
                        target_region = neighbor_data["target_region"]
                        dists_list = boundary_distances_m.get((region_id, direction), None)

                        dr1 = zones[region_id]["dr"]
                        dr2 = zones[target_region]["dr"]

                        # 片段
                        sl_ghost = segment_slices_m[region_id][direction]  # ghost segment slice
                        sl_phys = segment_slices_m[target_region]['phys']  # target phys slice

                        # 位移片段
                        Ur_g, Uz_g = Ur[region_id][sl_ghost], Uz[region_id][sl_ghost]
                        Ur_p, Uz_p = Ur[target_region][sl_phys], Uz[target_region][sl_phys]
                        # 速度片段
                        Vr_half_g, Vz_half_g = Vr_half[region_id][sl_ghost], Vz_half[region_id][sl_ghost]
                        Vr_half_p, Vz_half_p = Vr_half[target_region][sl_phys], Vz_half[target_region][sl_phys]

                        if abs(dr1 - dr2) < tolerance:
                            Ur_new = bc.interpolate_temperature_for_same(Ur_g, Ur_p, ghost_indices, phys_indices)
                            Uz_new = bc.interpolate_temperature_for_same(Uz_g, Uz_p, ghost_indices, phys_indices)
                            Vr_half_new = bc.interpolate_temperature_for_same(Vr_half_g, Vr_half_p, ghost_indices,
                                                                              phys_indices)
                            Vz_half_new = bc.interpolate_temperature_for_same(Vz_half_g, Vz_half_p, ghost_indices,
                                                                              phys_indices)
                        elif dr1 < dr2:
                            # ghost 在细网格：从粗插细
                            Ur_new = bc.interpolate_temperature_for_coarse(Ur_g, Ur_p, ghost_indices, phys_indices,
                                                                           dists_list)
                            Uz_new = bc.interpolate_temperature_for_coarse(Uz_g, Uz_p, ghost_indices, phys_indices,
                                                                           dists_list)
                            Vr_half_new = bc.interpolate_temperature_for_coarse(Vr_half_g, Vr_half_p, ghost_indices,
                                                                                phys_indices, dists_list)
                            Vz_half_new = bc.interpolate_temperature_for_coarse(Vz_half_g, Vz_half_p, ghost_indices,
                                                                                phys_indices, dists_list)
                        else:
                            # ghost 在粗网格：从细插粗
                            Ur_new = bc.interpolate_temperature_for_fine(Ur_g, Ur_p, ghost_indices, phys_indices)
                            Uz_new = bc.interpolate_temperature_for_fine(Uz_g, Uz_p, ghost_indices, phys_indices)
                            Vr_half_new = bc.interpolate_temperature_for_fine(Vr_half_g, Vr_half_p, ghost_indices,
                                                                              phys_indices)
                            Vz_half_new = bc.interpolate_temperature_for_fine(Vz_half_g, Vz_half_p, ghost_indices,
                                                                              phys_indices)
                        Ur[region_id][sl_ghost] = Ur_new
                        Uz[region_id][sl_ghost] = Uz_new
                        Vr_half[region_id][sl_ghost] = Vr_half_new
                        Vz_half[region_id][sl_ghost] = Vz_half_new
                        # ---- 收敛判据（RMS 位移增量）----
                    Ur_all_curr = np.concatenate([Ur[j] for j in range(n_slices)])
                    Uz_all_curr = np.concatenate([Uz[j] for j in range(n_slices)])

                    delta_Ur_all = Ur_all_curr - Ur_all_prev
                    delta_Uz_all = Uz_all_curr - Uz_all_prev
                    rms_increment = np.sqrt(np.mean(delta_Ur_all ** 2 + delta_Uz_all ** 2))
                    eps = 1e-30
                    phi_regions = {}
                    for i in range(n_slices):
                        edge_i = edge_i_m[i]  # (nnz,)
                        col_j = csr_indices_m[i]  # (nnz,)
                        area_e = csr_area_m[i].astype(np.float64)  # (nnz,)
                        mu_e = materials_m[i]["crack"].astype(np.float64)  # (nnz,)

                        N = len(csr_indptr_m[i]) - 1

                        # 如果 CSR 里存在对角项（i->i），把它剔除，等价于你原来的 fill_diagonal(A,0)
                        mask = (col_j != edge_i)

                        total_area = np.bincount(edge_i[mask], weights=area_e[mask], minlength=N)
                        intact_area = np.bincount(edge_i[mask], weights=area_e[mask] * mu_e[mask], minlength=N)

                        phi_i = 1.0 - intact_area / (total_area + eps)
                        phi_regions[i] = phi_i
                    phi_list = [phi_regions[i] for i in range(n_slices)]
                    for i, phi in enumerate(phi_list):
                        if np.any(np.asarray(phi) > 0.5):
                            has_frac = True
                            print(f"Region {i}: fracture, max(phi) = {np.max(phi):.6g}")

                    if rms_increment < rms:
                        print(
                            f"[Mechanical] converged at step {step} with RMS {rms_increment:.3e}, p = {p_s * dr1 :.6g}")
                        converged_this_inc = True
                        break

                    if step > 0 and step % 10 == 0:
                        print(f"[Mechanical] step {step}, RMS {rms_increment:.3e}")
                if has_frac:
                    break

    U_phys = {
        i: {
            "Ur": Ur[i][segment_slices_m[i]['phys']],
            "Uz": Uz[i][segment_slices_m[i]['phys']],
            "Umag": np.sqrt(
                Ur[i][segment_slices_m[i]['phys']] ** 2 +
                Uz[i][segment_slices_m[i]['phys']] ** 2
            )
        }
        for i in range(n_slices)

    }
    for i in range(n_slices):
        Ur_ = U_phys[i]["Ur"]
        Uz_ = U_phys[i]["Uz"]
        Umag_ok = np.hypot(Ur_, Uz_)  # 等价于 sqrt(Ur_**2 + Uz_**2)，数值更稳

        print("max Uz =", np.nanmax(np.abs(Uz_)))
        print("max |U| (recalc) =", np.nanmax(Umag_ok))
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['figure.titlesize'] = 10



    dr_min = min(z["dr"] for z in zones)
    plot.plot_temperature_contour_in_circle(
        phys_coords_list_t, dr_min, T_phys, cmap='viridis', radius=r,
    )
    plot.plot_displacement_contours_in_circle(
        phys_coords_list_m, U_phys, r, dr_min,
        titles=('Ur (m)', 'Uz (m)', 'Umag (m)'),
        levels=20
    )
    phi_regions = {}

    for i in range(n_slices):
        edge_i = edge_i_m[i]  # (nnz,)
        col_j = csr_indices_m[i]  # (nnz,)
        area_e = csr_area_m[i].astype(np.float64)  # (nnz,)
        mu_e = materials_m[i]["crack"].astype(np.float64)  # (nnz,)
        N = len(csr_indptr_m[i]) - 1
        # 如果 CSR 里存在对角项（i->i），把它剔除，等价于你原来的 fill_diagonal(A,0)
        mask = (col_j != edge_i)
        total_area = np.bincount(edge_i[mask], weights=area_e[mask], minlength=N)
        intact_area = np.bincount(edge_i[mask], weights=area_e[mask] * mu_e[mask], minlength=N)

        phi_i = 1.0 - intact_area / (total_area + 1e-30)
        phi_regions[i] = phi_i

    phi_list = [phi_regions[i] for i in range(n_slices)]
    # plot.plot_mu_field(r, coords_all_m_list, phi_list, grid_n=300, title_prefix="Core+Shell μ")

    end_time2 = time.time()
    print(f"Whole Calculation time = {end_time2 - start_time2:.2f}s")
    theory_time = np.linspace(0.0, total_time, len(time_history))

    ur_theory_histories = {pt["label"]: [] for pt in tracked_points}
    T_theory_histories = {pt["label"]: [] for pt in tracked_points}

    a = r
    center_r = 0.0
    center_z = r

    kappa_theory = ks / (rho_s * cs)
    alpha_theory = alpha_core
    nu_theory = nu_core

    for pt in tracked_points:
        r_pt = pt["coord"][0]
        z_pt = pt["coord"][1]

        for t_now in theory_time:
            rho_p = np.sqrt((r_pt - center_r) ** 2 + (z_pt - center_z) ** 2)

            T_th = pfc.temperature_sphere_dirichlet(
                np.array([rho_p]), t_now, a, kappa_theory, Tinit, Tsurr, n_terms=300
            )[0]

            ur_th = pfc.horizontal_disp_theory_at_point(
                r_pt, z_pt, t_now,
                a=a,
                center_r=center_r,
                center_z=center_z,
                kappa=kappa_theory,
                alpha=alpha_theory,
                nu=nu_theory,
                Tinit=Tinit,
                Tsurr=Tsurr,
                n_terms=300,
                n_r_int=3000
            )

            T_theory_histories[pt["label"]].append(T_th)
            ur_theory_histories[pt["label"]].append(ur_th)

    plt.figure(figsize=(5, 4))

    for pt in tracked_points:
        label = pt["label"]
        coord = pt["coord"]

        plt.plot(
            time_history,
            ur_histories[label],
            linewidth=1.8,
            label=f'Numerical {label}'
        )
        plt.plot(
            theory_time,
            ur_theory_histories[label],
            '--',
            linewidth=1.8,
            label=f'Analytical {label}'
        )

    plt.xlabel('Time (s)')
    plt.ylabel(r'Horizontal displacement $u_r$ (m)')
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))

    for pt in tracked_points:
        label = pt["label"]

        plt.plot(
            time_history,
            T_histories[label],
            linewidth=1.8,
            label=f'Numerical {label}'
        )
        plt.plot(
            theory_time,
            T_theory_histories[label],
            '--',
            linewidth=1.8,
            label=f'Analytical {label}'
        )

    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
