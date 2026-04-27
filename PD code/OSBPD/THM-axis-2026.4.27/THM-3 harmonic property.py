import generate_coordinates as gc
import ADR
import time
import Multiprocess_task_function as mt
import bc_funcs as bc
import core_funcs as cf
import Physical_Field_Calculation as pfc
import plot_utils as plot
import numpy as np
import matplotlib.pyplot as plt
import json
import os


rho_s, cs, ks = 7280.0, 230.0, 65
rho_l, cl, kl = 6800.0, 257.0, 31
rho_air, cair, kair = 1, 1010, 0.0263
rho_shell, c_shell, k_shell = 7020.0, 348.95, 40
nu_core, nu_shell = 0.33, 0.284  # Poisson's ratio
E_core, alpha_core_l, alpha_core_s = 44.3e9, 18e-6 , 18e-6
E_shell, alpha_shell = 222.72e9, 4e-6
sigmat = 803e6
G0 = 0.2  # J/m²
G1 = 100000  # J/m²

Ts = 498.65
Tl = 499.65
L = 60.627
number_pressure = 1
h = 1.
tolerance = 1e-14
Tsurr = 573.15
Tinit = 303.15
Tpre_avg = 303.15

""" Initialization of coarse regions and temperatures """
# dr1, dr2, dr3 are used to generate regional particle density by calling functions to create coordinates.

r = 24 * 1e-6  # Domain size in r and z directions (meters)
dshell = 2 * 1e-6
r_core = r - dshell
V_shell = 4 / 3 * np.pi * (r ** 3 - r_core ** 3)
size = 1e-6

only_one_slice = True
dr1 = 0.2 * size
dr2 = dr1
dr3 = 1 * dr2
dr_l = 0.6 * size
len1 = 12 * size
len2 = 12 * size
len3 = 12 * size
r_start = 0
ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction
n_slices = 4

"""Note that n_slices here refers to the minimum number of layers. 
Subsequent checks and modifications will be performed 
in the gc.compute_layer_dr_r_nr function to ensure that the granularity of each layer is reasonable."""

dt_ADR = 1
total_time = 2e-5
rms = 1e-11

graph_inner_shell = False
graph_point_t = False
graph_point_m = False

timemagnify = 1.5

# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
if __name__ == "__main__":
    # Shared zoning

    start_time2 = time.time()

    """1. Definition of regional coordinates and area matrix (shared for T and M)"""
    zones, n_slices = gc.compute_layer_dr_r_nr(
        r, n_slices, dr1, dr2, dr3, dr_l,
        len1, len2, len3, size, ghost_nodes_r,
        only_one_slice=only_one_slice,

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
            graph=graph_point_t,
            r_start=r_start
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
            graph=graph_point_m,
            r_start=r_start
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
        mask_phy_core = ((x_phy_t - r_start) ** 2 + (z_phy_t - r) ** 2) < r_core ** 2
        mask_core = ((x_t - r_start) ** 2 + (z_t - r) ** 2) < r_core ** 2
        mask_core_regions_t[i] = mask_core
        mask_core_regions_phy_t[i] = mask_phy_core


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
        mask_core = ((x_m - r_start) ** 2 + (z_m - r) ** 2) < r_core ** 2
        mask_core_regions_m[i] = mask_core

    """
      def plot_core_mask_one_slice(
              slice_id,
              phys_coords_list_m,
              ghost_coords_list_m,
              mask_core_regions_m,
              r,
              dshell,
              r_start=0.0,
              figsize=(7, 7)
      ):

          # 组合坐标
          coords_phys = phys_coords_list_m[slice_id][:, :2]
          coords_ghost = ghost_coords_list_m[slice_id][:, :2]
          coords_m = np.vstack([coords_phys, coords_ghost])

          mask_core = mask_core_regions_m[slice_id]
          n_phys = coords_phys.shape[0]

          # 分开物理点/ghost点对应的 mask
          mask_core_phys = mask_core[:n_phys]
          mask_core_ghost = mask_core[n_phys:]

          fig, ax = plt.subplots(figsize=figsize)

          # ---------- physical ----------
          ax.scatter(
              coords_phys[~mask_core_phys, 0], coords_phys[~mask_core_phys, 1],
              s=18, marker='o', label='phys non-core'
          )
          ax.scatter(
              coords_phys[mask_core_phys, 0], coords_phys[mask_core_phys, 1],
              s=18, marker='o', label='phys core'
          )

          # ---------- ghost ----------
          if len(coords_ghost) > 0:
              ax.scatter(
                  coords_ghost[~mask_core_ghost, 0], coords_ghost[~mask_core_ghost, 1],
                  s=28, marker='x', label='ghost non-core'
              )
              ax.scatter(
                  coords_ghost[mask_core_ghost, 0], coords_ghost[mask_core_ghost, 1],
                  s=28, marker='x', label='ghost core'
              )

          # ---------- theoretical boundaries ----------
          theta = np.linspace(0, 2 * np.pi, 400)
          r_core = r - dshell
          z_center = r

          x_outer = r_start + r * np.cos(theta)
          z_outer = z_center + r * np.sin(theta)

          x_core = r_start + r_core * np.cos(theta)
          z_core = z_center + r_core * np.sin(theta)

          ax.plot(x_outer, z_outer, '--', linewidth=1.5, label='outer boundary')
          ax.plot(x_core, z_core, '--', linewidth=1.5, label='core boundary')

          ax.set_aspect('equal')
          ax.set_xlabel('r')
          ax.set_ylabel('z')
          ax.set_title(f'Slice {slice_id}: core mask distribution')
          ax.legend()
          ax.grid(True)
          plt.tight_layout()
          plt.show()
      for i in range(n_slices):
          plot_core_mask_one_slice(
              slice_id=i,
              phys_coords_list_m=phys_coords_list_m,
              ghost_coords_list_m=ghost_coords_list_m,
              mask_core_regions_m=mask_core_regions_m,
              r=r,
              dshell=dshell,
              r_start=r_start
          )"""
    # ---- CSR containers ----
    csr_indptr_m = []
    csr_indices_m = []
    csr_dist_m = []
    csr_area_m = []
    weighted_volume_m = [None] * n_slices
    materials_m = {}
    edge_i_m = {}
    c_m = {}
    csr = {}
    nu_m = {}
    row_sum_area_m = {}

    coords_all_m_list = [np.vstack([phys_coords_list_m[i], ghost_coords_list_m[i]]) for i in range(n_slices)]
    coords_all_t_list = [np.vstack([phys_coords_list_t[i], ghost_coords_list_t[i]]) for i in range(n_slices)]
    results_m = [mt.compute_region_matrices(args) for args in task_args_m]

    lamda_node_m = [None] * n_slices
    miu_node_m = [None] * n_slices  # ✅ 新增：存 μ_node
    gama_node_m = [None] * n_slices
    kprime_node_m = [None] * n_slices

    E_edge_m = {}
    nu_edge_m = {}
    alpha_edge_m = {}
    miu_edge_m = {}
    lamda_edge_m = {}
    kprime_edge_m = {}
    L_core_edge_m = {}
    L_shell_edge_m = {}
    L_total_edge_m = {}

    E_node_corr_m = {}
    nu_node_corr_m = {}

    for i, (indptr, indices, dist, area) in enumerate(results_m):
        csr_indptr_m.append(indptr)
        csr_indices_m.append(indices)
        csr_dist_m.append(dist)
        csr_area_m.append(area)

        coords_all_m = coords_all_m_list[i]
        coords_all_t = coords_all_t_list[i]

        mask_core_i = mask_core_regions_m[i]
        r_node = coords_all_m[:, 0].astype(np.float64)
        # === Calculation of non-local mechanical parameters ===
        # --- 2) CSR（必须用机械场的 CSR，不要用 _t）---
        indptr = csr_indptr_m[i]
        indices = csr_indices_m[i]
        area = csr_area_m[i]  # (nnz,)
        N = len(indptr) - 1
        nnz = len(indices)
        eps = 1e-20

        edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
        edge_i_m[i] = edge_i

        L_core_edge, L_shell_edge, L_total_edge = pfc.precompute_edge_core_shell_lengths(
            coords_all=coords_all_m[:, :2],
            edge_i=edge_i,
            edge_j=indices,
            r_core=r_core,
            r_center=r_start,
            z_center=r
        )

        L_core_edge_m[i] = L_core_edge
        L_shell_edge_m[i] = L_shell_edge
        L_total_edge_m[i] = L_total_edge

        nu_node = np.where(mask_core_i, nu_core, nu_shell).astype(np.float64)
        E_node = np.where(mask_core_i, E_core, E_shell).astype(np.float64)
        alpha_node = np.where(mask_core_i, alpha_core_s, alpha_shell).astype(np.float64)
        # --- node-wise Lamé parameters (for br_geom term) ---
        lamda0_node = (E_node * nu_node) / ((1.0 - 2.0 * nu_node) * (1.0 + nu_node) + eps)
        miu_node = E_node / (2.0 * (1.0 + nu_node) + eps)
        lamda_node = lamda0_node - miu_node

        miu_node_m[i] = miu_node
        lamda_node_m[i] = lamda_node
        kprime_node_m[i] = (E_node / (1.0 - 2.0 * nu_node)) * alpha_node  # (N,)

        # --- 按键长修正后的 edge 属性 ---
        E_edge = pfc.build_edge_property_harmonic_from_lengths(
            prop_node=E_node,
            is_core_node=mask_core_i,
            edge_i=edge_i,
            edge_j=indices,
            L_core_edge=L_core_edge_m[i],
            L_shell_edge=L_shell_edge_m[i],
            L_total_edge=L_total_edge_m[i],
            r_node=r_node,
            mode="harm"
        )

        nu_edge = pfc.build_edge_property_harmonic_from_lengths(
            prop_node=nu_node,
            is_core_node=mask_core_i,
            edge_i=edge_i,
            edge_j=indices,
            L_core_edge=L_core_edge_m[i],
            L_shell_edge=L_shell_edge_m[i],
            L_total_edge=L_total_edge_m[i],
            r_node=r_node,
            mode="harm",

        )

        alpha_edge = pfc.build_edge_property_harmonic_from_lengths(
            prop_node=alpha_node,
            is_core_node=mask_core_i,
            edge_i=edge_i,
            edge_j=indices,
            L_core_edge=L_core_edge_m[i],
            L_shell_edge=L_shell_edge_m[i],
            L_total_edge=L_total_edge_m[i],
            r_node=r_node,
            mode="harm",

        )

        # --- 由 edge 属性继续构造 edge 力学参数 ---
        lamda0_edge = (E_edge * nu_edge) / ((1.0 - 2.0 * nu_edge) * (1.0 + nu_edge) + eps)
        miu_edge = E_edge / (2.0 * (1.0 + nu_edge) + eps)
        lamda_edge = lamda0_edge - miu_edge
        kprime_edge = (E_edge / (1.0 - 2.0 * nu_edge + eps)) * alpha_edge

        E_edge_m[i] = E_edge
        nu_edge_m[i] = nu_edge
        alpha_edge_m[i] = alpha_edge
        miu_edge_m[i] = miu_edge
        lamda_edge_m[i] = lamda_edge
        kprime_edge_m[i] = kprime_edge

        delta_i = zones[i]["delta"]

        # Use the axisymmetric measure A_ij * (r_j / r_i) instead of planar area
        # so the CSR normalization is consistent with the mechanical bond weights.
        row_sum_area = np.bincount(edge_i, weights=area, minlength=N).astype(np.float64)
        denom = row_sum_area[edge_i] + row_sum_area[indices]
        csr[i] = (2.0 * np.pi * delta_i ** 2 / denom)

        # ---- c_edge 已经算完：现在把 self-area 置 0（仅影响后续使用 area 的地方）----
        mask_self = (indices == edge_i)
        if np.any(mask_self):
            area = area.copy()
            area[mask_self] = 0.0
            csr_area_m[i] = area  # 更新列表里保存的 area（后续都用“self=0”的版本）
        # ==========================

    """
        def plot_cross_interface_bonds(
                slice_id,
                coords_all_m_list,
                edge_i_m,
                csr_indices_m,
                L_core_edge_m,
                L_shell_edge_m,
                r,
                dshell,
                r_start=0.0,
                max_bonds=3000
        ):
            coords = coords_all_m_list[slice_id][:, :2]
            ei = edge_i_m[slice_id]
            ej = csr_indices_m[slice_id]
            Lc = L_core_edge_m[slice_id]
            Ls = L_shell_edge_m[slice_id]

            # 三类 bond
            mask_core_only = (Lc > 1e-14) & (Ls <= 1e-14)
            mask_shell_only = (Ls > 1e-14) & (Lc <= 1e-14)
            mask_cross = (Lc > 1e-14) & (Ls > 1e-14)

            fig, ax = plt.subplots(figsize=(7, 7))

            def draw_bonds(mask, label, lw=0.5, alpha=0.5):
                idx = np.where(mask)[0]
                if len(idx) > max_bonds:
                    idx = np.random.choice(idx, max_bonds, replace=False)
                for k in idx:
                    i = ei[k]
                    j = ej[k]
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        linewidth=lw,
                        alpha=alpha,
                        label=label if k == idx[0] else None
                    )

            draw_bonds(mask_core_only, "core-only bond", lw=0.4, alpha=0.25)
            draw_bonds(mask_shell_only, "shell-only bond", lw=0.4, alpha=0.25)
            draw_bonds(mask_cross, "cross-interface bond", lw=0.8, alpha=0.8)

            # 理论边界
            theta = np.linspace(0, 2 * np.pi, 400)
            z_center = r
            r_core = r - dshell
            ax.plot(r_start + r * np.cos(theta), z_center + r * np.sin(theta), '--', linewidth=1.2, label='outer boundary')
            ax.plot(r_start + r_core * np.cos(theta), z_center + r_core * np.sin(theta), '--', linewidth=1.2,
                    label='core boundary')

            ax.set_aspect('equal')
            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.set_title(f'Slice {slice_id}: bond classification')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()


        for i in range(n_slices):
            plot_cross_interface_bonds(
                slice_id=i,
                coords_all_m_list=coords_all_m_list,
                edge_i_m=edge_i_m,
                csr_indices_m=csr_indices_m,
                L_core_edge_m=L_core_edge_m,
                L_shell_edge_m=L_shell_edge_m,
                r=r,
                dshell=dshell,
                r_start=r_start
            )"""
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

        coords_ghost_right_t = ghost_right_t[i][:, :2]
        coords_phys_t = phys_t[i][:, :2]
        dr = zones[i]["dr"]
        parts = []
        for a in (phys_coords_list_t[i], ghost_left_t[i], ghost_top_t[i], ghost_bot_t[i]):
            A = np.asarray(a)
            A = A[:, :2]  # 统一只取前两列
            parts.append(A)

        coords_t_pltb = np.vstack(parts) if parts else np.empty((0, 2), float)


        ghost_idx_right_t, phys_idx_right_t = bc.find_circle_mirror_pairs_multilayer(
            coords_ghost_right_t, coords_t_pltb, dr, r, r_start=r_start
        )
        boundary_neighbors_t[(i, 'right')] = {
            "ghost_indices": ghost_idx_right_t,
            "phys_indices": phys_idx_right_t,
            "target_region": i
        }



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
    T_increment = Tinit + (Tsurr - Tinit) / nsteps_th * 2
    for i in range(n_slices):
        threshold_distance = np.sqrt(2) * zones[i]["dr"] + tolerance
        factor_data_t[i] = np.where(
            csr_dist_t[i] <= threshold_distance,
            1.125,
            1.0
        ).astype(np.float64)
        T_phys[i] = np.full(phys_t[i].shape[0], Tinit)
        T_left[i] = np.full(ghost_left_t[i].shape[0], Tinit)
        T_right[i] = np.full(ghost_right_t[i].shape[0], Tsurr)
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
            T_right[region_id][ghost_indices] = 2 * T_increment  - T1[phys_indices]
            T[region_id] = np.concatenate(
                [T_phys[region_id], T_left[region_id], T_right[region_id], T_top[region_id], T_bot[region_id]])
            # 如果你以后要用镜像/反射形式，就写：
            # T_right[region_id][ghost_indices] = 2*T_increment - T_phys[region_id][phys_indices]

    """4. Definition of enthalpy for temperature field"""
    H = {}
    K = {}
    segment_slices_t = {}

    """4.1 空腔属性的定义"""
    rho_void = 0
    Cp_void = 0
    k_void = 0
    mask_void_regions_t = {}

    """4.2 焓及轴对称因子的初始定义"""
    shape_factor_t = {}
    for i in range(n_slices):
        H[i] = cf.get_enthalpy(
            T[i],
            mask_core_regions_t[i],
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
    alpha_node_field = {}
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
        T_m = T[i][CorrList_T[i]]
        # 或者直接 materials_m[i]["T_m"] = ...

        mask_core_i = mask_core_regions_m[i]


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
        crack = np.ones(nnz, dtype=np.int8)

        # --- alpha ---
        # --- s0 ---
        delta_i = zones[i]["delta"]

        s0_core = 10
        s0_shell = sigmat / E_shell
        s0_node = np.where(mask_core_i, s0_core, s0_shell).astype(np.float64)
        s0_edge = 0.5 * (s0_node[edge_i] + s0_node[indices])  # (nnz,)

        rho_node, rh_edge = cf.get_density(
            T_m, mask_core_i,
            rho_s, rho_l, Ts, Tl,
            rho_shell, rho_void,
            edge_i, edge_j
        )

        dir_r_m, dir_z_m = pfc.compute_direction_edges_csr_numba(
            coords_all_m_list[i], edge_i, edge_j, dist
        )
        dx0_edge_m.append(dir_r_m * dist)
        dz0_edge_m.append(dir_z_m * dist)

        # --- storage ---
        materials_m[i] = {
            "T_m": T_m,  # (N,)
            "csr": csr[i],
            "edge_i": edge_i_m[i],  # (N,)
            "s0_edge": s0_edge,  # (nnz,)
            "crack": crack,  # (nnz,)
            "rho_node": rho_node,
            "dir_r_edge": dir_r_m,
            "dir_z_edge": dir_z_m,
        }

        pre_m[i] = {
            "mask_core": mask_core_i,
            "edge_i": edge_i,
            "edge_j": edge_j,
            "s0_edge": s0_edge,
            "crack": crack,
            "dir_r_edge": dir_r_m,
            "dir_z_edge": dir_z_m,
        }

        dr_i = zones[i]["dr"]
        surface_info = pfc.find_inner_surface_layer(
            coords_all_m,
            r, dshell, dr_i,
            r_center=r_start
        )
        inner_surface_info_list.append(surface_info)


        Ur[i] = np.zeros(coords_all_m.shape[0])
        Uz[i] = np.zeros(coords_all_m.shape[0])
        Ar[i] = np.zeros(coords_all_m.shape[0])
        Az[i] = np.zeros(coords_all_m.shape[0])
        br[i] = np.zeros(coords_all_m.shape[0])
        bz[i] = np.zeros(coords_all_m.shape[0])

    for i in range(n_slices):
        T_m_i = materials_m[i]["T_m"]
        mask_core_i = mask_core_regions_m[i]
        indptr = csr_indptr_m[i]
        indices = csr_indices_m[i]
        area = csr_area_m[i]  # (nnz,)
        N = len(indptr) - 1
        nnz = len(indices)
        eps = 1e-20

        edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
        edge_i_m[i] = edge_i

        # 1) 温度对应的原始 alpha
        alpha_node = np.where(
            mask_core_i,
            np.where(T_m_i >= Tl, alpha_core_l, alpha_core_s),
            alpha_shell
        ).astype(np.float64)

        nu_node = np.where(mask_core_i, nu_core, nu_shell).astype(np.float64)
        E_node = np.where(mask_core_i, E_core, E_shell).astype(np.float64)
        core_mask = mask_core_i

        kprime_node_m[i][core_mask] = (E_node[core_mask] / (1.0 - 2.0 * nu_node[core_mask])) * alpha_node[core_mask]
        r_node = coords_all_m[:, 0].astype(np.float64)
        alpha_edge = pfc.build_edge_property_harmonic_from_lengths(
            prop_node=alpha_node,
            is_core_node=mask_core_i,
            edge_i=edge_i,
            edge_j=indices,
            L_core_edge=L_core_edge_m[i],
            L_shell_edge=L_shell_edge_m[i],
            L_total_edge=L_total_edge_m[i],
            r_node=r_node,
            mode="harm",

        )
        alpha_edge_m[i] = alpha_edge
        kprime_edge_m[i] = (E_edge_m[i] / (1.0 - 2.0 * nu_edge_m[i] + 1e-30)) * alpha_edge_m[i]

    # 循环外先定义
    axisym_cache_m = [None] * n_slices
    axis_mask_m = []
    # 定义左边界
    for i in range(n_slices):
        coords_all_m = coords_all_m_list[i]
        r_all = coords_all_m[:, 0]
        dr = zones[i]["dr"]
        eps = 1e-20
        axis_mask = (r_all > r_start) & (r_all < r_start + 1 * dr - eps)  # 0<r<dr
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
            indptr=indptr,
            indices=indices,
            area_edge=area,
            dist_edge=dist,
            w_maxe_edge=w_maxe_edge,
            shape_edge=shape_edge,
            lamda_edge=lamda_edge_m[i],
            miu_edge=miu_edge_m[i],
            r_node=r_flat,
            delta=delta,
            q=q
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
    # 在每个时间步里
    rho_void_old, Cp_void_old = rho_air, cair
    rho_void_new, Cp_void_new = rho_air, cair
    has_frac = False
    cracking_act = False
    T_right_pre = [None] * n_slices

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
        ("(r/2, r)", r / 2 + r_start, r),("(rshell, r)", r  + r_start - dshell, r),
        ("(r, r)", r + r_start, r), ("(r/3, r)", r /3, r)
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


    for step1 in range(nsteps_th):
        for i in range(n_slices):
            K_data, diag = cf.build_Kdata_and_rowsum_csr_numba(
                T[i], mask_core_regions_t[i],
                factor_data_t[i], csr_area_t[i], shape_factor_t[i], csr_dist_t[i],
                csr_indptr_t[i], csr_indices_t[i],
                ks, kl, Ts, Tl,
                k_shell, zones[i]["delta"],
                k_void,
                dt_th
            )

            dH = cf.apply_K_with_diag_csr_numba(
                csr_indptr_t[i], csr_indices_t[i], K_data, diag, T[i],
                rho_void_old, Cp_void_old,
                rho_void_new, Cp_void_new
            )
            H[i] = H[i] + dH
            # ② 直接调用：H -> T
            T[i] = cf.temperature_from_enthalpy_numba(
                H[i],
                mask_core_regions_t[i],
                rho_s, rho_l, cs, cl, L, Ts, Tl,
                rho_shell, c_shell,
                rho_void, Cp_void
            )
        # ✅ Split results
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

        for i in range(n_slices):
            sl = segment_slices_t[i]
            T_phys[i] = T[i][sl["phys"]]
            T_left[i] = T[i][sl["left"]]
            T_right[i] = T[i][sl["right"]]
            T_top[i] = T[i][sl["top"]]
            T_bot[i] = T[i][sl["bot"]]

        # ✅ Unified handling of boundary conditions
        T_increment = min(Tinit + (Tsurr - Tinit) * (step1 + 2) / nsteps_th * 2, Tsurr)
        print(f"T_increment = {T_increment}")
        print(f"Completed {step1 + 1}/{nsteps_th} steps of calculation")

        for (region_id, direction), neighbor_data in boundary_neighbors_t.items():
            if direction == 'right':
                T1 = np.concatenate(
                    [T_phys[region_id], T_left[region_id], T_top[region_id], T_bot[region_id]])
                ghost_indices = neighbor_data["ghost_indices"]
                phys_indices = neighbor_data["phys_indices"]  # 这些索引是相对于 phys_coords 的
                T_right[region_id][ghost_indices] = 2 * T_increment - T1[phys_indices]
                T[region_id] = np.concatenate(
                    [T_phys[region_id], T_left[region_id], T_right[region_id], T_top[region_id], T_bot[region_id]])

        rho_void_old = rho_void
        Cp_void_old = Cp_void

        rho_void = 0
        Cp_void = 0
        k_void = 0

        rho_void_new = rho_void
        Cp_void_new = Cp_void

        for i in range(n_slices):
            # --- 1) 更新 T_m ---
            T_m = T[i][CorrList_T[i]]
            materials_m[i]["T_m"] = T_m

            # 取初始化预存的常量/拓扑
            edge_i = pre_m[i]["edge_i"]
            edge_j = pre_m[i]["edge_j"]
            mask_core_i = pre_m[i]["mask_core"]


            # --- 3) 重新计算 rho_node / rho_edge ---
            rho_node, rho_edge = cf.get_density(
                T_m, mask_core_i,
                rho_s, rho_l, Ts, Tl,
                rho_shell, rho_void,
                edge_i, edge_j
            )
            materials_m[i]["rho_node"] = rho_node

        for i in range(n_slices):
            T_m_i = materials_m[i]["T_m"]
            mask_core_i = mask_core_regions_m[i]
            indptr = csr_indptr_m[i]
            indices = csr_indices_m[i]
            area = csr_area_m[i]  # (nnz,)
            N = len(indptr) - 1
            nnz = len(indices)
            eps = 1e-20

            edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
            edge_i_m[i] = edge_i

            # 1) 温度对应的原始 alpha
            alpha_node = np.where(
                mask_core_i,
                np.where(T_m_i >= Tl, alpha_core_l, alpha_core_s),
                alpha_shell
            ).astype(np.float64)

            nu_node = np.where(mask_core_i, nu_core, nu_shell).astype(np.float64)
            E_node = np.where(mask_core_i, E_core, E_shell).astype(np.float64)
            core_mask = mask_core_i
            kprime_node_m[i][core_mask] = (E_node[core_mask] / (1.0 - 2.0 * nu_node[core_mask])) * alpha_node[core_mask]

            alpha_edge = pfc.build_edge_property_harmonic_from_lengths(
                prop_node=alpha_node,
                is_core_node=mask_core_i,
                edge_i=edge_i,
                edge_j=indices,
                L_core_edge=L_core_edge_m[i],
                L_shell_edge=L_shell_edge_m[i],
                L_total_edge=L_total_edge_m[i],
                r_node=r_node,
                mode="harm",

            )
            alpha_edge_m[i] = alpha_edge
            kprime_edge_m[i] = (E_edge_m[i] / (1.0 - 2.0 * nu_edge_m[i] + 1e-30)) * alpha_edge_m[i]

        """6. Definition of physical strength: Note that the average value of the elastic modulus is taken here."""

        phi_regions = {}

        if step1 == 0:
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
                    cache["coeff"], csr[i]  # 标量
                )

                # 2) accel (按你最新签名)
                Ar[i], Az[i] = mt.compute_accel_osbpd_axisym_csr_numba(
                    csr_indptr_m[i], csr_indices_m[i], csr_area_m[i],
                    csr_dist_m[i],
                    eij_edge,
                    n_r_edge, n_z_edge,

                    lamda_edge_m[i],
                    miu_edge_m[i],
                    lamda_node_m[i],
                    miu_node_m[i],

                    coords_all_m_list[i][:, 0],
                    Ur[i],
                    dilation,
                    materials_m[i]["rho_node"],
                    br[i], bz[i],
                    zones[i]["delta"],
                    materials_m[i]["T_m"], Tpre_avg,
                    kprime_edge_m[i],
                    kprime_node_m[i],
                    csr[i]
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

        else:
            for step in range(nsteps_m):
                # 记录上一步位移（用于收敛判据）
                Ur_all_prev = np.concatenate([Ur[j] for j in range(n_slices)])
                Uz_all_prev = np.concatenate([Uz[j] for j in range(n_slices)])

                for i in range(n_slices):
                    cache = axisym_cache_m[i]  # 里面至少要有 r_flat,z_flat,edge_i,indices,dist_e,area_e,shape_e,coeff

                    Ur[i], Uz[i], Fr[i], Fz[i], Vr_half[i], Vz_half[i] = mt.compute_mechanical_step_csr(
                        csr_indptr_m[i], csr_indices_m[i], csr_dist_m[i], csr_area_m[i],
                        cache["r_flat"], cache["z_flat"], cache["edge_i"],
                        cache["shape_e"], cache["coeff"],

                        lamda_edge_m[i], miu_edge_m[i],
                        lamda_node_m[i], miu_node_m[i],

                        materials_m[i]["crack"], materials_m[i]["s0_edge"],
                        materials_m[i]["rho_node"],
                        Ur[i], Uz[i], br[i], bz[i],
                        Fr_0[i], Fz_0[i],
                        Vr_half[i], Vz_half[i],
                        lambda_diag_matrix[i], dt_ADR,
                        zones[i]["delta"], cracking_act, materials_m[i]["T_m"], Tpre_avg,

                        kprime_edge_m[i],
                        kprime_node_m[i],
                        alpha_edge_m[i],
                        materials_m[i]["csr"]
                    )
                    Fr_0[i] = Fr[i].copy()
                    Fz_0[i] = Fz[i].copy()
                    m = axis_mask_m[i]
                    Ur[i][m] = 0.0


                Ur_all_curr = np.concatenate([Ur[j] for j in range(n_slices)])
                Uz_all_curr = np.concatenate([Uz[j] for j in range(n_slices)])
                delta_Ur_all = Ur_all_curr - Ur_all_prev
                delta_Uz_all = Uz_all_curr - Uz_all_prev
                rms_increment = np.sqrt(np.mean(delta_Ur_all ** 2 + delta_Uz_all ** 2))

                if rms_increment < rms:
                    print(
                        f"[Mechanical] converged at step {step} with RMS {rms_increment:.3e}, p = {p:.6g}")
                    converged_this_inc = True
                    break

                if step > 0 and step % 10 == 0:
                    print(f"[Mechanical] step {step}, RMS {rms_increment:.3e}")

        current_time = (step1 + 1) * dt_th
        time_history.append(current_time)

        for pt in tracked_points:
            i_slice = pt["slice"]
            i_local = pt["local_idx"]

            current_ur = Ur[i_slice][i_local]
            current_T = materials_m[i_slice]["T_m"][i_local]

            ur_histories[pt["label"]].append(current_ur)
            T_histories[pt["label"]].append(current_T)

    U_phys = {}
    max_ur = -np.inf
    min_ur = np.inf
    max_uz = -np.inf
    min_uz = np.inf
    max_abs_ur = 0.0
    max_abs_uz = 0.0
    for i in range(n_slices):
        n_phys = phys_coords_list_m[i].shape[0]
        ur_phys = Ur[i][:n_phys]
        uz_phys = Uz[i][:n_phys]
        U_phys[i] = {
            "Ur": ur_phys,
            "Uz": uz_phys,
            "Umag": np.sqrt(ur_phys ** 2 + uz_phys ** 2)
        }
        max_ur = max(max_ur, float(np.nanmax(ur_phys)))
        min_ur = min(min_ur, float(np.nanmin(ur_phys)))
        max_uz = max(max_uz, float(np.nanmax(uz_phys)))
        min_uz = min(min_uz, float(np.nanmin(uz_phys)))
        max_abs_ur = max(max_abs_ur, float(np.nanmax(np.abs(ur_phys))))
        max_abs_uz = max(max_abs_uz, float(np.nanmax(np.abs(uz_phys))))

    dr_min = min(z["dr"] for z in zones)
    skip_plots = os.environ.get("THM_SKIP_PLOTS", "0") == "1"
    skip_tracked_csv = os.environ.get("THM_SKIP_TRACKED_CSV", "0") == "1"
    if not skip_plots:
        plot.plot_temperature_contour_in_circle(
            phys_coords_list_t, dr_min, T_phys, cmap='viridis', radius=r, r_start=r_start
        )
        plot.plot_displacement_contours_in_circle(
            phys_coords_list_m, U_phys, r, dr_min, r_start,
            titles=('Ur (m)', 'Uz (m)', 'Umag (m)'),
            levels=15
        )

    end_time2 = time.time()

    primary_zone = zones[0] if zones else None
    effective_delta_factor = None
    effective_delta = None
    if primary_zone is not None:
        effective_delta = float(primary_zone["delta"])
        if primary_zone["dr"] != 0:
            effective_delta_factor = float(primary_zone["delta"] / primary_zone["dr"])

    metrics = {
        "dr1": float(dr1),
        "size": float(size),
        "ghost_nodes_r": int(ghost_nodes_r),
        "delta_factor": effective_delta_factor,
        "delta": effective_delta,
        "delta_over_dx": effective_delta_factor,
        "max_ur": float(max_ur),
        "min_ur": float(min_ur),
        "max_uz": float(max_uz),
        "min_uz": float(min_uz),
        "max_abs_ur": float(max_abs_ur),
        "max_abs_uz": float(max_abs_uz),
        "nsteps_th": int(nsteps_th),
        "dt_th": float(dt_th),
        "n_slices": int(n_slices),
        "elapsed_s": float(end_time2 - start_time2),
    }

    metrics_json_path = os.environ.get("THM_METRICS_JSON_PATH")
    if metrics_json_path:
        metrics_dir = os.path.dirname(metrics_json_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if skip_plots and skip_tracked_csv:
        raise SystemExit(0)

    import pandas as pd

    # ========= 保存到桌面：一个文件 =========
    file_path = os.environ.get("THM_TRACKED_CSV_PATH")
    if not file_path:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        os.makedirs(desktop_path, exist_ok=True)
        file_path = os.path.join(desktop_path, "tracked_points_history_all_in_one.csv")
    else:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # 基础列：时间
    all_data = {
        "Time (s)": time_history
    }

    # 每个点的信息 + 对应时程
    for pt in tracked_points:
        label = pt["label"]
        safe_label = label.replace("(", "").replace(")", "").replace(",", "_").replace("/", "_").replace("\\",
                                                                                                         "_").replace(
            " ", "")

        # 点的固定信息
        all_data[f"{safe_label}_target_r"] = [pt["target_r"]] * len(time_history)
        all_data[f"{safe_label}_target_z"] = [pt["target_z"]] * len(time_history)
        all_data[f"{safe_label}_slice"] = [pt["slice"]] * len(time_history)
        all_data[f"{safe_label}_local_idx"] = [pt["local_idx"]] * len(time_history)
        all_data[f"{safe_label}_coord_r"] = [pt["coord"][0] if pt["coord"] is not None else None] * len(time_history)
        all_data[f"{safe_label}_coord_z"] = [pt["coord"][1] if pt["coord"] is not None else None] * len(time_history)

        # 时程数据
        all_data[f"{safe_label}_Ur (m)"] = ur_histories[label]
        all_data[f"{safe_label}_T (K)"] = T_histories[label]

    # 转成 DataFrame
    df_all = pd.DataFrame(all_data)

    # 保存
    if not skip_tracked_csv:
        df_all.to_csv(file_path, index=False, encoding="utf-8-sig")

        print(f"All data have been saved to one file:")
        print(file_path)

        ur_cols = [col for col in df_all.columns if col.endswith("_Ur (m)")]
        if ur_cols:
            max_tracked_ur = float(df_all[ur_cols].to_numpy().max())
            print(f"max tracked Ur = {max_tracked_ur:.12e}")

    if skip_plots:
        raise SystemExit(0)

    import matplotlib.pyplot as plt

    # ========= 读取文件 =========
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "tracked_points_history_all_in_one.csv")

    df = pd.read_csv(file_path)

    # ========= 时间列 =========
    time = df["Time (s)"]

    # ========= 自动查找所有位移列 =========
    ur_cols = [col for col in df.columns if col.endswith("_Ur (m)")]

    print("Found displacement columns:")
    for c in ur_cols:
        print(c)

    # ========= 画图 =========
    plt.figure(figsize=(8, 5))

    for col in ur_cols:
        # 图例名字：去掉结尾的 "_Ur (m)"
        label = col.replace("_Ur (m)", "")
        plt.plot(time, df[col], label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement Ur (m)")
    plt.title("Displacement time histories of tracked points")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
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
    theory_time = np.linspace(0.0, total_time, len(time_history))

    ur_theory_histories = {pt["label"]: [] for pt in tracked_points}
    T_theory_histories = {pt["label"]: [] for pt in tracked_points}

    a = r
    center_r = 0.0
    center_z = r

    kappa_theory = ks / (rho_s * cs)
    alpha_theory = alpha_core_l
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
    plt.show()    """
