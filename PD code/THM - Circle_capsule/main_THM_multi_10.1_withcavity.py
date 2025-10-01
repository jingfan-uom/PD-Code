import generate_coordinates as gc
import ADR
from multiprocessing import Pool
import time
import Multiprocess_task_function as mt
import bc_funcs as bc
import core_funcs as cf
import Physical_Field_Calculation as pfc
import plot_utils as plot
import numpy as np

rho_s, cs, ks = 7280.0, 230.0, 65
rho_l, cl, kl = 6800.0, 257.0, 31
rho_air, cair, kair = 1, 1010, 0.0263
rho_shell, c_shell, k_shell = 7020.0, 348.95, 40
nu_core, nu_shell = 0.25, 0.25  # Poisson's ratio
E_core, alpha_core = 43.3e9, 20e-6
E_shell, alpha_shell = 222.72e9, 4e-6
sigmat = 803e6
G0 = 0.2   # J/m²
G1 = 10000000000   # J/m²

Ts = 498.65
Tl = 499.65
L = 60.627

h = 1.0
tolerance = 1e-14
Tsurr = 573.15
Tinit = 303.15
Tpre_avg = 303.15
dT_step = 20.0
""" Initialization of coarse regions and temperatures """
# dr1, dr2, dr3 are used to generate regional particle density by calling functions to create coordinates.
size = 1e-6
r = 25 * size  # Domain size in r and z directions (meters)
dshell = 1 * size
r_core = r - dshell
V0_all = np.pi * r_core ** 2 / 2

dr1 = 0.5 * size
dr2 = 0.5 * size
dr3 = 1 * dr2
dr_l = 0.5 * size

len1 = 5 * size
len2 = 5 * size
len3 = 5 * size

ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction
n_slices = 10

"""Note that n_slices here refers to the minimum number of layers. 
Subsequent checks and modifications will be performed 
in the gc.compute_layer_dr_r_nr function to ensure that the granularity of each layer is reasonable."""
num_processes = 1
dt_ADR = 1
total_time = 1e-7
rms = 1e-10
f_void = 0.0
number_pressure = 1
graph_inner_shell = False
graph_point_t = False
graph_point_m = False
# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
if __name__ == "__main__":
    # Shared zoning
    z0 = pfc.void_z(f_void, r)
    start_time2 = time.time()
    """1. Definition of regional coordinates and area matrix (shared for T and M)"""
    # (a)  E and alpha
    # T_E = [273.15, 373.15, 473.15, 573.15, 673.15, 773.15, 873.15, 973.15, 1073.15, 1173.15, 1273.15, 1373.15, 1473.15]
    # E_vals = [70e9, 70e9, 70e9, 70e9, 70e9, 70e9, 70e9, 199.922e9, 197.124e9,
    #           195.026e9, 194.793e9, 192.461e9, 190.13e9]  # GPa
    # T_alpha = [273.15, 373.15, 473.15, 573.15, 673.15, 773.15, 873.15, 973.15, 1073.15, 1173.15, 1273.15, 1373.15,
    #            1473.15]
    # alpha_vals = [8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6, 8.2e-6,
    #               8.2e-6]  # 1e-6 /K
    #
    # # (b)  lambda and Cp
    # T_lambda = [573.15, 673.15, 873.15, 1073.15, 1273.15, 1473.15, 1773.15]
    # lambda_vals = [15, 15, 15, 15, 15, 15, 15]  # W/mK
    # T_Cp = [573.15, 673.15, 873.15, 1073.15, 1273.15, 1473.15, 1773.15]
    # Cp_vals = [1100, 1100, 1100, 1100, 1100, 1100, 1100]  # J/kgK
    T_E = [273.15, 373.15, 473.15, 573.15, 673.15]
    E_vals = [E_shell, E_shell, E_shell,E_shell, E_shell]  # GPa
    T_alpha = [273.15, 373.15, 473.15, 573.15, 673.15]
    alpha_vals = [alpha_shell, alpha_shell, alpha_shell, alpha_shell, alpha_shell]  # 1e-6 /K

    # (b)  lambda and Cp
    T_lambda = [573.15, 673.15, 873.15]
    lambda_vals = [k_shell, k_shell, k_shell]  # W/mK
    T_Cp = [573.15, 673.15, 873.15]
    Cp_vals = [c_shell, c_shell, c_shell]  # J/kgK
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
    s0_values = {}

    for i in range(n_slices):
        zone = zones[i]  # shared zone info
        dr_i = zone["dr"]
        Nr_i = zone["Nr"]
        delta_i = zone["delta"]
        s0_core = np.sqrt(5 * np.pi * G1 / (18 * E_core * delta_i))
        s0_shell = sigmat/E_shell
        s0_values[i] = {
            "core": s0_core,
            "shell": s0_shell
        }
        coords_phys_t, ghost_coords_t, n_points_t, ghost_dict_t = gc.generate_one_slice_coordinates(
            r, Nr_i, ghost_nodes_r,
            zones,
            r_ghost_left=True,
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
            r_ghost_left=True,
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
    mask_core_regions_t = {}
    mask_core_regions_phy_t = {}
    mask_void_regions_t = {}

    for i in range(n_slices):
        coords_t = np.vstack([phys_coords_list_t[i], ghost_coords_list_t[i]])
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

        cond = z_t >= (z0 - 1e-12)
        if np.any(cond):
            mask_void = mask_core & cond
        else:
            mask_void = np.zeros_like(mask_core, dtype=bool)
        mask_void_regions_t[i] = mask_void


    if num_processes == 1:
        results_t = [mt.compute_region_matrices(args) for args in task_args_t]
    else:
        with Pool(processes=num_processes) as pool:
            results_t = pool.map(mt.compute_region_matrices, task_args_t)

    distance_matrices_t = []
    partial_area_matrices_t = []
    horizon_masks_t = []
    true_indices_list_t = []

    for i, file_path in enumerate(results_t):
        data = np.load(file_path)
        distance_matrices_t.append(data["distance"])
        partial_area_matrices_t.append(data["area"])
        horizon_masks_t.append(data["mask"])
        true_indices_list_t.append(tuple(data["indices"]))

    end_time_t = time.time()
    print(
        f"[Temperature] Calculation of partial_area_matrices finished, elapsed real time = {end_time_t - start_time_t:.2f}s")

    # -------- Mechanical field --------
    start_time_m = time.time()
    task_args_m = []
    mask_core_regions_m = {}
    mask_void_regions_m = {}

    for i in range(n_slices):
        coords_m = np.vstack([phys_coords_list_m[i], ghost_coords_list_m[i]])
        dr = zones[i]["dr"]  # shared
        delta = zones[i]["delta"]  # shared
        slice_id = zones[i]["layer"]  # shared
        task_args_m.append((coords_m, dr, delta, tolerance, slice_id))
        x_m = coords_m[:, 0]
        z_m = coords_m[:, 1]
        mask_core = (x_m ** 2 + (z_m - r) ** 2) < r_core ** 2
        mask_core_regions_m[i] = mask_core

        cond = z_m >= (z0 - 1e-12)
        if np.any(cond):
            mask_void = mask_core & cond
        else:
            mask_void = np.zeros_like(mask_core, dtype=bool)
        mask_void_regions_m[i] = mask_void

    if num_processes == 1:
        results_m = [mt.compute_region_matrices(args) for args in task_args_m]
    else:
        with Pool(processes=num_processes) as pool:
            results_m = pool.map(mt.compute_region_matrices, task_args_m)

    distance_matrices_m = []
    partial_area_matrices_m = []
    horizon_masks_m = []
    true_indices_list_m = []

    for i, file_path in enumerate(results_m):
        data = np.load(file_path)
        distance_matrices_m.append(data["distance"])
        partial_area_matrices_m.append(data["area"])
        horizon_masks_m.append(data["mask"])
        true_indices_list_m.append(tuple(data["indices"]))

    end_time_m = time.time()
    print(
        f"[Temperature] Calculation of partial_area_matrices finished, elapsed real time = {end_time_m - start_time_m:.2f}s")

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
        ghost_idx_right_t, phys_idx_right_t = bc.find_circle_mirror_pairs_multilayer(
            coords_ghost_right_t, coords_phys_t, dr, r
        )
        boundary_neighbors_t[(i, 'right')] = {
            "ghost_indices": ghost_idx_right_t,
            "phys_indices": phys_idx_right_t,
            "target_region": i
        }

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
    T_phys = {}
    T_left = {}
    T_right = {}
    T_top = {}
    T_bot = {}
    factor_mats = {}
    dt_th = cf.compute_dt_cr_th_solid_with_dist(
        rho_s,
        cs,
        ks,
        partial_area_matrices_t[0],  # region 1 partial_area_matrix
        horizon_masks_t[0],  # region 1 horizon_mask
        distance_matrices_t[0],  # region 1 distance_matrix
        zones[0]["delta"]  # region 1 elta
    ) * 0.5
    nsteps_th = int(total_time / dt_th)
    T_increment = Tinit + (Tsurr - Tinit) / nsteps_th * 5

    for i in range(n_slices):
        threshold_distance = np.sqrt(2) * zones[i]["dr"]
        factor_mats[i] = np.where(distance_matrices_t[i] <= threshold_distance + tolerance, 1.125, 1.0)
        T_phys[i] = np.full(phys_t[i].shape[0], Tinit)
        T_left[i] = np.full(ghost_left_t[i].shape[0], Tinit)
        T_right[i] = np.full(ghost_right_t[i].shape[0], T_increment)
        T_top[i] = np.full(ghost_top_t[i].shape[0], Tinit)
        T_bot[i] = np.full(ghost_bot_t[i].shape[0], Tinit)

    # Boundary temperature assignment
    T = {}
    for (region_id, direction), neighbor_data in boundary_neighbors_t.items():
        if direction == 'right':
            T[region_id] = np.concatenate(
                [T_phys[region_id], T_left[region_id], T_top[region_id], T_bot[region_id]])
            ghost_indices = neighbor_data["ghost_indices"]
            phys_indices = neighbor_data["phys_indices"]
            T_right[region_id][ghost_indices] = T_increment #2 * T_increment - T[region_id][phys_indices]
    """4. Definition of enthalpy for temperature field"""
    H = {}
    K = {}
    shape_factor_matrices = {}
    region_lengths_t = {}  # Used to store length information for each region
    segment_slices_t = {}

    for i in range(n_slices):
        # Segment length
        n_phys = len(T_phys[i])
        n_left = len(T_left[i])
        n_right = len(T_right[i])
        n_top = len(T_top[i])
        n_bot = len(T_bot[i])
        # build slices for [phys | left | right | top | bot]
        s0 = 0
        s1 = s0 + n_phys
        s2 = s1 + n_left
        s3 = s2 + n_right
        s4 = s3 + n_top
        s5 = s4 + n_bot

        segment_slices_t[i] = {
            "phys": slice(s0, s1),
            "left": slice(s1, s2),
            "right": slice(s2, s3),
            "top": slice(s3, s4),
            "bot": slice(s4, s5),
            "total_len": s5
        }

        # Total temperature field of the patchwork
        T[i] = np.concatenate([T_phys[i], T_left[i], T_right[i], T_top[i], T_bot[i]])
        # Save the length information for each segment
        region_lengths_t[i] = {
            'n_phys': n_phys,
            'n_left': n_left,
            'n_right': n_right,
            'n_top': n_top,
            'n_bot': n_bot
        }

    props_t = []
    deltaV_total_all = 0.0  # Initialize total expansion amount

    for i in range(n_slices):
        dr_i = zones[i]["dr"]
        cell_volume = dr_i * dr_i
        deltaV_phase, deltaV_thermal = pfc.compute_melt_and_thermal_expansion(
            T_phys[i], mask_core_regions_phy_t[i], rho_s, rho_l, Ts, Tl, cell_volume,
            beta_s=0,
            beta_l=alpha_core,
            T_ref=Tpre_avg
        )
        deltaV_total_all += deltaV_phase + deltaV_thermal
    if f_void < 1e-10:
        phi_global = 1  # deltaV_total_all / (0.5 * f_void * np.pi * r ** 2)
    else:
        phi_global = min(1.0, deltaV_total_all / (0.5 * f_void * np.pi * r ** 2))

    if deltaV_total_all < 0.5 * f_void * np.pi * r ** 2:
        alpha_core_cal = 0
    else:
        alpha_core_cal = alpha_core

    for i in range(n_slices):
        # --- ⬇️ 更新氧化铝属性，用于 H 和 K ---
        props_i = pfc.update_aluminaoxide_properties(
            T[i],
            mask_core_regions_t[i],
            mask_void_regions_t[i],
            sigmat,
            T_E, E_vals, alpha_vals,T_lambda, lambda_vals,
            T_Cp, Cp_vals, E_core, alpha_core_cal, phi_global,  # 每个 slice 的 φ
            rho_l, cl, kl,
            rho_air, cair, kair
        )
        props_t.append(props_i)  # ✅ 添加到列表中

        H[i] = cf.get_enthalpy(
            T[i],
            mask_core_regions_t[i],
            mask_void_regions_t[i],
            rho_s, rho_l, rho_shell,
            cs, cl,
            props_t[i]["Cp"],  # 全部点的 Cp
            L, Ts, Tl,
            rho_void=props_t[i]["void"]["rho"],
            Cp_void=props_t[i]["void"]["Cp"]
        )

        shape_factor_matrices[i] = np.ones_like(horizon_masks_t[i], dtype=float)
        K[i] = cf.build_K_matrix(
            T[i],
            mask_core_regions_t[i],
            mask_void_regions_t[i],  # ✅ 新增
            cf.compute_thermal_conductivity_matrix,
            factor_mats[i],
            partial_area_matrices_t[i],
            shape_factor_matrices[i],
            distance_matrices_t[i],
            horizon_masks_t[i],
            true_indices_list_t[i],
            ks, kl, Ts, Tl,
            props_t[i]["k"],  # ✅ 壳层区域的插值热导率
            zones[i]["delta"],
            props_t[i]["void"]["k"],  # ✅ 空腔区域等效导热系数
            dt_th
        )

    """5. Definition of Mechanical field"""
    # Mechanical field initialization
    # ---------- 1) Initialize displacement/velocity/acceleration ----------
    Ur, Uz = {}, {}
    Ar, Az = {}, {}
    br, bz = {}, {}
    region_lengths_m = {}  # Used to store length information for each region
    segment_slices_m = {}  # per-region slices into concatenated order: [phys | left | top | bot]
    CorrList_T = {}
    T_m = {}
    dir_r_m, dir_z_m = {}, {}  # 方向矩阵仍可用 _m 表示“机械场的方向矩阵”
    coords_all_m_list = [np.vstack([phys_coords_list_m[i], ghost_coords_list_m[i]]) for i in range(n_slices)]
    coords_all_t_list = [np.vstack([phys_coords_list_t[i], ghost_coords_list_t[i]]) for i in range(n_slices)]
    inner_surface_info_list = []
    materials_m = {}
    props_m = []

    for i in range(n_slices):
        coords_all_m = np.vstack([phys_coords_list_m[i], ghost_coords_list_m[i]])
        coords_all_t = np.vstack([phys_coords_list_t[i], ghost_coords_list_t[i]])
        CorrList_T[i] = pfc.shrink_Tth_by_matching_coords(coords_all_m, coords_all_t)
        T_m = pfc.filter_array_by_indices_keep_only(T[i], CorrList_T[i])
        props_i = pfc.update_aluminaoxide_properties(
            T_m,
            mask_core_regions_m[i],
            mask_void_regions_m[i],
            sigmat,
            T_E, E_vals, alpha_vals,
            T_lambda, lambda_vals,
            T_Cp, Cp_vals,
            E_core,
            alpha_core_cal,
            phi_global,
            rho_l, cl, kl,
            rho_air, cair, kair
        )
        props_m.append(props_i)
        E_interp = props_m[i]["E"]
        E_interp[mask_core_regions_m[i]] = E_core
        alpha_interp = props_m[i]["alpha"]
        so_interp = props_m[i]["so"]

        # === Calculation of non-local mechanical parameters ===
        row_sum = np.sum(partial_area_matrices_m[i], axis=1)  # (Ni,)
        matrix_sum = row_sum[:, None] + row_sum[None, :]  # (Ni, Ni)
        delta_i = zones[i]["delta"]
        # --- c ---
        c_micro_all = (6.0 * E_interp) / (np.pi * (delta_i ** 3) * h * (1.0 - 2.0 * nu_core) * (1.0 + nu_core))
        c_micro_avg = 0.5 * (c_micro_all[:, None] + c_micro_all[None, :])
        c_avg = (2.0 * np.pi * delta_i ** 2 / matrix_sum) * c_micro_avg
        # --- nu ---
        nu_all = np.where(mask_core_regions_m[i], nu_core, nu_shell)
        nu_avg = 0.5 * (nu_all[:, None] + nu_all[None, :])
        # --- rho ---
        rho_avg, rho = cf.get_density(T_m, mask_core_regions_m[i], mask_void_regions_m[i], rho_s, rho_l, Ts, Tl,
                                      rho_shell, props_m[i]["void"]["rho"])
        # --- alpha ---
        alpha_avg = 0.5 * (alpha_interp[:, None] + alpha_interp[None, :])
        # --- s0 ---
        s0_avg = 0.5 * (so_interp[:, None] + so_interp[None, :])
        # --- mu ---
        # 你需要在循环外定义一次
        # 在模拟初始位置添加：

        mu = np.zeros_like(horizon_masks_m[i], dtype=int)
        mu[horizon_masks_m[i] & ~mask_core_regions_m[i]] = 1

        # --- storage ---
        materials_m[i] = {
            "c_avg": c_avg,
            "nu_avg": nu_avg,
            "rho_avg": rho_avg,
            "rho": rho,
            "alpha_avg": alpha_avg,
            "mu": mu,
            "s0_avg": s0_avg,
            "T_m": T_m
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

        dir_r_m[i], dir_z_m[i] = pfc.compute_direction_matrix(
            coords_all_m_list[i],
            Ur[i],
            Uz[i],
            horizon_masks_m[i]
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
    for step1 in range(nsteps_th):
        print(f"✅ Completed {step1 + 1}/{nsteps_th} steps of calculation")
        if has_frac:
            plot.plot_mu_field(r, coords_all_m_list, phi_list, grid_n=300, title_prefix="Core+Shell μ")
            break
        task_args = []
        for i in range(n_slices):
            delta = zones[i]["delta"]
            lengths = region_lengths_t[i]
            args = (
                i,
                T[i], H[i], K[i],
                factor_mats[i],
                partial_area_matrices_t[i],
                shape_factor_matrices[i],
                distance_matrices_t[i],
                horizon_masks_t[i],
                true_indices_list_t[i],
                mask_core_regions_t[i],
                delta,
                dt_th,
                rho_s, rho_l, cs, cl, L, Ts, Tl,
                rho_shell, props_t[i]["Cp"],
                ks, kl, props_t[i]["k"],
                mask_void_regions_t[i], props_t[i]["void"]["rho"], props_t[i]["void"]["Cp"], props_t[i]["void"]["k"]
            )
            task_args.append(args)

        if num_processes == 1:
            # Serial execution
            results = [mt.update_temperature_for_region(args) for args in task_args]
        else:
            # Parallel execution
            with Pool(processes=num_processes) as pool:
                results = pool.map(mt.update_temperature_for_region, task_args)

        # ✅ Split results
        for region_id, Tnew, Hnew, Knew in results:
            sl = segment_slices_t[region_id]
            T_phys[region_id] = Tnew[sl["phys"]]
            T_left[region_id] = Tnew[sl["left"]]
            T_right[region_id] = Tnew[sl["right"]]
            T_top[region_id] = Tnew[sl["top"]]
            T_bot[region_id] = Tnew[sl["bot"]]
            H[region_id] = Hnew
            K[region_id] = Knew

        # ✅ Unified handling of boundary conditions
        T_increment = min(Tinit + (Tsurr - Tinit) * (step1 + 2) / nsteps_th * 5,Tsurr)
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
                T[region_id] = np.concatenate(
                    [T_phys[region_id], T_left[region_id], T_top[region_id], T_bot[region_id]])
                ghost_indices = neighbor_data["ghost_indices"]
                phys_indices = neighbor_data["phys_indices"]
                T_right[region_id][ghost_indices] = 2 * T_increment - T[region_id][phys_indices]

        for (region_id, direction), neighbor_data in boundary_neighbors_t.items():

            ghost_indices = neighbor_data["ghost_indices"]
            phys_indices = neighbor_data["phys_indices"]
            target_region = neighbor_data["target_region"]

            dr1 = zones[region_id]["dr"]
            dr2 = zones[target_region]["dr"]

            if direction == "left":
                T[region_id] = np.concatenate(
                    [T_phys[region_id], T_left[region_id], T_right[region_id], T_top[region_id], T_bot[region_id]])
                T_left[region_id][ghost_indices] = T[region_id][phys_indices]
        deltaV_total_all = 0
        for i in range(n_slices):
            T[i] = np.concatenate([T_phys[i], T_left[i], T_right[i], T_top[i], T_bot[i]])
            materials_m[i]["T_m"] = pfc.filter_array_by_indices_keep_only(T[i], CorrList_T[i])
            dr_i = zones[i]["dr"]
            cell_volume = dr_i * dr_i
            deltaV_phase, deltaV_thermal = pfc.compute_melt_and_thermal_expansion(
                T_phys[i], mask_core_regions_phy_t[i], rho_s, rho_l, Ts, Tl, cell_volume,
                beta_s=0,
                beta_l=alpha_core,
                T_ref=Tpre_avg
            )
            deltaV_total_all += deltaV_phase + deltaV_thermal

        mu_initialized = False  # 标记：是否已执行过那次 for
        if f_void < 1e-10:
            phi_global = 1 #deltaV_total_all / (0.5 * f_void * np.pi * r ** 2)
            if not mu_initialized:
                for i in range(n_slices):
                    mu = np.ones_like(horizon_masks_m[i], dtype=int)
                    materials_m[i]["mu"] = mu
                mu_initialized = True
        else:
            phi_global = min(1.0, deltaV_total_all / (V0_all * f_void ))

        if deltaV_total_all < V0_all * f_void:
            alpha_core_cal = 0
        else:
            alpha_core_cal = alpha_core
            if not mu_initialized:
                for i in range(n_slices):
                    mu = np.ones_like(horizon_masks_m[i], dtype=int)
                    materials_m[i]["mu"] = mu
                mu_initialized = True

        for i in range(n_slices):
            props_i = pfc.update_aluminaoxide_properties(
                T[i],
                mask_core_regions_t[i],
                mask_void_regions_t[i],
                sigmat,
                T_E, E_vals, alpha_vals, T_lambda, lambda_vals,
                T_Cp, Cp_vals, E_core, alpha_core_cal, phi_global,  # 每个 slice 的 φ
                rho_l, cl, kl,
                rho_air, cair, kair
            )
            props_t.append(props_i)  # ✅ 添加到列表中

            props_i = pfc.update_aluminaoxide_properties(
                materials_m[i]["T_m"], mask_core_regions_m[i],
                mask_void_regions_m[i], sigmat,
                T_E, E_vals, alpha_vals,
                T_lambda, lambda_vals,
                T_Cp, Cp_vals,
                E_core, alpha_core_cal,
                phi_global,
                rho_l, cl, kl,
                rho_air, cair, kair
            )
            props_m.append(props_i)
            E_interp = props_m[i]["E"]  # (Ni,)
            E_interp[mask_core_regions_m[i]] = E_core
            alpha_interp = props_m[i]["alpha"]  # (Ni,)
            so_interp = props_m[i]["so"]  # (Ni,)

            row_sum = np.sum(partial_area_matrices_m[i], axis=1)  # (Ni,)
            matrix_sum = row_sum[:, None] + row_sum[None, :]  # (Ni, Ni)
            delta_i = zones[i]["delta"]

            c_micro_all = (6.0 * E_interp) / (np.pi * (delta_i ** 3) * h * (1.0 - 2.0 * nu_core) * (1.0 + nu_core))
            c_micro_avg = 0.5 * (c_micro_all[:, None] + c_micro_all[None, :])
            c_avg = (2.0 * np.pi * delta_i ** 2 / matrix_sum) * c_micro_avg

            rho_avg, rho = cf.get_density(materials_m[i]["T_m"], mask_core_regions_m[i], mask_void_regions_m[i], rho_s, rho_l, Ts, Tl,
                                          rho_shell, props_m[i]["void"]["rho"])
            alpha_avg = 0.5 * (alpha_interp[:, None] + alpha_interp[None, :])
            s0_avg = 0.5 * (so_interp[:, None] + so_interp[None, :])

            nu_all = np.where(mask_core_regions_m[i], nu_core, nu_shell)
            nu_avg = 0.5 * (nu_all[:, None] + nu_all[None, :])
            materials_m[i]["c_avg"] = c_avg
            materials_m[i]["rho_avg"] = rho_avg
            materials_m[i]["rho"] = rho
            materials_m[i]["alpha_avg"] = alpha_avg
            materials_m[i]["s0_avg"] = s0_avg
            materials_m[i]["nu_avg"] = nu_avg
        """6. Definition of physical strength: Note that the average value of the elastic modulus is taken here."""
        E_shell_all = []
        for i in range(n_slices):
            E_i = props_m[i]["E"]  # (N_i,)
            dr_i = zones[i]["dr"]
            mask_shell_i = ~mask_core_regions_m[i]  # 非核心区域
            E_shell_all.append(E_i[mask_shell_i])

        # 拼接所有壳层区域的 E
        E_shell_concat = np.concatenate(E_shell_all)
        E_mean = np.mean(E_shell_concat)   # 有效等效模量
        p_pre = p
        if deltaV_total_all <= V0_all * f_void:
            p = 0
        else:
            p = E_mean * ((deltaV_total_all - 0.5 * f_void * np.pi * r ** 2) / V0_all) / np.pi / r

        print(f"[Step {step1 + 1}] Time = {step1 * dt_th:.3e} s | Pressure p = {p:.3e} Pa")
        ninc = number_pressure
        delta_p = p - p_pre  # 本热步相对上一热步的新增压力
        inc_p = delta_p / ninc  # 每个增量的压力
        if step1 == 0:
            for i in range(n_slices):
                idx_local = inner_surface_info_list[i]["indices"]
                unit_vec = inner_surface_info_list[i]["unit_to_center"]
                dr_i = zones[i]["dr"]
                br[i][idx_local] += p * unit_vec[:, 0]
                bz[i][idx_local] += p * unit_vec[:, 1]

            """     7. Displacement initialization      """
            task_args_accel_m_ini = []
            for i in range(n_slices):
                task_args_accel_m_ini.append((
                    coords_all_m_list[i],
                    Ur[i], Uz[i], br[i], bz[i],
                    horizon_masks_m[i],
                    dir_r_m[i], dir_z_m[i],
                    materials_m[i]["c_avg"],
                    partial_area_matrices_m[i],
                    materials_m[i]["rho_avg"],
                    materials_m[i]["rho"],
                    materials_m[i]["T_m"], Tpre_avg,
                    materials_m[i]["nu_avg"],
                    materials_m[i]["alpha_avg"],
                    materials_m[i]["mu"],
                    materials_m[i]["s0_avg"]
                ))

            # ---------- 3) Calculation (serial/multiprocessing) ----------
            if num_processes == 1:
                results_accel_ini = [mt.compute_accelerated_velocity_initial(args) for args in task_args_accel_m_ini]
            else:
                with Pool(processes=num_processes) as pool:
                    results_accel_ini = pool.map(mt.compute_accelerated_velocity_initial, task_args_accel_m_ini)

            # 回填每个区域的加速度
            for idx, (Ar_new, Az_new, mu_new) in enumerate(results_accel_ini):
                Ar[idx] = Ar_new
                Az[idx] = Az_new
                materials_m[idx]["mu"] = mu_new
            Fr_0, Fz_0 = {}, {}
            Fr = {}
            Fz = {}
            cr_n = {}
            cz_n = {}
            lambda_diag_matrix = {}
            Vr_half, Vz_half = {}, {}
            phi_regions = {}
            # ---------- 4) Calculate Fr_0 / Fz_0, λ, and half-step velocity based on acceleration ----------
            for i in range(n_slices):
                Fr_0[i] = Ar[i] * materials_m[i]["rho"]
                Fz_0[i] = Az[i] * materials_m[i]["rho"]

                # λ Diagonal (ADR)
                lambda_diag_matrix[i] = ADR.compute_lambda_diag_matrix(
                    partial_area_matrices_m[i],  # (Ni, Ni)
                    distance_matrices_m[i],  # (Ni, Ni)
                    materials_m[i]["c_avg"],  # ✅ 刚度矩阵
                    horizon_masks_m[i]  # (Ni, Ni)
                )

                # Half-step speed
                Vr_half[i] = (1 / 2) * (Fr_0[i] / lambda_diag_matrix[i])
                Vz_half[i] = (1 / 2) * (Fz_0[i] / lambda_diag_matrix[i])
                Ur[i] = Vr_half[i] * dt_ADR + Ur[i]
                Uz[i] = Vz_half[i] * dt_ADR + Uz[i]

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
                    br[i][idx_local] = p_s  * unit_vec[:, 0]
                    bz[i][idx_local] = p_s  * unit_vec[:, 1]

                # ---- 本份的非线性/迭代，必须收敛才允许进入下一份 ----
                converged_this_inc = False
                for step in range(nsteps_m):
                    # 记录上一步位移（用于收敛判据）
                    Ur_all_prev = np.concatenate([Ur[j] for j in range(n_slices)])
                    Uz_all_prev = np.concatenate([Uz[j] for j in range(n_slices)])

                    # 组织并行任务参数（注意变量名统一）
                    task_args_accel_m = []
                    for i in range(n_slices):
                        task_args_accel_m.append((
                            coords_all_m_list[i], Ur[i], Uz[i], br[i], bz[i],
                            horizon_masks_m[i], dir_r_m[i], dir_z_m[i],
                            materials_m[i]["c_avg"], partial_area_matrices_m[i],
                            materials_m[i]["rho_avg"], materials_m[i]["rho"],
                            materials_m[i]["T_m"], Tpre_avg, materials_m[i]["nu_avg"],
                            materials_m[i]["alpha_avg"], CorrList_T[i],
                            materials_m[i]["mu"], materials_m[i]["s0_avg"],
                            Fr_0[i], Fz_0[i], Vr_half[i], Vz_half[i],
                            lambda_diag_matrix[i], ADR, dt_ADR, materials_m[i]["rho"]
                        ))

                    # 单进程 / 多进程
                    if num_processes == 1:
                        results_accel = [mt.compute_mechanical_step(args) for args in task_args_accel_m]
                    else:
                        from multiprocessing import Pool

                        with Pool(processes=num_processes) as pool:
                            results_accel = pool.map(mt.compute_mechanical_step, task_args_accel_m)  # 修正变量名

                    # 回填结果
                    for i, (Ur_new, Uz_new, Fr_0_new, Fz_0_new, Vr_half_new, Vz_half_new, mu_new) in enumerate(
                            results_accel):
                        Ur[i] = Ur_new
                        Uz[i] = Uz_new
                        Fr_0[i] = Fr_0_new
                        Fz_0[i] = Fz_0_new
                        Vr_half[i] = Vr_half_new
                        Vz_half[i] = Vz_half_new
                        materials_m[i]["mu"] = mu_new  # ✅ 回填 mu
                        Ur[i][segment_slices_m[i]['left']] = 0.0
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
                    for i in range(n_slices):
                        A = partial_area_matrices_m[i].copy()
                        np.fill_diagonal(A, 0)
                        mu_i = materials_m[i]["mu"]
                        phi_i = 1 - np.sum(mu_i * A, axis=1) / np.sum(A, axis=1)
                        phi_regions[i] = phi_i


                    phi_list = [phi_regions[i] for i in range(n_slices)]
                    if mu_initialized:
                        for i, phi in enumerate(phi_list):
                            if np.any(np.asarray(phi) > 0.5):
                                has_frac = True
                                print(f"Region {i}: fracture, max(phi) = {np.max(phi):.6g}")
                        if has_frac:
                            break

                    if rms_increment < rms:
                        print(
                            f"[Mechanical] converged at step {step} with RMS {rms_increment:.3e}, p = {p_s * dr1 :.6g}")
                        converged_this_inc = True
                        break

                    if step > 0 and step % 10 == 0:
                        print(f"[Mechanical] step {step}, RMS {rms_increment:.3e}")

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

    plot.plot_temperature_contour_in_circle(phys_coords_list_t, dr1, T_phys, radius=r)
    plot.plot_displacement_contours_in_circle(
        phys_coords_list_m, U_phys, r, dr1,
        titles=('Ur (m)', 'Uz (m)', 'Umag (m)'),
        levels=20
    )
    phi_regions = {}

    for i in range(n_slices):
        A = partial_area_matrices_m[i].copy()
        np.fill_diagonal(A, 0)
        mu_i = materials_m[i]["mu"]

        phi_i = 1 - np.sum(mu_i * A, axis=1) / np.sum(A, axis=1)
        phi_regions[i] = phi_i

    phi_list = [phi_regions[i] for i in range(n_slices)]
    plot.plot_mu_field(r, coords_all_m_list, phi_list, grid_n=300, title_prefix="Core+Shell μ")

    end_time2 = time.time()
    print(f"Whole Calculation time = {end_time2 - start_time2:.2f}s")
