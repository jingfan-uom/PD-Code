
import generate_coordinates as gc
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import Multiprocess_task_function as mt
import bc_funcs
import matplotlib.pyplot as plt

rho_s, cs, ks = 1000.0, 2060.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 312.65
Tl = 313.65
L = 333
tolerance = 1e-14
Tsurr = 400
Tinit = 283.15
""" Initialization of coarse regions and temperatures """
r = 20 * 1e-6  # Domain size in r and z directions (meters)
# d_s = 1.5 * 1e-6
dr1, dr2 = 0.1 * 1e-6, 0.1 * 1e-6
dr3 = 2 * dr2
len1 = 0.5 * 1e-6 ; len2 = 0.5 * 1e-6 ;len3 = 1 * 1e-6
dr_l = 0.4 * 1e-6
ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction
n_slices = 8

# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
if __name__ == "__main__":


    """1. Definition of regional coordinates and area matrix """

    zones = gc.compute_layer_dr_r_nr(r, n_slices, dr1, dr2, dr3, dr_l, len1, len2, len3)

    for z in zones:
        print(f"Layer {z['layer']:2d}: dr = {z['dr']:.2e}, delta = {z['delta']:.2e}, "
              f"length = {z['length']:.2e}, Nr = {z['Nr']}")

    phys_coords_list = []
    ghost_coords_list = []
    ghost_dict_list = []
    start_time = time.time()

    for i in range(n_slices):
        zone = zones[i]  # 获取当前层的配置
        dr_i = zone["dr"]
        Nr_i = zone["Nr"]

        coords_phys, ghost_coords, n_points,ghost_dict = gc.generate_one_slice_coordinates(
            r, Nr_i, ghost_nodes_r,
            zones,
            r_ghost_left=True,
            r_ghost_right=True,
            r_ghost_top=True,
            r_ghost_bot=True,
            n_slices=n_slices,
            slice_id=i,
            graph=False
        )
        print(f"The number of particles in region {i} is: {n_points}")

        phys_labeled = np.hstack([coords_phys, np.full((coords_phys.shape[0], 1), i)])
        ghost_labeled = np.hstack([ghost_coords, np.full((ghost_coords.shape[0], 1), i)])
        phys_coords_list.append(phys_labeled)
        ghost_coords_list.append(ghost_labeled)
        ghost_dict_list.append(ghost_dict)

    task_args = []
    for i in range(n_slices):
        coords = np.vstack([phys_coords_list[i], ghost_coords_list[i]])
        dr = zones[i]["dr"]
        delta = zones[i]["delta"]
        slice_id = zones[i]["layer"]
        task_args.append((coords, dr, delta, tolerance, slice_id))

    with Pool(processes=min(n_slices, 4)) as pool:
        results = pool.map(mt.compute_region_matrices, task_args)

    distance_matrices = []
    partial_area_matrices = []
    horizon_masks = []
    true_indices_list = []

    for i, file_path in enumerate(results):
        data = np.load(file_path)
        distance_matrices.append(data["distance"])
        partial_area_matrices.append(data["area"])
        horizon_masks.append(data["mask"])
        true_indices_list.append(tuple(data["indices"]))

    end_time = time.time()
    print(f"Calculation of partial_area_matrices finished, elapsed real time = {end_time - start_time:.2f}s")


    """2. Definition of particle information at the interface between regions """
    # 初始化结构化字典
    ghost_top = {}
    ghost_bot = {}
    ghost_left = {}
    ghost_right = {}
    phys = {}

    # 遍历每个区域
    for i in range(n_slices):
        ghost_top[i] = ghost_dict_list[i]["top"]
        ghost_bot[i] = ghost_dict_list[i]["bot"]
        ghost_left[i] = ghost_dict_list[i]["left"]
        ghost_right[i] = ghost_dict_list[i]["right"]
        phys[i] = phys_coords_list[i]

    # 保存每一对相邻区域的匹配关系
    boundary_neighbors = {}

    for i in range(n_slices):  # 区域 i 与区域 i+1 相邻（在竖直方向）
        # 第 i 区域的下边界（bot）鬼点，对应第 i+1 区域的物理点
        if i <= n_slices - 2:
            dr_i = zones[i]["dr"]
            dr_ip1 = zones[i + 1]["dr"]
            ghost_coords_bot = ghost_bot[i][:, :2]
            phys_coords_ip1 = phys[i + 1][:, :2]

            if abs(dr_i - dr_ip1) < tolerance:
                ghost_idx_bot, phys_idx_bot = bc_funcs.get_same_neighbor_points(
                    ghost_coords_bot, phys_coords_ip1, tol=tolerance)
            elif dr_i < dr_ip1:
                ghost_idx_bot, phys_idx_bot = bc_funcs.get_coarse_neighbor_points(
                    ghost_coords_bot, phys_coords_ip1, dr_fine=dr_i, tol=tolerance)
            else:  # dr_i > dr_ip1
                ghost_idx_bot, phys_idx_bot = bc_funcs.get_fine_neighbor_points(
                    ghost_coords_bot, phys_coords_ip1, dr_fine=dr_ip1, tol=tolerance)
            # 存入结构化结果字典，标明是 i 区域的下边界
            boundary_neighbors[(i, 'bot')] = {
                "ghost_indices": ghost_idx_bot,
                "phys_indices": phys_idx_bot,
                "target_region": i + 1
            }
        # 对称地处理第 i+1 区域的上边界 ghost 点（可选，建议也加）

            ghost_coords_top = ghost_top[i + 1][:, :2]
            phys_coords_i = phys[i][:, :2]
            if abs(dr_i - dr_ip1) < tolerance:
                ghost_idx_top, phys_idx_top = bc_funcs.get_same_neighbor_points(ghost_coords_top, phys_coords_i,
                                                                                  tol=tolerance)
            elif dr_ip1 < dr_i:
                ghost_idx_top, phys_idx_top = bc_funcs.get_coarse_neighbor_points(ghost_coords_top, phys_coords_i,
                                                                                    dr_fine=dr_ip1, tol=tolerance)
            else:
                ghost_idx_top, phys_idx_top = bc_funcs.get_fine_neighbor_points(ghost_coords_top, phys_coords_i,
                                                                                  dr_fine=dr_i, tol=tolerance)
            boundary_neighbors[(i + 1, 'top')] = {
                "ghost_indices": ghost_idx_top,
                "phys_indices": phys_idx_top,
                "target_region": i
            }
            print(f"[Region {i} -> {i + 1}] bottom: ghost = {len(ghost_idx_bot)}, phys = {len(phys_idx_bot)}")
            print(f"[Region {i + 1} -> {i}] top:    ghost = {len(ghost_idx_top)}, phys = {len(phys_idx_top)}")
        coords_ghost_left = ghost_left[i][:, :2]
        coords_ghost_right = ghost_right[i][:, :2]
        coords_phys = phys[i][:, :2]
        dr = zones[i]["dr"]

        # 左边界：关于 r=0 的线对称镜像
        ghost_idx_left, phys_idx_left = bc_funcs.find_mirror_pairs(
            coords_ghost_left, coords_phys, tolerance
        )
        boundary_neighbors[(i, 'left')] = {
            "ghost_indices": ghost_idx_left,
            "phys_indices": phys_idx_left,
            "target_region": i  # 左右边界是自身区域内
        }

        # 右边界：关于圆心的圆对称镜像
        ghost_idx_right, phys_idx_right = bc_funcs.find_circle_mirror_pairs_multilayer(
            coords_ghost_right, coords_phys, dr, r
        )
        boundary_neighbors[(i, 'right')] = {
            "ghost_indices": ghost_idx_right,
            "phys_indices": phys_idx_right,
            "target_region": i
        }

        print(f"[Region {i}] left:   ghost = {len(ghost_idx_left)}, phys = {len(phys_idx_left)}")
        print(f"[Region {i}] right:  ghost = {len(ghost_idx_right)}, phys = {len(phys_idx_right)}")

    """3. Definition of temperature """
    T_phys = {}
    T_left = {}
    T_right = {}
    T_top = {}
    T_bot = {}
    factor_mats = {}

    for i in range(n_slices):
        dr = zones[i]["dr"]
        coords_phys = phys[i][:, :2]
        coords_ghost_left = ghost_left[i][:, :2]
        coords_ghost_right = ghost_right[i][:, :2]
        distance_matrix = distance_matrices[i]

        # 1. 局部因子矩阵
        threshold_distance = np.sqrt(2) * dr
        factor_mat = np.where(distance_matrix <= threshold_distance + tolerance, 1.125, 1.0)
        factor_mats[i] = factor_mat

        # 2. 初始化温度场
        T_phys[i] = np.full(coords_phys.shape[0], Tinit)
        T_left[i] = np.full(coords_ghost_left.shape[0], Tinit)
        T_right[i] = np.full(coords_ghost_right.shape[0], Tsurr)
        T_phys[1] = np.full(coords_phys.shape[0], Tinit-100)

        T_phys[4] = np.full(coords_phys.shape[0], Tinit - 150)
        T_phys[6] = np.full(coords_phys.shape[0], Tinit - 200)
        # 3. 上边界 ghost 点温度（区域 0 没有上边界）
        if i != 0:
            coords_ghost_top = ghost_top[i][:, :2]
            T_top[i] = np.full(coords_ghost_top.shape[0], Tinit)

        # 4. 下边界 ghost 点温度（区域 n_slices-1 没有下边界）
        if i != n_slices - 1:
            coords_ghost_bot = ghost_bot[i][:, :2]
            T_bot[i] = np.full(coords_ghost_bot.shape[0], Tinit)

    for (region_id, direction), neighbor_data in boundary_neighbors.items():
        ghost_indices = neighbor_data["ghost_indices"]
        phys_indices = neighbor_data["phys_indices"]
        target_region = neighbor_data["target_region"]

        print(f"\n====== [{region_id}] direction: {direction} ======")
        print(f"  ghost_indices.shape = {np.shape(ghost_indices)}")
        print(f"  phys_indices.shape = {np.shape(phys_indices)}")


        if direction in ['left', 'right']:
            print(f"  T_phys[{region_id}].shape = {np.shape(T_phys[region_id])}")
            if direction == 'left':
                print(f"  T_left[{region_id}].shape = {np.shape(T_left[region_id])}")
                T_left[region_id][ghost_indices] = T_phys[region_id][phys_indices]
            else:
                print(f"  T_right[{region_id}].shape = {np.shape(T_right[region_id])}")
                T_right[region_id][ghost_indices] = 2 * Tsurr - T_phys[region_id][phys_indices]



    for i in range(n_slices):
        coords_phys = phys[i][:, :2]
        coords_ghost_left = ghost_left[i][:, :2]
        coords_ghost_right = ghost_right[i][:, :2]

        T_p = T_phys[i]
        T_l = T_left[i]
        T_r = T_right[i]

        plt.figure(figsize=(6, 5))

        # 物理点温度图
        sc1 = plt.scatter(coords_phys[:, 0], coords_phys[:, 1], c=T_p, cmap='viridis', s=8, label='Physical')

        # 左 ghost
        if len(coords_ghost_left) > 0:
            sc2 = plt.scatter(coords_ghost_left[:, 0], coords_ghost_left[:, 1], c=T_l, cmap='cool', s=8,
                              label='Ghost Left')

        # 右 ghost
        if len(coords_ghost_right) > 0:
            sc3 = plt.scatter(coords_ghost_right[:, 0], coords_ghost_right[:, 1], c=T_r, cmap='hot', s=8,
                              label='Ghost Right')

        plt.gca().set_aspect('equal')
        plt.xlabel("r (radius direction)")
        plt.ylabel("z (axial direction)")
        plt.title(f"Temperature Field - Region {i}")
        plt.grid(True)
        plt.colorbar(sc1, label='Temperature (K)')

        plt.legend()
        plt.tight_layout()
        plt.show()





















