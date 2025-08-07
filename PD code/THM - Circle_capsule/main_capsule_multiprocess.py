
import generate_coordinates as gc
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import Multiprocess_task_function as mt
import bc_funcs as bc
import core_funcs as cf

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

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
dr1 = 0.05 * 1e-6
dr2 = 0.05 * 1e-6
dr3 = 2 * dr2
dr_l = 0.4 * 1e-6

len1 = 0.5 * 1e-6
len2 = 0.5 * 1e-6
len3 = 1 * 1e-6

ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction
n_slices = 8
num_processes = 4
dt = 1e-8  # Time step in seconds
total_time = 100e-8  # Total simulation time (5 hours)
# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
if __name__ == "__main__":
    start_time2 = time.time()
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

    # ✅ You can use serial computation (the simplest method and avoids process isolation issues).
    if num_processes == 1:
        # serial computation
        results = [mt.compute_region_matrices(args) for args in task_args]
    else:
        # Multi-process execution
        with Pool(processes= num_processes) as pool:
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
                ghost_idx_bot, phys_idx_bot = bc.get_same_neighbor_points(
                    ghost_coords_bot, phys_coords_ip1, tol=tolerance)
            elif dr_i < dr_ip1:
                ghost_idx_bot, phys_idx_bot = bc.get_coarse_neighbor_points(
                    ghost_coords_bot, phys_coords_ip1, dr_fine=dr_i, tol=tolerance)
            else:  # dr_i > dr_ip1
                ghost_idx_bot, phys_idx_bot = bc.get_fine_neighbor_points(
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
                ghost_idx_top, phys_idx_top = bc.get_same_neighbor_points(ghost_coords_top, phys_coords_i,
                                                                                  tol=tolerance)
            elif dr_ip1 < dr_i:
                ghost_idx_top, phys_idx_top = bc.get_coarse_neighbor_points(ghost_coords_top, phys_coords_i,
                                                                                    dr_fine=dr_ip1, tol=tolerance)
            else:
                ghost_idx_top, phys_idx_top = bc.get_fine_neighbor_points(ghost_coords_top, phys_coords_i,
                                                                                  dr_fine=dr_i, tol=tolerance)
            boundary_neighbors[(i + 1, 'top')] = {
                "ghost_indices": ghost_idx_top,
                "phys_indices": phys_idx_top,
                "target_region": i
            }

        coords_ghost_left = ghost_left[i][:, :2]
        coords_ghost_right = ghost_right[i][:, :2]
        coords_phys = phys[i][:, :2]
        dr = zones[i]["dr"]

        # 左边界：关于 r=0 的线对称镜像
        ghost_idx_left, phys_idx_left = bc.find_mirror_pairs(
            coords_ghost_left, coords_phys, tolerance
        )
        boundary_neighbors[(i, 'left')] = {
            "ghost_indices": ghost_idx_left,
            "phys_indices": phys_idx_left,
            "target_region": i  # 左右边界是自身区域内
        }

        # 右边界：关于圆心的圆对称镜像
        ghost_idx_right, phys_idx_right = bc.find_circle_mirror_pairs_multilayer(
            coords_ghost_right, coords_phys, dr, r
        )
        boundary_neighbors[(i, 'right')] = {
            "ghost_indices": ghost_idx_right,
            "phys_indices": phys_idx_right,
            "target_region": i
        }


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
        coords_ghost_top = ghost_top[i][:, :2]
        coords_ghost_bot = ghost_bot[i][:, :2]
        distance_matrix = distance_matrices[i]

        # 1. 因子矩阵
        threshold_distance = np.sqrt(2) * dr
        factor_mats[i] = np.where(distance_matrix <= threshold_distance + tolerance, 1.125, 1.0)

        # 2. 初始化物理温度场（区域内不同初始温度）
        T_phys[i] = np.full(coords_phys.shape[0], Tinit)

        # 3. 初始化 ghost 温度场
        T_left[i] = np.full(coords_ghost_left.shape[0], Tinit)
        T_right[i] = np.full(coords_ghost_right.shape[0], Tsurr)
        T_top[i] = np.full(coords_ghost_top.shape[0], Tinit)
        T_bot[i] = np.full(coords_ghost_bot.shape[0], Tinit)

    for (region_id, direction), neighbor_data in boundary_neighbors.items():
        ghost_indices = neighbor_data["ghost_indices"]
        phys_indices = neighbor_data["phys_indices"]
        target_region = neighbor_data["target_region"]

        # 左右边界直接处理
        if direction in ['left', 'right']:
            if direction == 'left':
                T_left[region_id][ghost_indices] = T_phys[region_id][phys_indices]
            else:
                T_right[region_id][ghost_indices] = 2 * Tsurr - T_phys[region_id][phys_indices]

        # 上下边界插值处理
        elif direction in ['top', 'bot']:

            dr1 = zones[region_id]["dr"]
            dr2 = zones[target_region]["dr"]

            if abs(dr1 - dr2) < tolerance:
                if direction == 'top':
                    T_top[region_id] = bc.interpolate_temperature_for_same(
                        T_top[region_id],
                        T_phys[target_region],
                        ghost_indices,
                        phys_indices
                    )
                else:
                    T_bot[region_id] = bc.interpolate_temperature_for_same(
                        T_bot[region_id],
                        T_phys[target_region],
                        ghost_indices,
                        phys_indices
                    )
            else:
                if direction == 'top':
                    T_top[region_id] = bc.interpolate_temperature_for_coarse_and_fine(
                        T_top[region_id],
                        T_phys[target_region],
                        ghost_indices,
                        phys_indices
                    )
                else:
                    T_bot[region_id] = bc.interpolate_temperature_for_coarse_and_fine(
                        T_bot[region_id],
                        T_phys[target_region],
                        ghost_indices,
                        phys_indices
                    )
    """3. Definition of enthalpy """
    T = {}
    H = {}
    K = {}
    shape_factor_matrices = {}
    region_lengths = {}  # 用于保存每个区域的长度信息

    for i in range(n_slices):
        # 获取当前区域各类温度数组
        T_p = T_phys[i]
        T_l = T_left.get(i, np.array([]))
        T_r = T_right.get(i, np.array([]))
        T_t = T_top.get(i, np.array([]))
        T_b = T_bot.get(i, np.array([]))

        # 分段长度
        n_phys = len(T_p)
        n_left = len(T_l)
        n_right = len(T_r)
        n_top = len(T_t)
        n_bot = len(T_b)

        # 拼接总温度场
        T[i] = np.concatenate([T_p, T_l, T_r, T_t, T_b])
        delta = zones[i]["delta"]

        # 保存每段长度信息
        region_lengths[i] = {
            'n_phys': n_phys,
            'n_left': n_left,
            'n_right': n_right,
            'n_top': n_top,
            'n_bot': n_bot
        }

        H[i] = cf.get_enthalpy(T[i], rho_l, cs, cl, L, Ts, Tl)
        shape_factor_matrices[i] = np.ones_like(horizon_masks[i], dtype=float)
        K[i] = cf.build_K_matrix(
            T[i],
            cf.compute_thermal_conductivity_matrix,
            factor_mats[i],
            partial_area_matrices[i],
            shape_factor_matrices[i],
            distance_matrices[i],
            horizon_masks[i],
            true_indices_list[i],
            ks, kl, Ts, Tl, delta, dt
        )

    # Build initial conductivity matrix



    nsteps = int(total_time / dt)
    print_interval = int(10 / dt)  # Print progress every 10 simulated seconds
    print(f"Total steps: {nsteps}")
    start_time = time.time()

    # ------------------------
    # Time-stepping loop
    # ------------------------
    save_times = [2, 4, 6, 8, 10]  # Save snapshots (in hours)
    save_steps = [int(t * 3600 / dt) for t in save_times]
    T_record = []  # Store temperature snapshots

    for step in range(nsteps):
        if (step + 1) % 10 == 0:
            print(f"✅ Completed {step + 1} steps of calculation")

        task_args = []
        for i in range(n_slices):
            delta = zones[i]["delta"]
            lengths = region_lengths[i]
            args = (
                i,
                T[i],
                H[i],
                K[i],
                factor_mats[i],
                partial_area_matrices[i],
                shape_factor_matrices[i],
                distance_matrices[i],
                horizon_masks[i],
                true_indices_list[i],
                delta,
                dt,
                rho_s, cs, cl, L, Ts, Tl, ks, kl
            )
            task_args.append(args)

        if num_processes == 1:
            # 串行执行
            results = [mt.update_temperature_for_region(args) for args in task_args]
        else:
            # 并行执行
            with Pool(processes=num_processes) as pool:
                results = pool.map(mt.update_temperature_for_region, task_args)

        # ✅ 拆分结果
        for region_id, Tnew, Hnew, Knew in results:
            lengths = region_lengths[region_id]
            n_phys = lengths["n_phys"]
            n_left = lengths["n_left"]
            n_right = lengths["n_right"]
            n_top = lengths["n_top"]
            n_bot = lengths["n_bot"]

            T_phys[region_id] = Tnew[:n_phys]
            T_left[region_id] = Tnew[n_phys: n_phys + n_left]
            T_right[region_id] = Tnew[n_phys + n_left: n_phys + n_left + n_right]
            T_top[region_id] = Tnew[n_phys + n_left + n_right: n_phys + n_left + n_right + n_top]
            T_bot[region_id] = Tnew[n_phys + n_left + n_right + n_top:]

            T[region_id] = np.concatenate(
                [T_phys[region_id], T_left[region_id], T_right[region_id], T_top[region_id], T_bot[region_id]])
            H[region_id] = Hnew
            K[region_id] = Knew

        # ✅ 边界条件统一处理
        for (region_id, direction), neighbor_data in boundary_neighbors.items():
            ghost_indices = neighbor_data["ghost_indices"]
            phys_indices = neighbor_data["phys_indices"]
            target_region = neighbor_data["target_region"]

            dr1 = zones[region_id]["dr"]
            dr2 = zones[target_region]["dr"]

            if direction == "left":
                T_left[region_id][ghost_indices] = T_phys[region_id][phys_indices]
            elif direction == "right":
                T_right[region_id][ghost_indices] = 2 * Tsurr - T_phys[region_id][phys_indices]
            elif direction == "top":
                if abs(dr1 - dr2) < tolerance:
                    T_top[region_id] = bc.interpolate_temperature_for_same(
                        T_top[region_id], T_phys[target_region], ghost_indices, phys_indices
                    )
                else:
                    T_top[region_id] = bc.interpolate_temperature_for_coarse_and_fine(
                        T_top[region_id], T_phys[target_region], ghost_indices, phys_indices
                    )
            elif direction == "bot":
                if abs(dr1 - dr2) < tolerance:
                    T_bot[region_id] = bc.interpolate_temperature_for_same(
                        T_bot[region_id], T_phys[target_region], ghost_indices, phys_indices
                    )
                else:
                    T_bot[region_id] = bc.interpolate_temperature_for_coarse_and_fine(
                        T_bot[region_id], T_phys[target_region], ghost_indices, phys_indices
                    )

        # ✅ 边界更新后再重新拼接
        for i in range(n_slices):
            T[i] = np.concatenate([T_phys[i], T_left[i], T_right[i], T_top[i], T_bot[i]])


    def plot_temperature_contour_in_circle(phys_coords_list, T_phys, radius, title='Temperature Contour', cmap='jet',
                                           levels=20):
        """
        只绘制半圆区域（r,z）内的温度场等高线图。
        - radius: 圆的半径（单位应和坐标单位一致）
        """
        # 1. 合并所有区域的物理点坐标
        all_coords = np.vstack([arr[:, :2] for arr in phys_coords_list])  # shape: (N, 2)

        # 2. 合并所有温度
        if isinstance(T_phys, dict):
            all_temps = np.concatenate([T_phys[i] for i in range(len(phys_coords_list))])
        elif isinstance(T_phys, list):
            all_temps = np.concatenate(T_phys)
        else:
            raise TypeError("T_phys 必须是 list 或 dict 类型")

        # 3. 创建规则网格
        r_vals = all_coords[:, 0]
        z_vals = all_coords[:, 1]
        r_lin = np.linspace(np.min(r_vals)+dr1/2, np.max(r_vals), 1000)
        z_lin = np.linspace(np.min(z_vals), np.max(z_vals), 1000)
        r_grid, z_grid = np.meshgrid(r_lin, z_lin)

        # 4. 插值
        T_grid = griddata(all_coords, all_temps, (r_grid, z_grid), method='linear')

        # 5. 掩码处理（只保留半圆区域）
        mask_circle = (r_grid ** 2 + (z_grid - radius) ** 2 <= radius ** 2 + tolerance) & \
                      (r_grid > 0) & (z_grid > 0) & (z_grid < 2 * radius)

        T_grid[~mask_circle] = np.nan  # 将圆外区域设为 NaN，不绘制

        # 6. 设置等高线等级
        vmin = np.nanmin(T_grid)
        vmax = np.nanmax(T_grid)
        level_values = np.linspace(vmin, vmax, levels)

        # 7. 绘图
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(r_grid, z_grid, T_grid, levels=level_values, cmap=cmap)

        cbar = plt.colorbar(contour)
        cbar.set_label(
            f"Temperature (K)\nMin: {vmin:.2f} K\nMax: {vmax:.2f} K",
            rotation=270, labelpad=30, va='bottom'
        )

        plt.xlabel('r (m)')
        plt.ylabel('z (m)')
        plt.title(title)
        plt.xlim([0, r])
        plt.ylim([0, 2 * r])
        plt.axis('equal')  # 可保留或移除，视需要定比
        plt.tight_layout()
        plt.show()


    # 假设你的圆半径为 r
    plot_temperature_contour_in_circle(phys_coords_list, T_phys, radius=r)
    end_time2 = time.time()
    print(f"Whole Calculation time = {end_time2 - start_time2:.2f}s")












































