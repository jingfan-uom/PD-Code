import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator
import grid_generator
# ------------------------
# Set material properties arrays (match shape of Rmat)
# ------------------------
rho_s, cs, ks = 1000.0, 2060.0, 2.14
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 273.1
Tl = 273.2
L = 333

Lr, Lz = 0.4, 0.4        # Domain size in r and z directions (meters)
r_start, z_start = 0.0, 0.0  # Starting positions in r and z

# Coarse and fine grid spacing
dr_coarse, dz_coarse = 0.04, 0.04
dr_fine, dz_fine = 0.01, 0.01
tolerance= 1e-8
# Define fine region (set to None or empty if no fine region)
def fine_mask_func(r, z):
    return (z >= 0.15) and (r >= 0.1)
fine_mask_func = fine_mask_func
delta_fine = 3 * dr_fine
delta_coarse = 3 * dr_coarse
# Ghost node number
ghost_nodes_r, ghost_nodes_z = 3, 3

# ==========================
#   r (or x) direction
# ==========================
grid_info = grid_generator.generate_fine_and_coarse_grids(r_start, z_start,Lr, Lz,dr_coarse, dz_coarse,dr_fine, dz_fine,fine_mask_func, delta=3)

fine_mask_phys = grid_info["fine_mask_phys"]
fine_mask_interface_boundary = grid_info["fine_mask_intf_boundary"]

coarse_mask_phys = grid_info["coarse_mask_phys"]
coarse_mask_interface_boundary = grid_info["coarse_mask_intf_boundary"]

# 坐标矩阵
fine_phys_coords = grid_info["fine_phys_coords"]
fine_intf_coords = grid_info["fine_intf_coords"]

coarse_phys_coords = grid_info["coarse_phys_coords"]
coarse_intf_coords = grid_info["coarse_intf_coords"]


# ------------------------
# Define material region (0=water, 1=ice)
# ------------------------
TEMP_ICE  = 268.15   # 细网格物理域（冰）
TEMP_WATER = 373.15  # 粗网格物理域（热水）

# 各自温度向量
temp_fine_phys   = np.full(fine_phys_coords.shape[0],   TEMP_ICE)
temp_coarse_phys = np.full(coarse_phys_coords.shape[0], TEMP_WATER)

# === 2) 计算 “粗网格接口点” 的温度 ======================================
# 依赖前面写好的 4 点均值插值函数：
# interpolate_coarse_interface_temperature(...)
coarse_intf_temps = grid_generator.interpolate_coarse_interface_temperature(
    coarse_intf_coords,      # 红点（粗接口点）
    fine_phys_coords,        # 绿点（细物理域）
    temp_fine_phys,          # 细物理域温度
    dr_fine,
    dz_fine
)

# === 3) 计算 “细网格接口点” 的温度 ======================================
# 依赖前面写好的 IDW 插值函数：
# interpolate_fine_interface_temperature_idw(...)
fine_intf_temps = grid_generator.interpolate_fine_interface_temperature(
    fine_intf_coords,        # 细接口点
    coarse_phys_coords,      # 粗物理域坐标
    temp_coarse_phys,        # 粗物理域温度
    dr_coarse,
    dz_coarse,
    power=1.0                # IDW 权重指数；1=线性衰减，2=平方反比
)
# ------------------------
# Compute distance matrix and horizon mask
# ------------------------

# 组合坐标
fine_all_coords = np.vstack((fine_phys_coords, fine_intf_coords))  # shape = (N1 + N2, 2)
coarse_all_coords = np.vstack((coarse_phys_coords, coarse_intf_coords))  # shape = (N1 + N2, 2)


# 拆分为 r 和 z 分量
r_fine = fine_all_coords[:, 0]
z_fine = fine_all_coords[:, 1]
r_coarse = fine_all_coords[:, 0]
z_coarse = fine_all_coords[:, 1]
# 构造距离矩阵（欧氏距离）
dx_r_fine = r_fine[:, None] - r_fine[None, :]
dx_z_fine = z_fine[:, None] - z_fine[None, :]
dx_r_coarse= r_coarse[:, None] - r_coarse[None, :]
dx_z_coarse= z_coarse[:, None] - z_coarse[None, :]

fine_distance_matrix = np.sqrt(dx_r_fine**2 + dx_z_fine**2)  # shape = (N1 + N2, N1 + N2)
coarse_distance_matrix = np.sqrt(dx_r_coarse**2 + dx_z_coarse**2)  # shape = (N1 + N2, N1 + N2)


# Compute partial area overlap matrix
fine_partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_fine, z_fine, dr_fine, dz_fine, delta_fine, fine_distance_matrix, tolerance
)
coarse_partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_coarse, z_coarse, dr_coarse, dz_coarse, delta_coarse, coarse_distance_matrix, tolerance
)

fine_horizon_mask = (fine_distance_matrix > tolerance) & (fine_partial_area_matrix != 0.0)
coarse_horizon_mask = (coarse_distance_matrix > tolerance) & (coarse_partial_area_matrix != 0.0)

fine_true_indices = np.where(fine_horizon_mask)
coarse_true_indices = np.where(coarse_horizon_mask)

# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
fine_tshape_factor_matrix = cf.compute_shape_factor_matrix(r_fine, fine_true_indices)
coarse_shape_factor_matrix = cf.compute_shape_factor_matrix(r_coarse, coarse_true_indices)

fine_threshold_distance = np.sqrt(2) * dr_fine
coarse_threshold_distance = np.sqrt(2) * dr_coarse

fine_factor_mat = np.where(fine_distance_matrix <= fine_threshold_distance + tolerance, 1.125, 1.0)  # Local adjustment factor
coarse_factor_mat = np.where(coarse_distance_matrix <= coarse_threshold_distance + tolerance, 1.125, 1.0)  # Local adjustment factor
# ------------------------
# Temperature update function
# ------------------------
def update_temperature(Tcurr, Hcurr, Kmat):

    Knew = Kmat
    flux = Knew @ Tcurr.flatten()               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_tot, Nr_tot)         # Reshape to 2D
    flux[(flux > -tolerance) & (flux < tolerance)] = 0  # Eliminate small fluctuations
    Hnew = Hcurr + flux                          # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_s , cs , cl, L, Ts, Tl)
  # Convert to temperature

    # Apply boundary conditions
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_top, interior_inds_top, axis=0)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_bottom, interior_inds_bottom, axis=0)
    #Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_left, interior_inds_left, axis=1)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_right, interior_inds_right, axis=1)

    Knew = cf.build_K_matrix(Tnew, cf.compute_thermal_conductivity_matrix, factor_mat,
                             partial_area_matrix, shape_factor_matrix,
                             distance_matrix, horizon_mask, true_indices, ks, kl, Ts, Tl, delta, r_flat,
                             dt,dz)  # k_mat 无需再传

    return Tnew, Hnew ,Knew


# Get ghost node indices from bc_funcs
ghost_inds_top, interior_inds_top = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_bottom, interior_inds_bottom = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z)
#ghost_inds_left, interior_inds_left = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x)
ghost_inds_right, interior_inds_right = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x)

# Apply initial boundary conditions
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_top, interior_inds_top, axis=0)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_bottom, interior_inds_bottom, axis=0)
#T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_left, interior_inds_left, axis=1)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_right, interior_inds_right, axis=1)

H = cf.get_enthalpy(T, rho_l, cs, cl, L, Ts, Tl) # Initial enthalpy

# ------------------------
# Simulation loop settings
# ------------------------
dt = 2  # Time step in seconds

# Build initial conductivity matrix
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                             partial_area_matrix, shape_factor_matrix,
                             distance_matrix, horizon_mask, true_indices, ks, kl, Ts, Tl, delta, r_flat,
                             dt,dz)   # dz is used to determine whether a one-dimensional


total_time =  1200  # Total simulation time (5 hours)
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
    T, H, Kmat = update_temperature(T, H, Kmat)

    if step in save_steps:
        T_record.append(T.copy())

    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * dt:.2f}s")

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")

# ------------------------
# Post-processing: visualization
# ------------------------
plot.temperature(Rmat, Zmat, T, total_time, nsteps, dt)

# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""
