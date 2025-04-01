import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import  plot_utils as plot

# 模拟区域和物理参数
k_mat = 50.0
rho_mat = 7850.0
Cp_mat = 420.0
Lr, Lz = 0.8, 0.8
Nr, Nz = 40, 40
dr, dz = Lr / Nr, Lz / Nz
delta = 3 * dr
ghost_nodes = 3
# 构建坐标
r_start = 0.2
r_phys = np.linspace(r_start + dr / 2, r_start + Lr - dr / 2, Nr)
r_ghost_left = np.linspace(r_start - ghost_nodes * dr + dr / 2, r_start - dr / 2, ghost_nodes)
r_ghost_right = np.linspace(r_start + Lr + dr / 2, r_start + Lr + dr / 2 + (ghost_nodes - 1) * dr, ghost_nodes)

z_phys = np.linspace(Lz - dz / 2, dz / 2, Nz)
z_ghost_top = np.linspace(Lz + (ghost_nodes - 1) * dz + dz / 2, Lz + dz / 2, ghost_nodes)
z_ghost_bot = np.linspace(0 - dz / 2, -ghost_nodes * dz + dz / 2, ghost_nodes)

r_all = np.concatenate([r_ghost_left, r_phys, r_ghost_right])
z_all = np.concatenate([z_ghost_top, z_phys, z_ghost_bot])

Nr_tot = len(r_all)
Nz_tot = len(z_all)

Rmat, Zmat = np.meshgrid(r_all, z_all, indexing='xy')

# 计算距离和 horizon mask
r_flat = Rmat.flatten()
z_flat = Zmat.flatten()

dx_r = r_flat[:, None] - r_flat[None, :]
dx_z = z_flat[:, None] - z_flat[None, :]

distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)
horizon_mask = (distance_matrix > 0) & (distance_matrix <= delta + 1e-6)
true_indices = np.where(horizon_mask)

# 预处理
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat, true_indices)

partial_area_matrix = cf.compute_partial_area_matrix(r_flat, z_flat, dr, dz, delta, distance_matrix)
threshold_distance = np.sqrt(2) * dr
factor_mat = np.where(distance_matrix <= threshold_distance, 1.125, 1.0)

def update_temperature(Tcurr, Hcurr, Kmat):
    flux = Kmat @ Tcurr.flatten() * dt
    flux = flux.reshape(Nz_tot, Nr_tot)
    Hnew = Hcurr + flux
    Tnew = cf.get_temperature(Hnew, rho_mat, Cp_mat)
    Tnew = cf.apply_mixed_bc(Tnew, z_all, Nr_tot, Nz_tot, ghost_nodes)
    return Tnew, Hnew

# 初始化温度和通量矩阵
T = np.full(Rmat.shape, 200)
T = cf.apply_mixed_bc(T, z_all, Nr_tot, Nz_tot, ghost_nodes)
H = cf.get_enthalpy(T, rho_mat, Cp_mat)
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                      partial_area_matrix, shape_factor_matrix,
                      distance_matrix, horizon_mask, true_indices, r_flat,
                      k_mat, delta)

dt = 10
total_time = 5 * 3600
nsteps = int(total_time / dt)
print_interval = int(10 / dt)

print(f"Total steps: {nsteps}")
start_time = time.time()
for step in range(nsteps):
    T, H = update_temperature(T, H, Kmat)
    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * dt:.2f}s")
end_time = time.time()
print(f"Calculation finished, elapsed real time={end_time - start_time:.2f}s")

plot.temperature(Rmat, Zmat, T, total_time, nsteps)
