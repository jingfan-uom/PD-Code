import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import time
import core_funcs_1 as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator
import Physical_Field_Calculation as pfc  # Import module containing three field functions
import ADR
import generate_coordinates as gc

import matplotlib
matplotlib.use('TkAgg')  # 或者试试 'Qt5Agg'


# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s = 897.6           # 密度 [kg/m³]
Lr, Lz = 0.05, 0.05        # 模型区域尺寸 (m)
Nr, Nz = 60, 60            # 网格数量
dr, dz = Lr / Nr, Lz / Nz  # 单元尺寸

E = 5.5e9                  # 冰的杨氏模量 (Pa)
nu = 1/4                   # 泊松比
KI = 134e3                 # 断裂韧性 N/m^1.5
G0 = KI**2 / E             # 能量释放率 (J/m²)

alpha = 1.8e-5             # 热膨胀系数 [1/K]（如不确定可保留原值）

delta = 5 * dr         # 非局部作用范围
ghost_nodes_x = 5
ghost_nodes_z = 5
h = 1

# 使用公式 (2-82) 重新计算裂纹起裂应力 s0
s0 = np.sqrt(5 * np.pi * G0 / (18 * E * delta))
# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.0
z_start = 0.0    # Starting position in r-direction
tolerance = 1e-12

# ==========================
#  Mechanical field coordinates
# ==========================
r_ghost_left_m = False
r_ghost_right_m = False
z_ghost_top_m = True
z_ghost_bot_m = True
r_all_m, z_all_m, Nr_tot_m, Nz_tot_m = gc.generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz, ghost_nodes_x, ghost_nodes_z,
    r_ghost_left_m, r_ghost_right_m,
    z_ghost_top_m, z_ghost_bot_m)
Nr_tot_m = len(r_all_m)
Nz_tot_m = len(z_all_m)
Rmat_m, Zmat_m = np.meshgrid(r_all_m, z_all_m, indexing='xy')
# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat_m = Rmat_m.flatten()
z_flat_m= Zmat_m.flatten()
# 2D: use Euclidean distance
dx_r_m = r_flat_m[None, :] - r_flat_m[:, None]
dx_z_m = z_flat_m[None, :] - z_flat_m[:, None]
distance_matrix_m = np.sqrt(dx_r_m ** 2 + dx_z_m ** 2)
# Compute partial area overlap matrix
partial_area_matrix_m = area_matrix_calculator.compute_partial_area_matrix(
    r_flat_m, z_flat_m, dr, dz, delta, distance_matrix_m, tolerance)
horizon_mask_m = ((distance_matrix_m > tolerance) & (partial_area_matrix_m != 0.0))

row_sum = np.sum(partial_area_matrix_m, axis=1)  # (N,)
matrix_sum = row_sum[:, None] + row_sum[None, :]  # (N, N)
 # 结果是一维 (N,) 各行的总和
c = (12 * E) / (np.pi * delta**4 * h )
c_matrix = 2 * np.pi * delta**2 / matrix_sum * c

mu = np.zeros_like(horizon_mask_m, dtype=int) # Failure matrix
mu[horizon_mask_m] = 1
crack_start = [0.02, 0.025]
crack_end = [0.03, 0.025]
mu = pfc.mark_prebroken_bonds_from_mesh(mu, r_flat_m, z_flat_m, horizon_mask_m, crack_start, crack_end)


"""
false_indices_2d = np.where(mask_corner == False)
false_indices_flat = np.ravel_multi_index(false_indices_2d, mask_corner.shape)

for idx in false_indices_flat:
    horizon_mask[idx, :] = False
    horizon_mask[:, idx] = False
"""
# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------

threshold_distance = np.sqrt(2) * dr
 # Local adjustment factor
condition = distance_matrix_m <= (threshold_distance + tolerance)

# ------------------------
# Initialize displacement, velocity, and acceleration fields, T
# ------------------------
Ur = np.zeros_like(Rmat_m).flatten()  # Radial displacement (1D)
Uz = np.zeros_like(Rmat_m).flatten()  # Axial displacement (1D)
Vr = np.zeros_like(Rmat_m).flatten()  # Radial velocity (1D)
Vz = np.zeros_like(Rmat_m).flatten()  # Axial velocity (1D)
Ar = np.zeros_like(Rmat_m).flatten()  # Radial acceleration (1D)
Az = np.zeros_like(Rmat_m).flatten()  # Axial acceleration (1D)

"""
br = np.zeros_like(Rmat_m)
bz = np.zeros_like(Zmat_m)
# Pressure value
pressure = 1000e3/dz  # Pa, downward pressure
inds_top =[0]
bz[inds_top,:] = 0
br = br.flatten()
bz = bz.flatten()
"""
# Get ghost node indices from bc_funcs
ghost_inds_left_m, interior_inds_left_m, ghost_inds_left_1d_m = bc_funcs.get_left_ghost_indices(r_all_m, ghost_nodes_x, Nz_tot_m)
ghost_inds_right_m, interior_inds_right_m, ghost_inds_right_1d_m = bc_funcs.get_right_ghost_indices(r_all_m, ghost_nodes_x, Nz_tot_m)
ghost_inds_top_m, interior_inds_top_m, ghost_inds_top_1d_m = bc_funcs.get_top_ghost_indices(z_all_m, ghost_nodes_z, Nr_tot_m)
ghost_inds_bottom_m, interior_inds_bottom_m, ghost_inds_bottom_1d_m = bc_funcs.get_bottom_ghost_indices(z_all_m, ghost_nodes_z, Nr_tot_m)

dir_r, dir_z = pfc.compute_direction_matrix(r_flat_m, z_flat_m, Ur, Uz, horizon_mask_m)

# core function of thermal and mechanical
def compute_accelerated_velocity(Ur_curr, Uz_curr,mu):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr
    Relative_elongation = pfc.compute_s_matrix(r_flat_m, z_flat_m, Ur_new, Uz_new, horizon_mask_m,distance_matrix_m)
    mu = pfc.update_mu_by_failure(mu, Relative_elongation, s0)
    Ar_new = mu * dir_r * c_matrix * (Relative_elongation) * partial_area_matrix_m / rho_s
    Az_new = mu * dir_z * c_matrix * (Relative_elongation) * partial_area_matrix_m / rho_s

    Ar_new = np.sum(Ar_new, axis=1).reshape(Ur_curr.shape)  # Shape matches Ur_curr
    Az_new = np.sum(Az_new, axis=1).reshape(Uz_curr.shape)

    return Ar_new, Az_new,mu  # Or return other desired quantities

def compute_next_displacement_field(Ur_curr, Uz_curr, Vr_curr, Vz_curr, Ar_new, Az_new):
    Vr_half = Vr_curr + 0.5 * dt_m * Ar_new
    Vz_half = Vz_curr + 0.5 * dt_m * Az_new
    Ur_next = Ur_curr + dt_m * Vr_half
    Uz_next = Uz_curr + dt_m * Vz_half
    return Ur_next, Uz_next, Vr_half, Vz_half

def compute_next_velocity_third_step(Vr_half, Vz_half, Ur_next, Uz_next, dt_m,mu):
    Ar_next, Az_next,mu = compute_accelerated_velocity(Ur_next, Uz_next, mu)
    Vr_new = Vr_half + 0.5 * dt_m * Ar_next
    Vz_new = Vz_half + 0.5 * dt_m * Az_next
    return Vr_new, Vz_new, Ar_next, Az_next

# time_step calculation
dt_m = 5e-9 # Time step in seconds
dt = dt_m
# ------------------------
# Simulation loop settings
# ------------------------
total_time = 200 # Total simulation time (e.g., 5 hours)

nsteps_m = int(100000)
print_interval = int(total_time / dt_m)  # Print progress every 10 simulated seconds
start_time = time.time()

# ------------------------
# Time-stepping loop
# ------------------------
mask_m = np.ones(Rmat_m.shape, dtype=bool)
mask_m[ghost_inds_top_m, :] = ~z_ghost_top_m
mask_m[ghost_inds_bottom_m, :] = ~z_ghost_bot_m
mask_m[:, ghost_inds_left_m] = ~r_ghost_left_m
mask_m[:, ghost_inds_right_m] = ~r_ghost_right_m
area_matrix_m = partial_area_matrix_m.copy()  # 避免影响原数组
np.fill_diagonal(area_matrix_m, 0)
save_times = [9.0e-6,1.0e-5, 1.1e-5, 1.2e-5, 1.3e-5,1.4e-5, 1.5e-5,1.6e-5, 1.7e-5]
x_list = []
y_list = []
phi = 1 - np.sum(mu * area_matrix_m, axis=1) / np.sum(area_matrix_m, axis=1)
phi = phi.reshape(Rmat_m.shape)
for step1 in range(nsteps_m):
    previous_Ur = Ur.copy()
    previous_Uz = Uz.copy()
    Ar, Az, mu = compute_accelerated_velocity(Ur, Uz,mu)
    Ur, Uz, Vr_half, Vz_half = compute_next_displacement_field(Ur, Uz, Vr, Vz, Ar, Az)
    Vr, Vz, Ar, Az = compute_next_velocity_third_step(Vr_half, Vz_half, Ur, Uz, dt_m,mu)
    Vz[ghost_inds_bottom_1d_m] = -0.5
    Vz[ghost_inds_top_1d_m] = 0.5
    dir_r, dir_z = pfc.compute_direction_matrix(r_flat_m, z_flat_m, Ur, Uz, horizon_mask_m)
    phi = 1 - np.sum(mu * area_matrix_m, axis=1) / np.sum(area_matrix_m, axis=1)
    phi = phi.reshape(Rmat_m.shape)
    if step1 % 10 == 0:
        print(f"Computed up to {step1 * dt_m:.2e} seconds (step {step1})")
        mask = phi >= 0.5
        # 用 mask 取出对应的 r 坐标
        r_selected = Rmat_m[mask]
        if r_selected.size > 0:
            result = np.max(r_selected) - 0.03
            x = step1 * dt_m
            y = result
            x_list.append(x)
            y_list.append(y)
            print(f"{result:.6f}")
        else:
            print("no phi >= 0.5 ")

    for t in save_times:
        if abs(step1 * dt_m - t) < 1e-10:
            filename = f"C:/Users/郑靖凡/Desktop/mu_field_{t:.1e}.png"
            plot.plot_mu_field(Rmat_m, Zmat_m, phi, mask_m, Lr, Lz,
                               title_prefix=f"Bond Failure State t={t:.1e}s",
                               save=True,
                               filename=filename)
            print(f"Saved figure at {filename}")
    if step1 * dt_m > 1.71e-5:
        break

data = np.column_stack((x_list, y_list))
desktop = os.path.expanduser("~/Desktop")
save_path = os.path.join(desktop, "r_vs_time.csv")
np.savetxt(save_path, data, delimiter=',', header="time,r", comments='')

print(f"数据已保存到: {save_path}")

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")
time = end_time - start_time

Ur = Ur.reshape(Rmat_m.shape)
Uz = Uz.reshape(Zmat_m.shape)
plot.plot_displacement_field(Rmat_m, Zmat_m, Ur, Uz, mask_m, Lr, Lz, title_prefix="Final Displacement", save=False)
U = np.sqrt(Ur**2 + Uz**2)

