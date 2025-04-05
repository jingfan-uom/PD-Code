import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
# ------------------------
#  Physical and simulation parameters
# ------------------------
k_mat = 50.0             # Thermal conductivity (W/m·K)
rho_mat = 7850.0         # Density (kg/m³)
Cp_mat = 420.0           # Specific heat capacity (J/kg·K)
Lr, Lz = 0.8, 0.8        # Domain size in r and z directions (meters)
Nr, Nz = 40, 40        # Number of cells in r and z directions
dr, dz = Lr / Nr, Lz / Nz    # Cell size in r and z
if Nz  == 1:
    dz = 0
delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes_x = 3  # x(或 r)方向需要的幽单元层数
ghost_nodes_z = 3  # z方向需要的幽单元层数          # Number of ghost cells for boundary treatment

# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.2  # Start position in r-direction

# ==========================
#   x (或 r) 方向
# ==========================
r_phys = np.linspace(r_start + dr/2, r_start + Lr - dr/2, Nr)
r_ghost_left = np.linspace(r_start - ghost_nodes_x*dr + dr/2,r_start - dr/2,ghost_nodes_x)
r_ghost_right = np.linspace(
    r_start + Lr + dr/2,
    r_start + Lr + dr/2 + (ghost_nodes_x - 1)*dr,
    ghost_nodes_x
)
r_all = np.concatenate([r_ghost_left, r_phys, r_ghost_right])
Nr_tot = len(r_all)
# ==========================
#   z 方向
# ==========================
z_phys = np.linspace(Lz - dz/2, dz/2, Nz)
z_ghost_top = np.linspace(Lz + (ghost_nodes_z - 1)*dz + dz/2, Lz + dz/2,ghost_nodes_z)
z_ghost_bot = np.linspace(
    0 - dz/2,
    -ghost_nodes_z*dz + dz/2,
    ghost_nodes_z
)
z_all = np.concatenate([z_ghost_top, z_phys, z_ghost_bot])
Nz_tot = len(z_all)

if len(z_all) == 1:
    Rmat = r_all
    Zmat = z_all
else:
    Rmat, Zmat = np.meshgrid(r_all, z_all, indexing='xy')


# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat = Rmat.flatten()
z_flat = Zmat.flatten()

if Nz_tot == 1:
    # 一维计算，仅考虑 r 方向的距离
    dx_r = r_flat[:, None] - r_flat[None, :]
    distance_matrix = np.abs(dx_r)  # 一维距离
else:
    # 二维计算，考虑 r 和 z 方向
    dx_r = r_flat[:, None] - r_flat[None, :]
    dx_z = z_flat[:, None] - z_flat[None, :]
    distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)

# Mask: which points are within the horizon (excluding self)
horizon_mask = (distance_matrix > 1e-6) & (distance_matrix <= delta + 1e-6)
true_indices = np.where(horizon_mask)


# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat, true_indices)
partial_area_matrix = cf.compute_partial_area_matrix(r_flat, z_flat, dr, dz, delta, distance_matrix)
threshold_distance = np.sqrt(2) * dr
factor_mat = np.where(distance_matrix <= threshold_distance + 1e-4, 1.5, 1.0)  # Local adjustment factor

# ------------------------
# Temperature update function
# ------------------------
def update_temperature(Tcurr, Hcurr, Kmat):
    flux = Kmat @ Tcurr.flatten() * dt               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_tot, Nr_tot)              # Reshape back to 2D
    Hnew = Hcurr + flux                              # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_mat, Cp_mat) # Convert to temperature
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_top, interior_inds_top, axis=0)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_bottom, interior_inds_bottom, axis=0)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_left, interior_inds_left, axis=1)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_right, interior_inds_right, axis=1)
    Tnew = bc_funcs.apply_bc_dirichlet_mirror(Tnew, ghost_inds_right, interior_inds_right,
                                           T_bc=500.0, axis=1, z_mask=z_mask)
    return Tnew, Hnew

# ------------------------
# Initialization
# ------------------------
T = np.full(Rmat.shape, 200)  # Initial temperature field (uniform 200 K)
# 在这里调用 bc_funcs 文件中的函数,对边界条件进行初始化
# ------------------------------------
ghost_inds_top, interior_inds_top = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_bottom, interior_inds_bottom = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_left, interior_inds_left = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x)
ghost_inds_right, interior_inds_right = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x)

T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_top, interior_inds_top, axis=0)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_bottom, interior_inds_bottom, axis=0)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_left, interior_inds_left, axis=1)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_right, interior_inds_right, axis=1)
z_mask = (z_all >= 0.3 - 1e-6) & (z_all <= 0.5 + 1e-6)
T = bc_funcs.apply_bc_dirichlet_mirror(T, ghost_inds_right, interior_inds_right,
                                           T_bc=500.0, axis=1, z_mask=z_mask)


H = cf.get_enthalpy(T, rho_mat, Cp_mat)  # Initial enthalpy
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                         partial_area_matrix, shape_factor_matrix,
                         distance_matrix, horizon_mask, true_indices, r_flat,
                         k_mat, delta)  # Initial stiffness matrix

# ------------------------
# Simulation loop settings
# ------------------------
dt = 5                     # Time step (s)
total_time = 5 * 3600      # Total simulation time (10 hours)
nsteps = int(total_time / dt)
print_interval = int(10 / dt)  # Print progress every 10 seconds of simulated time
print(f"Total steps: {nsteps}")
start_time = time.time()
# ------------------------
# Time-stepping loop

for step in range(nsteps):
    T, H = update_temperature(T, H, Kmat)
    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * (dt):.2f}s")

end_time = time.time()
print(f"Calculation finished, elapsed real time={end_time - start_time:.2f}s")

# ------------------------
# Post-processing: visualization
# ------------------------
plot.temperature(Rmat, Zmat, T, total_time, nsteps,dt)
