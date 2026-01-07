import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator
import Physical_Field_Calculation as pfc  # Import module containing three field functions
import ADR
import generate_coordinates as gc
import GPU_accelerate as ga
from numba import njit, prange, set_num_threads
# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s, cs, ks = 1800.0, 1688.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 372.65
Tl = 373.65
L = 333

Lr, Lz = 0.1, 0.1        # Domain size in r and z directions (meters)
Nr, Nz = 100, 100       # Number of cells in r and z directions
dr, dz = Lr / Nr, Lz / Nz  # Cell size in r and z directions

E = 1e9                  # Elastic modulus [Pa]
nu = 0.25                # Poisson's ratio
K = 1.0                  # Thermal conductivity [W/(m·K)]
alpha = 1.8e-5           # Thermal expansion coefficient [1/K]

if Nz == 1:
    dz = 0  # Treat as 1D if z direction is disabled

delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes_x = 3        # Number of ghost cells in the x (or r) direction
ghost_nodes_z = 3        # Number of ghost cells in the z direction
h = 1


# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.0  # Starting position in r-direction
z_start = 0.0
tolerance = 1e-8

# ==========================
#   r (or x) direction
# ==========================

r_ghost_left = True
r_ghost_right = False
z_ghost_top = False
z_ghost_bot = True
r_all, z_all, Nr_tot, Nz_tot = gc.generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz, ghost_nodes_x, ghost_nodes_z,
    r_ghost_left, r_ghost_right,
    z_ghost_top, z_ghost_bot)
Nr_tot = len(r_all)
Nz_tot = len(z_all)
Rmat, Zmat = np.meshgrid(r_all, z_all, indexing='xy')

# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat = Rmat.flatten()
z_flat = Zmat.flatten()

indptr, indices, dist = ga.build_neighbor_csr(
    r_flat, z_flat, delta, dr, dz, tolerance
)
# Compute partial area overlap matrix
partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_flat, z_flat,
    dr, dz,
    delta,
    indptr, indices, dist,
    tolerance,
)

N = len(indptr) - 1
nnz = len(indices)

edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
row_sum = np.bincount(edge_i, weights=partial_area_matrix , minlength=N)
denom = row_sum[edge_i] + row_sum[indices]
 # 结果是一维 (N,) 各行的总和
c = (6 * E) / (np.pi * delta**3 * h * (1 - 2 * nu) * (1 + nu))
c_matrix = 2 * np.pi * delta**2 / denom * c

# 自身边满足：j == i  ⇔  indices[p] == edge_i[p]
partial_area_matrix[indices == edge_i] = 0.0

# ------------------------
# Initialize displacement, velocity, and acceleration fields
# ------------------------
Ur = np.zeros_like(Rmat).flatten()  # Radial displacement (1D)
Uz = np.zeros_like(Zmat).flatten()  # Axial displacement (1D)

Vr = np.zeros_like(Rmat).flatten()  # Radial velocity (1D)
Vz = np.zeros_like(Zmat).flatten()  # Axial velocity (1D)

Ar = np.zeros_like(Rmat).flatten()  # Radial acceleration (1D)
Az = np.zeros_like(Zmat).flatten()  # Axial acceleration (1D)


# Get ghost node indices from bc_funcs

ghost_inds_left, interior_inds_left, ghost_inds_left_1d = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x, Nz_tot)
ghost_inds_right, interior_inds_right, ghost_inds_right_1d = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x, Nz_tot)
ghost_inds_top, interior_inds_top, ghost_inds_top_1d = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z, Nr_tot)
ghost_inds_bottom, interior_inds_bottom, ghost_inds_bottom_1d = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z, Nr_tot)

# core function of thermal and mechanical

dir_r, dir_z = pfc.compute_direction_matrix(r_flat, z_flat, Ur, Uz, indptr, indices)

def compute_accelerated_velocity(
    Ur_curr, Uz_curr,
    r_flat, z_flat,
    indptr, indices, dist,              # CSR + 初始距离 d0
    c_flat, partial_area_flat,          # CSR 边数据
    rho, br, bz
):
    Ur_new = Ur_curr
    Uz_new = Uz_curr
    Relative_elongation = pfc.compute_s_matrix(r_flat, z_flat, Ur_new, Uz_new, indptr, indices, dist)
    ar_edge = dir_r * (c_flat * Relative_elongation * partial_area_flat) / rho
    az_edge = dir_z * (c_flat * Relative_elongation * partial_area_flat) / rho
    # ---- 6) 按行求和：得到每个 i 的 Ar/Az ----
    Ar_new = np.bincount(edge_i, weights=ar_edge, minlength=N) + np.asarray(br) / rho
    Az_new = np.bincount(edge_i, weights=az_edge, minlength=N) + np.asarray(bz) / rho

    return Ar_new, Az_new

# Apply initial boundary conditions
br = np.zeros_like(Rmat)
bz = np.zeros_like(Zmat)
# Pressure value
pressure = 1000e3/dz  # Pa, downward pressure
inds_top =[0]
bz[inds_top,:] = -pressure
br = br.flatten()
bz = bz.flatten()
dt_m = np.sqrt((2 * rho_s) / (np.pi * delta**2 * c)) * 0.1  # Time step in seconds
dt_th = cf.compute_dt_cr_th_solid_with_dist(
    rho_s, cs, ks,
    partial_area_matrix,
    indptr, dist,
    delta
)

Ar, Az = compute_accelerated_velocity(
    Ur, Uz,
    r_flat, z_flat,
    indptr, indices, dist,          # CSR 邻居 & 初始距离
    c_matrix, partial_area_matrix,      # CSR 边系数 & 面积
    rho_s, br, bz
)
lambda_diag_matrix = ADR.compute_lambda_diag_matrix(
    partial_area_matrix, dist, indptr, indices,
    r_flat, z_flat,
    c_matrix, 1
)

Fr_0 = Ar * rho_s
Fz_0 = Az * rho_s
Vr_half = (1 / 2) * (Fr_0 / lambda_diag_matrix)
Vz_half = (1 / 2) * (Fz_0 / lambda_diag_matrix)
Ur = Vr_half * 1 + Ur
Uz = Vz_half * 1 + Uz
Uz[ghost_inds_bottom_1d] = 0
Ur[ghost_inds_left_1d] = 0
Ur[ghost_inds_bottom_1d] = 0
Uz[ghost_inds_left_1d] = 0

# ------------------------
# Simulation loop settings
# ------------------------

total_time = 1000  # Total simulation time (e.g., 5 hours)
nsteps = int(2000)
print_interval = int(10 / dt_m)  # Print progress every 10 simulated seconds
start_time = time.time()

# ------------------------
# Time-stepping loop
# ------------------------
save_times = [2, 4, 6, 8, 10]  # Save snapshots (in hours)
save_steps = [int(t * 3600 / dt_m) for t in save_times]
T_record = []  # Store temperature snapshots

for step in range(nsteps):
    previous_Ur = Ur
    previous_Uz = Uz
    Ar, Az = compute_accelerated_velocity(
        Ur, Uz,
        r_flat, z_flat,
        indptr, indices, dist,  # CSR 邻居 & 初始距离
        c_matrix, partial_area_matrix,  # CSR 边系数 & 面积
        rho_s, br, bz
    )

    Fr = Ar * rho_s
    Fz = Az * rho_s
    cr_n = ADR.compute_local_damping_coefficient(Fr, Fr_0, Vr_half, lambda_diag_matrix, Ur, 1)
    cz_n = ADR.compute_local_damping_coefficient(Fz, Fz_0, Vz_half, lambda_diag_matrix, Uz, 1)
    Fr_0 = Fr
    Fz_0 = Fz
    Vr_half, Ur = ADR.adr_update_velocity_displacement(Ur, Vr_half, Fr, cr_n, lambda_diag_matrix, 1)
    Vz_half, Uz = ADR.adr_update_velocity_displacement(Uz, Vz_half, Fz, cz_n, lambda_diag_matrix, 1)

    Uz[ghost_inds_bottom_1d] = 0
    Ur[ghost_inds_left_1d] = 0
    Ur[ghost_inds_bottom_1d] = 0
    Uz[ghost_inds_left_1d] = 0

    #Ur[ghost_inds_right_1d] = 0
    #Ur[ghost_inds_left_1d] = 0


    #Ur, Uz, Vr_half, Vz_half = pfc.compute_next_displacement_field(Ur, Uz, Vr, Vz, Ar, Az,dt_m)
    # Vr, Vz, Ar, Az = pfc.compute_next_velocity_third_step(Vr_half, Vz_half, Ur, Uz, dt_m)
    #Uz[ghost_inds_bottom, :] = 0
    #Ur[:, ghost_inds_left] = 0
    # Ur[:, ghost_inds_right] = 0
    # 计算当前位移增量的RMS
    delta_Ur = Ur - previous_Ur
    delta_Uz = Uz - previous_Uz

    rms_increment = np.sqrt(np.mean(delta_Ur ** 2 + delta_Uz ** 2))

    if rms_increment < 1e-12:
        print(f"Convergence reached at step {step} with RMS displacement increment {rms_increment}")

        break
    if step % 10 == 0:
        print(f"Step {step}/{nsteps} completed")

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")

time = end_time - start_time

# ------------------------
# Post-processing: visualization
# ------------------------
mask = np.ones(Rmat.shape, dtype=bool)
# 将 ghost 点置为 False
mask[ghost_inds_top, :] = ~z_ghost_top
mask[ghost_inds_bottom, :] = ~z_ghost_bot
mask[:, ghost_inds_left] = ~r_ghost_left
mask[:, ghost_inds_right] = ~r_ghost_right

Ur = Ur.reshape(Rmat.shape)
Uz = Uz.reshape(Zmat.shape)
plot.plot_displacement_field(Rmat, Zmat, Ur, Uz,mask, Lr, Lz, title_prefix="Final Displacement", save=False)

# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""
