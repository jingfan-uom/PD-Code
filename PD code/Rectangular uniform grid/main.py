import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator

# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s, cs, ks = 1800.0, 1688.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 372.65
Tl = 373.65
L = 333

Lr, Lz = 0.1, 0.1        # Domain size in r and z directions (meters)
Nr, Nz = 40, 40          # Number of cells in r and z directions
dr, dz = Lr / Nr, Lz / Nz  # Cell size in r and z directions

E = 25.7e9                # 弹性模量 [Pa]
nu = 0.25                 # 泊松比
K = 1.0                    # 热传导系数 [W/(m·K)]

rho = 1800.0              # 密度 [kg/m³]
C = 1688.0                # 比热容 [J/(kg·K)]
alpha = 1.8e-5            # 热膨胀系数 [1/K]

if Nz == 1:
    dz = 0  # Treat as 1D if z direction is disabled

delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes_x = 3        # Number of ghost cells in the x (or r) direction
ghost_nodes_z = 3        # Number of ghost cells in the z direction

# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.  # Starting position in r-direction
tolerance = 1e-8

# ==========================
#   r (or x) direction
# ==========================
r_phys = np.linspace(r_start + dr/2, r_start + Lr - dr/2, Nr)
r_ghost_left = np.linspace(r_start - ghost_nodes_x * dr + dr/2, r_start - dr/2, ghost_nodes_x)
r_ghost_right = np.linspace(
    r_start + Lr + dr/2,
    r_start + Lr + dr/2 + (ghost_nodes_x - 1) * dr,
    ghost_nodes_x
)
r_all = np.concatenate([r_ghost_left, r_phys, r_ghost_right])
Nr_tot = len(r_all)

# ==========================
#   z direction
# ==========================
z_phys = np.linspace(Lz - dz/2, dz/2, Nz)
z_ghost_top = np.linspace(Lz + (ghost_nodes_z - 1) * dz + dz/2, Lz + dz/2, ghost_nodes_z)
z_ghost_bot = np.linspace(0 - dz/2, -ghost_nodes_z * dz + dz/2, ghost_nodes_z)
z_all = np.concatenate([z_ghost_top, z_phys, z_ghost_bot])
Nz_tot = len(z_all)

# Meshgrid
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
    # 1D: only consider radial distances
    dx_r = r_flat[:, None] - r_flat[None, :]
    distance_matrix = np.abs(dx_r)
else:
    # 2D: use Euclidean distance
    dx_r = r_flat[:, None] - r_flat[None, :]
    dx_z = z_flat[:, None] - z_flat[None, :]
    distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)

# Compute partial area overlap matrix
partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_flat, z_flat, dr, dz, delta, distance_matrix, tolerance
)
horizon_mask = (distance_matrix > tolerance) & (partial_area_matrix != 0.0)
true_indices = np.where(horizon_mask)

# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat, true_indices)
threshold_distance = np.sqrt(2) * dr
factor_mat = np.where(distance_matrix <= threshold_distance + tolerance, 1.125, 1.0)  # Local adjustment factor

# ------------------------
# Temperature update function
# ------------------------
def update_temperature(Tcurr, Hcurr, Kcurr):
    flux = Kcurr @ Tcurr.flatten()               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_tot, Nr_tot)         # Reshape to 2D
    flux[(flux > -tolerance) & (flux < tolerance)] = 0  # Eliminate small fluctuations

    Hnew = Hcurr + flux                          # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_s, cs, cl, L, Ts, Tl)   # Convert to temperature

    # Apply boundary conditions
    Tnew = bc_funcs.apply_bc_dirichlet_mirror(Tnew, ghost_inds_top, interior_inds_top,  274.15, 0)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_bottom, interior_inds_bottom, axis=0)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_left, interior_inds_left, axis=1)
    Tnew = bc_funcs.apply_bc_dirichlet_mirror(Tnew, ghost_inds_right, interior_inds_right, 274.15, 1)

    Knew = Kcurr

    return Tnew, Hnew, Knew

# ------------------------
# Initialization
# ------------------------

T = np.full(Rmat.shape, 283.15)  # Initial temperature field (uniform 200 K)

# Get ghost node indices from bc_funcs
ghost_inds_top, interior_inds_top = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_bottom, interior_inds_bottom = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_left, interior_inds_left = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x)
ghost_inds_right, interior_inds_right = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x)

# Apply initial boundary conditions
T = bc_funcs.apply_bc_dirichlet_mirror(T, ghost_inds_top, interior_inds_top,  274.15, 0)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_bottom, interior_inds_bottom, axis=0)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_left, interior_inds_left, axis=1)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_right, interior_inds_right, axis=1)

# Only apply Dirichlet on right side within this z region

H = cf.get_enthalpy(T, rho_s, cs, cl, L, Ts, Tl)  # Initial enthalpy

# ------------------------
# Simulation loop settings
# ------------------------
dt = 5  # Time step in seconds

# Build initial conductivity matrix
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                         partial_area_matrix, shape_factor_matrix,
                         distance_matrix, horizon_mask, true_indices, r_flat,
                         ks, kl, Ts, Tl, delta, dz, dt)



total_time = 200  # Total simulation time (5 hours)
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
    T, H, Kmat = update_temperature(T, H,Kmat)

    if step in save_steps:
        T_record.append(T.copy())

    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * dt:.2f}s")
    if step == 100/dt:
        aa=1
end_time = time.time()



print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")
time = end_time - start_time
# ------------------------
# Post-processing: visualization
# ------------------------
plot.temperature(Rmat, Zmat, T, total_time, nsteps, dr,dz,time)

# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""
