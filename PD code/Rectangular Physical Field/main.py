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

E = 1e6                  # Elastic modulus [Pa]
nu = 0.25                # Poisson's ratio
K = 1.0                  # Thermal conductivity [W/(m·K)]
alpha = 1.8e-5           # Thermal expansion coefficient [1/K]

if Nz == 1:
    dz = 0  # Treat as 1D if z direction is disabled

delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes_x = 3        # Number of ghost cells in the x (or r) direction
ghost_nodes_z = 3        # Number of ghost cells in the z direction
h = 1
c = (6 * E) / (np.pi * delta**3 * h * (1 - nu))

# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.0  # Starting position in r-direction
tolerance = 1e-8

# ==========================
#   r (or x) direction
# ==========================
r_phys = np.linspace(r_start + dr / 2, r_start + Lr - dr / 2, Nr)
r_ghost_left = np.linspace(r_start - ghost_nodes_x * dr + dr / 2, r_start - dr / 2, ghost_nodes_x)
r_ghost_right = np.linspace(
    r_start + Lr + dr / 2,
    r_start + Lr + dr / 2 + (ghost_nodes_x - 1) * dr,
    ghost_nodes_x
)
r_all = np.concatenate([r_ghost_left, r_phys, r_ghost_right])
Nr_tot = len(r_all)

# ==========================
#   z direction
# ==========================
z_phys = np.linspace(Lz - dz / 2, dz / 2, Nz)
z_ghost_top = np.linspace(Lz + (ghost_nodes_z - 1) * dz + dz / 2, Lz + dz / 2, ghost_nodes_z)
z_ghost_bot = np.linspace(0 - dz / 2, -ghost_nodes_z * dz + dz / 2, ghost_nodes_z)
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

def compute_accelerated_velocity(Ur_curr, Uz_curr):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr

    Relative_elongation = pfc.compute_s_matrix(Rmat, Zmat, Ur_new, Uz_new, horizon_mask)

    Ar_new = dir_r * c * (Relative_elongation) * partial_area_matrix / rho_s
    Az_new = dir_z * c * (Relative_elongation) * partial_area_matrix / rho_s

    Ar_new = np.sum(Ar_new, axis=1).reshape(Ur_curr.shape)  # Shape matches Ur_curr
    Az_new = np.sum(Az_new, axis=1).reshape(Uz_curr.shape)

    return Ar_new, Az_new  # Or return other desired quantities

def compute_next_displacement_field(Ur_curr, Uz_curr, Vr_curr, Vz_curr, Ar_new, Az_new):
    Vr_half = Vr_curr + 0.5 * dt * Ar_new
    Vz_half = Vz_curr + 0.5 * dt * Az_new
    Ur_next = Ur_curr + dt * Vr_half
    Uz_next = Uz_curr + dt * Vz_half
    return Ur_next, Uz_next, Vr_half, Vz_half

def compute_next_velocity_third_step(Vr_half, Vz_half, Ur_next, Uz_next, dt):
    Ar_next, Az_next = compute_accelerated_velocity(Ur_next, Uz_next)
    Vr_new = Vr_half + 0.5 * dt * Ar_next
    Vz_new = Vz_half + 0.5 * dt * Az_next
    return Vr_new, Vz_new, Ar_next, Az_next

# ------------------------
# Initialize displacement, velocity, and acceleration fields
# ------------------------
Ur = np.zeros_like(Rmat)  # Radial displacement
Uz = np.zeros_like(Rmat)  # Axial displacement

Vr = np.zeros_like(Rmat)  # Radial velocity
Vz = np.zeros_like(Rmat)  # Axial velocity

Ar = np.zeros_like(Rmat)  # Radial acceleration
Az = np.zeros_like(Rmat)  # Axial acceleration

dir_r, dir_z = pfc.compute_direction_matrix(Rmat, Zmat, horizon_mask)

# Get ghost node indices from bc_funcs
ghost_inds_top, interior_inds_top = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_bottom, interior_inds_bottom = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_left, interior_inds_left = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x)
ghost_inds_right, interior_inds_right = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x)

# Apply initial boundary conditions
# Fix left, right, and bottom boundaries (U = 0)
Ur = bc_funcs.apply_bc_dirichlet_displacement(Ur, ghost_inds_left, interior_inds_left, U_target=0.0, axis=1)
Ur = bc_funcs.apply_bc_dirichlet_displacement(Ur, ghost_inds_right, interior_inds_right, U_target=0.0, axis=1)
Uz = bc_funcs.apply_bc_dirichlet_displacement(Uz, ghost_inds_bottom, interior_inds_bottom, U_target=0.0, axis=0)

# Apply a small upward displacement at the top boundary, e.g., 0.000001 m
Uz = bc_funcs.apply_bc_dirichlet_displacement(Uz, ghost_inds_top, interior_inds_top, U_target=0.000001, axis=0)

# ------------------------
# Simulation loop settings
# ------------------------
dt = 1  # Time step in seconds
total_time = 100  # Total simulation time (e.g., 5 hours)
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

    Ar, Az = compute_accelerated_velocity(Ur, Uz)
    Ur, Uz, Vr_half, Vz_half = compute_next_displacement_field(Ur, Uz, Vr, Vz, Ar, Az)
    Vr, Vz, Ar, Az = compute_next_velocity_third_step(Vr_half, Vz_half, Ur, Uz, dt)

    Ur = bc_funcs.apply_bc_dirichlet_displacement(Ur, ghost_inds_left, interior_inds_left, U_target=0.0, axis=1)
    Ur = bc_funcs.apply_bc_dirichlet_displacement(Ur, ghost_inds_right, interior_inds_right, U_target=0.0, axis=1)
    Uz = bc_funcs.apply_bc_dirichlet_displacement(Uz, ghost_inds_bottom, interior_inds_bottom, U_target=0.0, axis=0)

    # Apply upward displacement at top boundary (e.g., 0.0001 m)
    Uz = bc_funcs.apply_bc_dirichlet_displacement(Uz, ghost_inds_top, interior_inds_top, U_target=0.0001, axis=0)

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")

time = end_time - start_time

# ------------------------
# Post-processing: visualization
# ------------------------
plot.temperature(Rmat, Zmat, T, total_time, nsteps, dr, dz, time)

# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""
