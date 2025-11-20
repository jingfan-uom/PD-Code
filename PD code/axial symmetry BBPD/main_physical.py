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
import matplotlib

# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s, cs, ks = 1800.0, 1688.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 372.65
Tl = 373.65
L = 333

Lr, Lz = 0.15, 0.1        # Domain size in r and z directions (meters)
Nr, Nz = 45, 30         # Number of cells in r and z directions
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
# ------------------------r
r_start = 0.5  # Starting position in r-direction0
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
z_all = np.concatenate([ z_phys,z_ghost_bot])
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
    dx_r = r_flat[None, :] - r_flat[:, None]
    distance_matrix = np.abs(dx_r)
else:
    # 2D: use Euclidean distance
    dx_r = r_flat[None, :] - r_flat[:, None]
    dx_z = z_flat[None, :] - z_flat[:, None]
    distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)

# Compute partial area overlap matrix
partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_flat, z_flat, dr, dz, delta, distance_matrix, tolerance
)
horizon_mask = (distance_matrix > tolerance) & (partial_area_matrix != 0.0)
row_sum_shell_m = np.sum(partial_area_matrix, axis=1)
matrix_sum_shell_m = row_sum_shell_m[:, None] + row_sum_shell_m[None, :]
c10 = (12 * E) / (np.pi * (delta ** 3) * (1.0 + nu))
c20 = (3 * E) / (np.pi * (delta ** 3) * (1.0 + nu))

c = 2 * np.pi * delta**2 / matrix_sum_shell_m * c10
c2 = 2 * np.pi * delta**2 / matrix_sum_shell_m * c20

# ------------------------
# Temperature update function
# ------------------------
true_indices = np.where(horizon_mask)
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat, true_indices)

# ------------------------
# Initialize displacement, velocity, and acceleration fields
# ------------------------
Ur = np.zeros_like(Rmat)  # Radial displacement
Uz = np.zeros_like(Rmat)  # Axial displacement

Vr = np.zeros_like(Rmat)  # Radial velocity
Vz = np.zeros_like(Rmat)  # Axial velocity

Ar = np.zeros_like(Rmat)  # Radial acceleration
Az = np.zeros_like(Rmat)  # Axial acceleration

dir_r, dir_z = pfc.compute_direction_matrix(Rmat, Zmat, Ur, Uz, horizon_mask)

# Get ghost node indices from bc_funcs
ghost_inds_top, interior_inds_top = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_bottom, interior_inds_bottom = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_left, interior_inds_left = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x)
ghost_inds_right, interior_inds_right = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x)




# core function of thermal and mechanical
def compute_accelerated_velocity(Ur_curr, Uz_curr,Rmat, Zmat,horizon_mask,dir_r ,dir_z ,c ,partial_area_matrix , rho,bz):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr
    Urr = Ur_curr.reshape(-1)
    Relative_elongation = pfc.compute_s_matrix(Rmat, Zmat, Ur_new, Uz_new, horizon_mask)

    Ar_new1 = dir_r * c * Relative_elongation * shape_factor_matrix * partial_area_matrix
    Ar_new2 = dir_r * c2 * shape_factor_matrix * partial_area_matrix

    Az_new1 = dir_z * c * Relative_elongation * shape_factor_matrix * partial_area_matrix
    Az_new2 = dir_z * c2 * shape_factor_matrix * partial_area_matrix

    br1 = -6 / np.pi / delta ** 3 * E / (1+nu) * distance_matrix * Relative_elongation * shape_factor_matrix * partial_area_matrix
    br1 = np.sum(br1, axis=1).reshape(Ur_curr.shape)/Rmat

    br2 = -((Urr/r_flat**2) * 3 * E / (1+nu)).reshape(Ur_curr.shape)
    br =br1 + br2
    Ar_new1 = np.sum(Ar_new1, axis=1).reshape(Ur_curr.shape)
    Ar_new2 = np.sum(Ar_new2, axis=1).reshape(-1) * (Urr/r_flat)
    Ar_new2 = Ar_new2.reshape(Ur_curr.shape)
    Ar_new = Ar_new1 + Ar_new2 + br

    Az_new1 = np.sum(Az_new1, axis=1).reshape(Ur_curr.shape)
    Az_new2 = np.sum(Az_new2, axis=1).reshape(-1) * (Urr / r_flat)
    Az_new2 = Az_new2.reshape(Ur_curr.shape)
    Az_new = Az_new1 + Az_new2 + bz
    # Set acceleration to zero at top ghost region
    return Ar_new/ rho, Az_new/ rho  # Or return other desired quantities

# Apply initial boundary conditions
bz = np.zeros_like(Rmat)

# Pressure value
pressure = 1000e3/dr
inds_top =[0]

bz[inds_top, :] = -pressure

dt_m = np.sqrt((2 * rho_s) / (np.pi * delta**2 * c10)) * 0.1  # Time step in seconds
dt_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks, partial_area_matrix, horizon_mask,distance_matrix,delta)

Ar, Az = compute_accelerated_velocity(Ur, Uz,Rmat, Zmat,horizon_mask,dir_r ,dir_z ,c ,partial_area_matrix , rho_s,bz)

Fr_0 = Ar * rho_s
Fz_0 = Az * rho_s
lambda_diag_matrix = ADR.compute_lambda_diag_matrix(partial_area_matrix, distance_matrix, c, horizon_mask)
Vr_half = ADR.adr_initial_velocity(Fr_0, dt_m, lambda_diag_matrix)
Vz_half = ADR.adr_initial_velocity(Fz_0, dt_m, lambda_diag_matrix)


# ------------------------
# Simulation loop settings
# ------------------------

total_time = 1000  # Total simulation time (e.g., 5 hours)
nsteps = int(1000)
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
    Ar, Az = compute_accelerated_velocity(Ur, Uz, Rmat, Zmat, horizon_mask, dir_r, dir_z, c, partial_area_matrix,
                                              rho_s, bz)

    Fr = Ar * rho_s
    Fz = Az * rho_s
    cr_n = ADR.compute_local_damping_coefficient(Fr, Fr_0, Vr_half, lambda_diag_matrix, Ur, 1)
    cz_n = ADR.compute_local_damping_coefficient(Fz, Fz_0, Vz_half, lambda_diag_matrix, Uz, 1)
    Fr_0 = Fr
    Fz_0 = Fz
    Vr_half, Ur = ADR.adr_update_velocity_displacement(Ur, Vr_half, Fr, cr_n, lambda_diag_matrix, 1)
    Vz_half, Uz = ADR.adr_update_velocity_displacement(Uz, Vz_half, Fz, cz_n, lambda_diag_matrix, 1)

    Uz[ghost_inds_bottom, :] = 0
    Ur[:, ghost_inds_left] = 0
    Ur[:, ghost_inds_right] = 0
    Ur[ghost_inds_bottom, :] = 0

    Uz[:, ghost_inds_right] = 0
    dir_r, dir_z = pfc.compute_direction_matrix(Rmat, Zmat, Ur, Uz, horizon_mask)

    #Ur, Uz, Vr_half, Vz_half = pfc.compute_next_displacement_field(Ur, Uz, Vr, Vz, Ar, Az,dt_m)
    # Vr, Vz, Ar, Az = pfc.compute_next_velocity_third_step(Vr_half, Vz_half, Ur, Uz, dt_m)

    # 计算当前位移增量的RMS
    delta_Ur = Ur - previous_Ur
    delta_Uz = Uz - previous_Uz

    rms_increment = np.sqrt(np.mean(delta_Ur ** 2 + delta_Uz ** 2))

    if rms_increment < 1e-10:
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
mask = np.ones(Ur.shape, dtype=bool)

# 将 ghost 点置为 False

mask[ghost_inds_bottom, :] = False
mask[:, ghost_inds_left] = False
mask[:, ghost_inds_right] = False
plot.plot_displacement_field(Rmat, Zmat, r_start,Ur, Uz,mask, title_prefix="Final Displacement", save=False)

# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""
