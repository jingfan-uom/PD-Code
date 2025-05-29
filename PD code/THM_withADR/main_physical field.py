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
# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s, cs, ks = 1800.0, 1688.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 372.65
Tl = 373.65
L = 333

Lr, Lz = 0.1, 0.1        # Domain size in r and z directions (meters)
Nr, Nz = 60, 60       # Number of cells in r and z directions
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

# 2D: use Euclidean distance
dx_r = r_flat[None, :] - r_flat[:, None]
dx_z = z_flat[None, :] - z_flat[:, None]
distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)

# Compute partial area overlap matrix
partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_flat, z_flat, dr, dz, delta, distance_matrix, tolerance)

horizon_mask = ((distance_matrix > tolerance) & (partial_area_matrix != 0.0))

row_sum = np.sum(partial_area_matrix, axis=1)  # (N,)
matrix_sum = row_sum[:, None] + row_sum[None, :]  # (N, N)
 # 结果是一维 (N,) 各行的总和
c = (6 * E) / (np.pi * delta**3 * h * (1 - 2 * nu) * (1 + nu))
c_matrix = 2 * np.pi * delta**2 / matrix_sum * c
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

dir_r, dir_z = pfc.compute_direction_matrix(r_flat, z_flat, Ur, Uz, horizon_mask)
def compute_accelerated_velocity(Ur_curr, Uz_curr,r_flat, z_flat, horizon_mask,dir_r ,dir_z ,c ,partial_area_matrix ,rho ,br, bz ):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr
    Relative_elongation = pfc.compute_s_matrix(r_flat, z_flat, Ur_new, Uz_new, horizon_mask,distance_matrix)
    Ar_new = dir_r * c_matrix * Relative_elongation * partial_area_matrix / rho
    Az_new = dir_z * c_matrix * Relative_elongation * partial_area_matrix / rho
    Ar_new = np.sum(Ar_new, axis=1) + br / rho # Shape matches Ur_curr
    Az_new = np.sum(Az_new, axis=1) + bz / rho

    return Ar_new, Az_new  # Or return other desired quantities



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
dt_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks, partial_area_matrix, horizon_mask,distance_matrix,delta)

Ar, Az = compute_accelerated_velocity(Ur, Uz,r_flat, z_flat, horizon_mask, dir_r ,dir_z , c_matrix , partial_area_matrix , rho_s,br, bz)
lambda_diag_matrix = ADR.compute_lambda_diag_matrix(partial_area_matrix, distance_matrix, c_matrix, horizon_mask,1 ,dx_r, dx_z)
Fr_0 = Ar * rho_s
Fz_0 = Az * rho_s
Vr_half = (1 / 2) * (Fr_0  / lambda_diag_matrix)
Vz_half = (1 / 2) * (Fz_0  / lambda_diag_matrix)
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
    Ar, Az = compute_accelerated_velocity(Ur, Uz, r_flat, z_flat, horizon_mask, dir_r, dir_z, c_matrix, partial_area_matrix,
                                              rho_s, br, bz)

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
    dir_r, dir_z = pfc.compute_direction_matrix(r_flat, z_flat, Ur, Uz, horizon_mask)

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
