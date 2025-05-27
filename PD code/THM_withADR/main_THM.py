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
rho_l, cl, kl = 6000.0, 6182.0, 0.6
Ts = 372.65
Tl = 373.65
L = 333

Lr, Lz = 0.01, 0.01        # Domain size in r and z directions (meters)
Nr, Nz = 10, 10      # Number of cells in r and z directions
dr, dz = Lr / Nr, Lz / Nz  # Cell size in r and z directions

E = 25.7e9               # Elastic modulus [Pa]
nu = 0.25                # Poisson's ratio
alpha = 1.8e-5           # Thermal expansion coefficient [1/K]

delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes_x = 3        # Number of ghost cells in the x (or r) direction
ghost_nodes_z = 3        # Number of ghost cells in the z direction
h = 1
# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.0
z_start = 0.0# Starting position in r-direction
tolerance = 1e-8

# ==========================
#  Mechanical field coordinates
# ==========================
r_all_m, z_all_m, Nr_tot_m, Nz_tot_m = gc.generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz, ghost_nodes_x, ghost_nodes_z,
    r_ghost_left=True, r_ghost_right=True,
    z_ghost_top=True, z_ghost_bot=True)
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
c = (6 * E) / (np.pi * delta**3 * h * (1 - 2 * nu) * (1 + nu))
c_matrix = 2 * np.pi * delta**2 / matrix_sum * c

# Compute partial area overlap matrix
r_all_th, z_all_th, Nr_tot_th, Nz_tot_th = gc.generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz, ghost_nodes_x, ghost_nodes_z,
    r_ghost_left=True, r_ghost_right=True,
    z_ghost_top=True, z_ghost_bot=True)
Nr_tot_th = len(r_all_th)
Nz_tot_th = len(z_all_th)
Rmat_th, Zmat_th = np.meshgrid(r_all_th, z_all_th, indexing='xy')
# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat_th = Rmat_th.flatten()
z_flat_th = Zmat_th.flatten()
# 2D: use Euclidean distance
dx_r_th = r_flat_th[None, :] - r_flat_th[:, None]
dx_z_th = z_flat_th[None, :] - z_flat_th[:, None]
distance_matrix_th = np.sqrt(dx_r_th ** 2 + dx_z_th ** 2)
# Compute partial area overlap matrix
partial_area_matrix_th = area_matrix_calculator.compute_partial_area_matrix(
    r_flat_th, z_flat_th, dr, dz, delta, distance_matrix_th, tolerance)
horizon_mask_th = ((distance_matrix_th > tolerance) & (partial_area_matrix_th != 0.0))
true_indices_th = np.where(horizon_mask_th)

mask_corner = np.ones((Nz_tot_th, Nr_tot_th), dtype=bool)
mask_corner [0:ghost_nodes_z, 0:ghost_nodes_x] = False        # 左上
mask_corner [0:ghost_nodes_z, -ghost_nodes_x:] = False        # 右上
mask_corner [-ghost_nodes_z:, 0:ghost_nodes_x] = False        # 左下
mask_corner [-ghost_nodes_z:, -ghost_nodes_x:] = False
"""# 右下
false_indices_2d = np.where(mask_corner == False)
false_indices_flat = np.ravel_multi_index(false_indices_2d, mask_corner.shape)

for idx in false_indices_flat:
    horizon_mask[idx, :] = False
    horizon_mask[:, idx] = False
"""
# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat_th, true_indices_th)
threshold_distance = np.sqrt(2) * dr
factor_mat = np.where(distance_matrix_th <= threshold_distance + tolerance, 1.125, 1.0)  # Local adjustment factor

# ------------------------
# Initialize displacement, velocity, and acceleration fields, T
# ------------------------
Ur = np.zeros_like(Rmat_m).flatten()  # Radial displacement (1D)
Uz = np.zeros_like(Rmat_m).flatten()  # Axial displacement (1D)
Vr = np.zeros_like(Rmat_m).flatten()  # Radial velocity (1D)
Vz = np.zeros_like(Rmat_m).flatten()  # Axial velocity (1D)
Ar = np.zeros_like(Rmat_m).flatten()  # Radial acceleration (1D)
Az = np.zeros_like(Rmat_m).flatten()  # Axial acceleration (1D)
dir_r, dir_z = pfc.compute_direction_matrix(r_flat_m, z_flat_m, Ur, Uz, horizon_mask_m)

T = np.full(Rmat_th.shape, 283.15)  # Initial temperature field (uniform 200 K)
Tpre_avg = 283.15

# Get ghost node indices from bc_funcs
ghost_inds_left_m, interior_inds_left_m, ghost_inds_left_1d_m = bc_funcs.get_left_ghost_indices(r_all_m, ghost_nodes_x, Nz_tot_m)
ghost_inds_right_m, interior_inds_right_m, ghost_inds_right_1d_m = bc_funcs.get_right_ghost_indices(r_all_m, ghost_nodes_x, Nz_tot_m)
ghost_inds_top_m, interior_inds_top_m, ghost_inds_top_1d_m = bc_funcs.get_top_ghost_indices(z_all_m, ghost_nodes_z, Nr_tot_m)
ghost_inds_bottom_m, interior_inds_bottom_m, ghost_inds_bottom_1d_m = bc_funcs.get_bottom_ghost_indices(z_all_m, ghost_nodes_z, Nr_tot_m)

ghost_inds_left_th, interior_inds_left_th, ghost_inds_left_1d_th = bc_funcs.get_left_ghost_indices(r_all_th, ghost_nodes_x, Nz_tot_th)
ghost_inds_right_th, interior_inds_right_th, ghost_inds_right_1d_th = bc_funcs.get_right_ghost_indices(r_all_th, ghost_nodes_x, Nz_tot_th)
ghost_inds_top_th, interior_inds_top_th, ghost_inds_top_1d_th = bc_funcs.get_top_ghost_indices(z_all_th, ghost_nodes_z, Nr_tot_th)
ghost_inds_bottom_th, interior_inds_bottom_th, ghost_inds_bottom_1d_th = bc_funcs.get_bottom_ghost_indices(z_all_th, ghost_nodes_z, Nr_tot_th)

# Apply initial conditions
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_bottom_th, interior_inds_bottom_th, axis=0)
T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_left_th, interior_inds_left_th, axis=1)
T = bc_funcs.apply_bc_dirichlet_mirror(T, ghost_inds_right_th, interior_inds_right_th, 274.15, axis=1, z_mask=None, r_mask=None)
T = bc_funcs.apply_bc_dirichlet_mirror(T, ghost_inds_top_th, interior_inds_top_th, 274.15, axis=0, z_mask=None, r_mask=None)
T [~mask_corner] =265.15



# core function of thermal and mechanical
def compute_accelerated_velocity(Ur_curr, Uz_curr,r_flat, z_flat, horizon_mask_m,dir_r ,dir_z ,c_matrix ,partial_area_matrix_m ,rho , T_curr,Tpre_avg):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr
    T_curr = T_curr.flatten()
    T_m = pfc.filter_array_by_indices_keep_only(T_curr, CorrList_T)
    Tavg = pfc.compute_delta_temperature(T_m, Tpre_avg)
    Relative_elongation = pfc.compute_s_matrix(r_flat, z_flat, Ur_new, Uz_new, horizon_mask_m, distance_matrix_m)

    Ar_new = dir_r * c_matrix * (Relative_elongation - alpha * Tavg) * partial_area_matrix_m / rho
    Az_new = dir_z * c_matrix * (Relative_elongation - alpha * Tavg) * partial_area_matrix_m / rho

    Ar_new = np.sum(Ar_new, axis=1)  # Shape matches Ur_curr
    Az_new = np.sum(Az_new, axis=1)
    # Set acceleration to zero at top ghost region

    return Ar_new, Az_new  # Or return other desired quantities

# Temperature update function

def update_temperature(Tcurr, Hcurr, Kcurr):
    flux = Kcurr @ Tcurr.flatten()               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_tot_th, Nr_tot_th)
    Hnew = Hcurr + flux                          # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_s, cs, cl, L, Ts, Tl)   # Convert to temperature

    #Knew = cf.build_K_matrix(Tnew , cf.compute_thermal_conductivity_matrix, factor_mat,
    #                        partial_area_matrix, shape_factor_matrix,
    #                        distance_matrix, horizon_mask, true_indices, r_flat,
    #                         ks, kl, Ts, Tl, delta, dz, dt_th)
    Knew = Kcurr
    return Tnew, Hnew, Knew

# time_step calculation
dt_m = np.sqrt((2 * rho_s) / (np.pi * delta**2 * c)) * 0.05  # Time step in seconds
dt_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks, partial_area_matrix_th, horizon_mask_th,distance_matrix_th, delta) * 0.5
dt = 1

H = cf.get_enthalpy(T, rho_s, cs, cl, L, Ts, Tl)  # Initial enthalpy
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                         partial_area_matrix_th, shape_factor_matrix,
                         distance_matrix_th, horizon_mask_th, true_indices_th, r_flat_th,
                         ks, kl, Ts, Tl, delta, dz, dt_th)

CorrList_T = pfc.shrink_Tth_by_matching_coords(Rmat_m, Zmat_m, Rmat_th, Zmat_th)
lambda_diag_matrix = ADR.compute_lambda_diag_matrix(partial_area_matrix_m, distance_matrix_m, c_matrix, horizon_mask_m,dt ,dx_r_m,dx_z_m)

Ar, Az = compute_accelerated_velocity(Ur, Uz, r_flat_m, z_flat_m, horizon_mask_m, dir_r, dir_z, c_matrix, partial_area_matrix_m,
                                              rho_s, T, Tpre_avg)
Fr_0 = Ar * rho_s
Fz_0 = Az * rho_s
Vr_half = (dt / 2) * (Fr_0  / lambda_diag_matrix)
Vz_half = (dt / 2) * (Fz_0  / lambda_diag_matrix)
Ur = Vr_half * dt + Ur
Uz = Vz_half * dt + Uz
Uz[ghost_inds_bottom_1d_m] = 0
Ur[ghost_inds_left_1d_m] = 0


# ------------------------
# Simulation loop settings
# ------------------------
total_time = 200  # Total simulation time (e.g., 5 hours)
nsteps_th = int(total_time/dt_th )
nsteps_m = int(10000)
print_interval = int(total_time / dt_m)  # Print progress every 10 simulated seconds
start_time = time.time()

# ------------------------
# Time-stepping loop
# ------------------------
for step in range(nsteps_th):

    T, H, Kmat = update_temperature(T, H, Kmat)
    T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_bottom_th, interior_inds_bottom_th, axis=0)
    T = bc_funcs.apply_bc_zero_flux(T, ghost_inds_left_th, interior_inds_left_th, axis=1)
    T = bc_funcs.apply_bc_dirichlet_mirror(T, ghost_inds_right_th, interior_inds_right_th, 274.15, axis=1, z_mask=None, r_mask=None)
    T = bc_funcs.apply_bc_dirichlet_mirror(T, ghost_inds_top_th, interior_inds_top_th, 274.15, axis=0, z_mask=None, r_mask=None)

    if step % 10 == 0:
        print(f"[Temperature Step {step + 1}/{nsteps_th}]")
    for step in range(nsteps_m):
        previous_Ur = Ur.copy()
        previous_Uz = Uz.copy()

        Ar, Az = compute_accelerated_velocity(Ur, Uz, r_flat_m, z_flat_m, horizon_mask_m, dir_r, dir_z, c_matrix, partial_area_matrix_m,
                                              rho_s, T, Tpre_avg)
        Fr = Ar * rho_s
        Fz = Az * rho_s
        cr_n = ADR.compute_local_damping_coefficient(Fr, Fr_0, Vr_half, lambda_diag_matrix, Ur, dt)
        cz_n = ADR.compute_local_damping_coefficient(Fz, Fz_0, Vz_half, lambda_diag_matrix, Uz, dt)
        Fr_0 = Fr.copy()
        Fz_0 = Fz.copy()
        Vr_half, Ur = ADR.adr_update_velocity_displacement(Ur, Vr_half, Fr, cr_n, lambda_diag_matrix, dt)
        Vz_half, Uz = ADR.adr_update_velocity_displacement(Uz, Vz_half, Fz, cz_n, lambda_diag_matrix, dt)
        Uz[ghost_inds_bottom_1d_m] = 0
        Ur[ghost_inds_left_1d_m] = 0


        dir_r, dir_z = pfc.compute_direction_matrix(r_flat_m, z_flat_m, Ur, Uz, horizon_mask_m)

        # 计算当前位移增量的RMS
        delta_Ur = Ur - previous_Ur
        delta_Uz = Uz - previous_Uz

        rms_increment = np.sqrt(np.mean(delta_Ur ** 2 + delta_Uz ** 2))

        if rms_increment < 1e-11:
            print(f"Convergence reached at step {step} with RMS displacement increment {rms_increment}")

            break
        if step % 10 == 0:
            print(f"Step_m {step} completed")


    #Ur, Uz, Vr_half, Vz_half = pfc.compute_next_displacement_field(Ur, Uz, Vr, Vz, Ar, Az,dt_m)
    # Vr, Vz, Ar, Az = pfc.compute_next_velocity_third_step(Vr_half, Vz_half, Ur, Uz, dt_m)
    #Uz[ghost_inds_bottom, :] = 0
    #Ur[:, ghost_inds_left] = 0
    # Ur[:, ghost_inds_right] = 0

    if step % 10 == 0:
        print(f"Step {step}/{nsteps_m} completed")

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")
time = end_time - start_time

# ------------------------
# Post-processing: visualization
# ------------------------
#mask_m and mask_th Used to exclude displacement and temperature of boundary particles

mask_m = np.ones(Rmat_m.shape, dtype=bool)
mask_m[ghost_inds_top_m, :] = False
mask_m[ghost_inds_bottom_m, :] = False
mask_m[:, ghost_inds_left_m] = False
mask_m[:, ghost_inds_right_m] = False
Ur = Ur.reshape(Rmat_m.shape)
Uz = Uz.reshape(Zmat_m.shape)
plot.plot_displacement_field(Rmat_m, Zmat_m, Ur, Uz, mask_m, Lr, Lz, title_prefix="Final Displacement", save=False)

mask_th = np.ones(T.shape, dtype=bool)
mask_th[ghost_inds_top_m, :] = False
mask_th[ghost_inds_bottom_m, :] = False
mask_th[:, ghost_inds_left_m] = False
mask_th[:, ghost_inds_right_m] = False
plot.temperature(Rmat_th, Zmat_th, T, total_time, nsteps_th, dr, dz, time, mask_th, Lr,Lz)
# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""


