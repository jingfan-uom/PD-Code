import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator
import grid_generator_rectangle
# ------------------------
# Set material properties arrays (match shape of Rmat)
# ------------------------
rho_s, cs, ks = 1000.0, 2060.0, 2.14
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 272.65
Tl = 273.65
L = 333
ghost_nodes_r, ghost_nodes_z = 3, 3
tolerance= 1e-8

""" Initialization of coarse regions and temperatures """
Lr_coarse, Lz_coarse = 0.05, 0.1        # Domain size in r and z directions (meters)
# Coarse and fine grid spacing
dr_coarse, dz_coarse = 0.005, 0.005
Nz_coarse = int(Lz_coarse/dz_coarse)
delta_coarse = 3 * dr_coarse
r_start_coarse, z_start_coarse = 0.0, 0.0  # Starting positions in r and z

Lr_fine, Lz_fine = 0.05, 0.1       # Domain size in r and z directions (meters)
# Fine grid spacing
dr_fine, dz_fine = 0.0025, 0.0025
Nz_fine = int(Lz_fine/dz_fine)
delta_fine = 3 * dr_fine
ghost_nodes_r, ghost_nodes_z = 3, 3
r_start_fine, z_start_fine = 0.05, 0.0  # Starting positions in r and z

r_phys_coarse = np.arange(r_start_coarse + dr_coarse / 2, r_start_coarse + Lr_coarse, dr_coarse)
z_phys_coarse = np.linspace(Lz_coarse + z_start_coarse- dz_coarse/2, z_start_coarse+dz_coarse/2, Nz_coarse)
r_ghost_left_coarse = np.linspace(r_start_coarse - ghost_nodes_r * dr_coarse + dr_coarse/2, r_start_coarse - dr_coarse/2, ghost_nodes_r)
r_ghost_right_coarse = np.linspace(r_start_coarse + Lr_coarse + dr_coarse/2,
                                   r_start_coarse + Lr_coarse + dr_coarse/2 + (ghost_nodes_r - 1) * dr_coarse,
                                   ghost_nodes_r)
z_ghost_top_coarse = np.linspace(z_start_coarse + Lz_coarse + (ghost_nodes_z - 1) * dz_coarse + dz_coarse/2,
                                 z_start_coarse + Lz_coarse + dz_coarse/2,
                                   ghost_nodes_z)
z_ghost_bot_coarse = np.linspace(z_start_coarse - dz_coarse/2, z_start_coarse - ghost_nodes_z * dz_coarse + dz_coarse/2, ghost_nodes_z)
r_all_coarse = np.concatenate([r_ghost_left_coarse, r_phys_coarse, r_ghost_right_coarse])
z_all_coarse = np.concatenate([z_ghost_top_coarse, z_phys_coarse, z_ghost_bot_coarse])
Nr_all_coarse = len(r_all_coarse)
Nz_all_coarse = len(z_all_coarse)

ghost_inds_top_coarse, interior_inds_top_coarse = bc_funcs.get_top_ghost_indices(z_all_coarse, ghost_nodes_z)
ghost_inds_bottom_coarse, interior_inds_bottom_coarse = bc_funcs.get_bottom_ghost_indices(z_all_coarse, ghost_nodes_z)
ghost_inds_left_coarse, interior_inds_left_coarse = bc_funcs.get_left_ghost_indices(r_all_coarse, ghost_nodes_r)
ghost_inds_right_coarse, interior_inds_right_coarse = bc_funcs.get_right_ghost_indices(r_all_coarse, ghost_nodes_r)

Rmat_coarse, Zmat_coarse = np.meshgrid(r_all_coarse, z_all_coarse, indexing='xy')

# ==========================
""" Initialization of fine regions and temperatures """
# ==========================

# Define fine region (set to None or empty if no fine region)

r_phys_fine = np.arange(r_start_fine + dr_fine / 2, r_start_fine + Lr_fine, dr_fine)
z_phys_fine = np.linspace(Lz_fine - dz_fine/2 +z_start_fine, dz_fine/2 + z_start_fine, Nz_fine)

# Initialization of regions and temperatures

r_ghost_left_fine = np.linspace(r_start_fine - ghost_nodes_r * dr_fine + dr_fine/2, r_start_fine - dr_fine/2, ghost_nodes_r)
r_ghost_right_fine = np.linspace(r_start_fine + Lr_fine + dr_fine/2,
                                 r_start_fine + Lr_fine + dr_fine/2 + (ghost_nodes_r - 1) * dr_fine,
                                 ghost_nodes_r)
z_ghost_top_fine = np.linspace(z_start_fine + Lz_fine + (ghost_nodes_z - 1) * dz_fine + dz_fine/2,
                               z_start_fine + Lz_fine + dz_fine/2,
                               ghost_nodes_z)
z_ghost_bot_fine = np.linspace(z_start_fine - dz_fine/2, z_start_fine - ghost_nodes_z * dz_fine + dz_fine/2, ghost_nodes_z)

r_all_fine = np.concatenate([r_ghost_left_fine, r_phys_fine, r_ghost_right_fine])
z_all_fine = np.concatenate([z_ghost_top_fine, z_phys_fine, z_ghost_bot_fine])
Nr_all_fine = len(r_all_fine)
Nz_all_fine = len(z_all_fine)

ghost_inds_top_fine, interior_inds_top_fine = bc_funcs.get_top_ghost_indices(z_all_fine, ghost_nodes_z)
ghost_inds_bottom_fine, interior_inds_bottom_fine = bc_funcs.get_bottom_ghost_indices(z_all_fine, ghost_nodes_z)
ghost_inds_left_fine, interior_inds_left_fine = bc_funcs.get_left_ghost_indices(r_all_fine, ghost_nodes_r)
ghost_inds_right_fine, interior_inds_right_fine = bc_funcs.get_right_ghost_indices(r_all_fine, ghost_nodes_r)

Rmat_fine, Zmat_fine = np.meshgrid(r_all_fine, z_all_fine, indexing='xy')


# Set the region with r < 0.04 and z < 0.04 as the ice region (low temperature)
ice_region_mask = (Rmat_coarse < 0.04 + tolerance) & (Zmat_coarse < 0.04 + tolerance)
T_coarse = np.full(Rmat_coarse.shape, 373.15)  # Initial temperature field (uniform 373.15 K)
T_coarse[ice_region_mask] = 268.15

T_fine = np.full(Rmat_fine.shape, 373.15)  # Initial temperature field (uniform 373.15 K)

# === 2) Compute the temperature at "coarse grid interface points" ==========================
# This step uses the previously implemented 4-point average interpolation function:
# get_neighbor_points_array_from_matrices(...)


coord_index_coarse = grid_generator_rectangle.get_coarse_neighbor_points(
    Rmat_coarse,
    Zmat_coarse,
    ghost_inds_right_coarse,
    dr_fine,
    dz_fine,
    r_all_fine,
    z_all_fine,
    axis=1,
)

T_coarse = grid_generator_rectangle.interpolate_temperature_for_coarse(
    T_coarse,
    T_fine,
    coord_index_coarse,
    ghost_inds_right_coarse,
    axis=1,
)

coord_index_left_fine = grid_generator_rectangle.get_fine_neighbor_points(
    Rmat_fine,
    Zmat_fine,
    ghost_inds_left_fine,
    dr_fine,
    dz_fine,
    Rmat_coarse,
    Zmat_coarse,
    r_all_coarse,
    z_all_coarse,
    axis=1)

# === Step 2: Interpolate to obtain the temperature in the fine grid ghost region ===

T_fine = grid_generator_rectangle.interpolate_temperature_for_fine(
    T_fine,
    T_coarse,
    coord_index_left_fine,
    ghost_inds_left_fine,
    axis=1)

# Apply initial boundary conditions
T_coarse = bc_funcs.apply_bc_zero_flux(T_coarse, ghost_inds_top_coarse, interior_inds_top_coarse, axis=0)
T_coarse = bc_funcs.apply_bc_zero_flux(T_coarse, ghost_inds_bottom_coarse, interior_inds_bottom_coarse, axis=0)
T_coarse = bc_funcs.apply_bc_zero_flux(T_coarse, ghost_inds_left_coarse, interior_inds_left_coarse, axis=1)

# Apply initial boundary conditions
T_fine = bc_funcs.apply_bc_zero_flux(T_fine, ghost_inds_top_fine, interior_inds_top_fine, axis=0)
T_fine = bc_funcs.apply_bc_zero_flux(T_fine, ghost_inds_bottom_fine, interior_inds_bottom_fine, axis=0)
T_fine = bc_funcs.apply_bc_zero_flux(T_fine, ghost_inds_right_fine, interior_inds_right_fine, axis=1)

# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat_fine = Rmat_fine.flatten()
z_flat_fine = Zmat_fine.flatten()
r_flat_coarse = Rmat_coarse.flatten()
z_flat_coarse = Zmat_coarse.flatten()

# Construct the distance matrix (Euclidean distance)

dx_r_fine = r_flat_fine[:, None] - r_flat_fine[None, :]
dx_z_fine = z_flat_fine[:, None] - z_flat_fine[None, :]
dx_r_coarse= r_flat_coarse[:, None] - r_flat_coarse[None, :]
dx_z_coarse= z_flat_coarse[:, None] - z_flat_coarse[None, :]

distance_matrix_fine = np.sqrt(dx_r_fine**2 + dx_z_fine**2)  # shape = (N1 + N2, N1 + N2)
distance_matrix_coarse = np.sqrt(dx_r_coarse**2 + dx_z_coarse**2)  # shape = (N1 + N2, N1 + N2)


# Compute partial area overlap matrix
partial_area_matrix_fine = area_matrix_calculator.compute_partial_area_matrix(
    r_flat_fine, z_flat_fine, dr_fine, dz_fine, delta_fine, distance_matrix_fine, tolerance
)
partial_area_matrix_coarse = area_matrix_calculator.compute_partial_area_matrix(
    r_flat_coarse, z_flat_coarse, dr_coarse, dz_coarse, delta_coarse, distance_matrix_coarse, tolerance
)

horizon_mask_fine = (distance_matrix_fine > tolerance) & (partial_area_matrix_fine != 0.0)
horizon_mask_coarse = (distance_matrix_coarse > tolerance) & (partial_area_matrix_coarse != 0.0)

true_indices_fine = np.where(horizon_mask_fine)
true_indices_coarse = np.where(horizon_mask_coarse)

# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
shape_factor_matrix_fine = cf.compute_shape_factor_matrix(Rmat_fine, true_indices_fine)
shape_factor_matrix_coarse = cf.compute_shape_factor_matrix(Rmat_coarse, true_indices_coarse)

threshold_distance_fine = np.sqrt(2) * dr_fine
threshold_distance_coarse  = np.sqrt(2) * dr_coarse

factor_mat_fine = np.where(distance_matrix_fine <= threshold_distance_fine + tolerance, 1.125, 1.0)  # Local adjustment factor
factor_mat_coarse = np.where(distance_matrix_coarse <= threshold_distance_coarse + tolerance, 1.125, 1.0)  # Local adjustment factor
# ------------------------
# Temperature update function
# ------------------------
def update_temperature(T1,Tcurr, Hcurr, Kmat,Nz_all, Nr_all,
                       ghost_inds_top, interior_inds_top,
                       ghost_inds_bottom, interior_inds_bottom,
                       ghost_inds_left, interior_inds_left,
                       ghost_inds_right, interior_inds_right,
                       factor_mat,
                       partial_area_matrix, shape_factor_matrix,
                       distance_matrix, horizon_mask, true_indices,
                       delta, r_flat,dt, coarse=1
                       ):

    Knew = Kmat
    flux = Knew @ Tcurr.flatten()               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_all, Nr_all)         # Reshape to 2D
    flux[(flux > -tolerance) & (flux < tolerance)] = 0  # Eliminate small fluctuations
    Hnew = Hcurr + flux                          # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_s , cs , cl, L, Ts, Tl)
  # Convert to temperature

    # Apply boundary conditions

    if coarse == 1:
        Tnew = grid_generator_rectangle.interpolate_temperature_for_coarse(
          Tnew,
          T_fine,
          coord_index_coarse,
          ghost_inds_right_coarse,
          axis=1)
        Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_left, interior_inds_left, axis=1)

    else:

        Tnew = grid_generator_rectangle.interpolate_temperature_for_fine(
            Tnew,
            T_coarse,
            coord_index_left_fine,
            ghost_inds_left_fine,
            axis=1,
        )
        Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_right, interior_inds_right, axis=1)

    Knew = cf.build_K_matrix(Tnew, cf.compute_thermal_conductivity_matrix, factor_mat,
                             partial_area_matrix, shape_factor_matrix,
                             distance_matrix, horizon_mask, true_indices, ks, kl, Ts, Tl, delta, r_flat,
                             dt)  
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_top, interior_inds_top, axis=0)
    Tnew = bc_funcs.apply_bc_zero_flux(Tnew, ghost_inds_bottom, interior_inds_bottom, axis=0)

    return Tnew, Hnew ,Knew



H_fine = cf.get_enthalpy(T_fine, rho_l, cs, cl, L, Ts, Tl) # Initial enthalpy
H_coarse = cf.get_enthalpy(T_coarse, rho_l, cs, cl, L, Ts, Tl) # Initial enthalpy
# ------------------------
# Simulation loop settings
# ------------------------
dt = 5  # Time step in seconds

# Build initial conductivity matrix

Kmat_coarse = cf.build_K_matrix(T_coarse, cf.compute_thermal_conductivity_matrix, factor_mat_coarse,
                             partial_area_matrix_coarse, shape_factor_matrix_coarse,
                             distance_matrix_coarse, horizon_mask_coarse, true_indices_coarse, ks, kl, Ts, Tl, delta_coarse, r_flat_coarse,
                             dt)
Kmat_fine = cf.build_K_matrix(T_fine, cf.compute_thermal_conductivity_matrix, factor_mat_fine,
                             partial_area_matrix_fine, shape_factor_matrix_fine,
                             distance_matrix_fine, horizon_mask_fine, true_indices_fine, ks, kl, Ts, Tl, delta_fine, r_flat_fine,
                             dt)   # dz is used to determine whether a one-dimensional



total_time =  600  # Total simulation time (5 hours)
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
    T_coarse, H_coarse, Kmat_coarse = update_temperature(T_fine,
        T_coarse, H_coarse, Kmat_coarse, Nz_all_coarse, Nr_all_coarse,
        ghost_inds_top_coarse, interior_inds_top_coarse,
        ghost_inds_bottom_coarse, interior_inds_bottom_coarse,
        ghost_inds_left_coarse, interior_inds_left_coarse,
        ghost_inds_right_coarse, interior_inds_right_coarse,
        factor_mat_coarse,
        partial_area_matrix_coarse, shape_factor_matrix_coarse,
        distance_matrix_coarse, horizon_mask_coarse, true_indices_coarse,
        delta_coarse, r_flat_coarse, dt, coarse=1)
    T_fine, H_fine, Kmat_fine = update_temperature(T_coarse,T_fine, H_fine, Kmat_fine, Nz_all_fine, Nr_all_fine,
                                                   ghost_inds_top_fine, interior_inds_top_fine,
                                                   ghost_inds_bottom_fine, interior_inds_bottom_fine,
                                                   ghost_inds_left_fine, interior_inds_left_fine,
                                                   ghost_inds_right_fine, interior_inds_right_fine,
                                                   factor_mat_fine,
                                                   partial_area_matrix_fine, shape_factor_matrix_fine,
                                                   distance_matrix_fine, horizon_mask_fine, true_indices_fine,
                                                   delta_fine, r_flat_fine, dt,  coarse=0)
    if step == 100/dt:
        aa=1
    if step in save_steps:
        T_record.append(T_fine.copy())

    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * dt:.2f}s")

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")
time1 =end_time - start_time
# ------------------------
# Post-processing: visualization
# plot1.temperature(Rmat_coarse, Zmat_coarse, T_coarse, total_time, nsteps, dt)
levels= np.arange(270, 380, 5)
plot.temperature_combined(
    Rmat_coarse, Zmat_coarse, T_coarse,
    Rmat_fine, Zmat_fine, T_fine,
    total_time,time1, nsteps,
    levels,
    r_start_coarse,Lr_coarse,dr_coarse,dz_coarse,ghost_nodes_r,
    r_start_fine,Lr_fine,dr_fine,dz_fine,
    ghost_nodes_r)




# Optional: 1D profile plots
"""
for i, T_snap in enumerate(T_record):
    sim_time = save_times[i] * 3600
    plot.plot_1d_temperature(r_all, T_snap, sim_time)
"""
