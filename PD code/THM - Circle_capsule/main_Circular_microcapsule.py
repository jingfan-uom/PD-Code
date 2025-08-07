import sys
import os
sys.path.append(os.path.dirname(__file__))

import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator
import generate_coordinates as gc
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from multiprocessing import Pool, cpu_count
import time


rho_s, cs, ks = 1000.0, 2060.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 312.65
Tl = 313.65
L = 333
tolerance = 1e-15
Tsurr = 400
Tinit = 283.15
dt = 1e-8  # Time step in seconds
total_time = 100e-8  # Total simulation time (5 hours)
""" Initialization of coarse regions and temperatures """
r = 20 * 1e-6  # Domain size in r and z directions (meters)
dr = 0.5 * 1e-6
Nr = int(r / dr)
delta = 3 * dr
ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction
quarter_circle = True
ghost_left = True
start_time2 = time.time()
coords_phys, coords_ghost_left, coords_ghost_circle = gc.generate_half_circle_coordinates \
    (dr, r, Nr, ghost_nodes_r, ghost_left, quarter_circle)
coords = np.vstack([coords_phys, coords_ghost_left, coords_ghost_circle])
# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat = coords[:, 0]
z_flat = coords[:, 1]

dx_r = r_flat[None, :] - r_flat[:, None]  # shape (N, N)
dx_z = z_flat[None, :] - z_flat[:, None]
distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)  # shape (N, N)
partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_flat, z_flat, dr, dr, delta, distance_matrix, tolerance)
horizon_mask = ((distance_matrix > tolerance) & (partial_area_matrix != 0.0))
true_indices = np.where(horizon_mask)

# Apply initial boundary conditions
threshold_distance = np.sqrt(2) * dr
factor_mat = np.where(distance_matrix <= threshold_distance + tolerance, 1.125, 1.0)  # Local adjustment factor

T_phys = np.full((coords_phys.shape[0],), Tinit)  # Initial temperature field (uniform 200 K)
T_left = np.full((coords_ghost_left.shape[0],), Tinit)  # Initial temperature field (uniform 200 K)
T_circle = np.full((coords_ghost_circle.shape[0],), Tsurr)  # Initial temperature field (uniform 200 K)
ghost_indices_left, phys_indices_left = bc_funcs.find_mirror_pairs(coords_ghost_left, coords_phys, dr)
ghost_indices_circle, phys_indices_circle = bc_funcs.find_circle_mirror_pairs_multilayer(coords_ghost_circle,
                                                                                         coords_phys, dr, r)

T = np.concatenate([T_phys, T_left, T_circle])
print
n_phys = T_phys.shape[0]
n_left = T_left.shape[0]
n_circle = T_circle.shape[0]


# ------------------------
# Temperature update function
# ------------------------
def update_temperature(Tcurr, Hcurr, Kmat,
                       factor_mat,
                       partial_area_matrix, shape_factor_matrix,
                       distance_matrix, horizon_mask, true_indices,
                       delta, dt):
    flux = Kmat @ Tcurr  # Apply nonlocal heat flux
    Hnew = Hcurr + flux  # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_s, cs, cl, L, Ts, Tl)

    Knew = cf.build_K_matrix(Tnew, cf.compute_thermal_conductivity_matrix, factor_mat,
                             partial_area_matrix, shape_factor_matrix,
                             distance_matrix, horizon_mask, true_indices, ks, kl, Ts, Tl, delta, dt)  # k_mat 无需再传
    T_phys = Tnew[:n_phys]
    T_left = Tnew[n_phys: n_phys + n_left]
    T_circle = Tnew[n_phys + n_left:]
    T_left[ghost_indices_left] = T_phys[phys_indices_left]
    T_circle[ghost_indices_circle] = 2 * Tsurr - T_phys[phys_indices_circle]
    Tnew = np.concatenate([T_phys, T_left, T_circle])

    return Tnew, Hnew, Knew


H = cf.get_enthalpy(T, rho_l, cs, cl, L, Ts, Tl)  # Initial enthalpy
shape_factor_matrix = np.ones_like(horizon_mask, dtype=float)

# ------------------------
# Simulation loop settings
# ------------------------


# Build initial conductivity matrix
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                         partial_area_matrix, shape_factor_matrix,
                         distance_matrix, horizon_mask, true_indices, ks, kl, Ts, Tl, delta, dt)

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
    T, H, Kmat = update_temperature(
        T, H, Kmat,
        factor_mat,
        partial_area_matrix, shape_factor_matrix,
        distance_matrix, horizon_mask, true_indices,
        delta, dt, )

    if step in save_steps:
        T_record.append(T.copy())

    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * dt:.2f}s")

end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")
time1 = end_time - start_time
# ------------------------
# Post-processing: visualization
mask_circle = (coords[:, 0] ** 2 + (coords[:, 1] - r) ** 2 <= r ** 2 + tolerance) & (coords[:, 0] > 0) \
              & (coords[:, 1] > 0) & (coords[:, 1] <= 2 * r)
plot.temperature_contour(T, coords, mask_circle, total_time, nsteps, dr, r)
end_time2 = time.time()
print("Number of particles:", len(T))
print(f"Whole Calculation time = {end_time2 - start_time2:.2f}s")



