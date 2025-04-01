import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import time
import core_funcs as cf
import plot_utils as plot

# ------------------------
# Physical and simulation parameters
# ------------------------
k_mat = 50.0             # Thermal conductivity (W/m·K)
rho_mat = 7850.0         # Density (kg/m³)
Cp_mat = 420.0           # Specific heat capacity (J/kg·K)
Lr, Lz = 0.8, 0.8        # Domain size in r and z directions (meters)
Nr, Nz = 40, 40          # Number of cells in r and z directions
dr, dz = Lr / Nr, Lz / Nz    # Cell size in r and z
delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes = 3          # Number of ghost cells for boundary treatment

# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------
r_start = 0.2  # Start position in r-direction

# Physical domain (excluding ghost)
r_phys = np.linspace(r_start + dr / 2, r_start + Lr - dr / 2, Nr)
# Left and right ghost layers in r-direction
r_ghost_left = np.linspace(r_start - ghost_nodes * dr + dr / 2, r_start - dr / 2, ghost_nodes)
r_ghost_right = np.linspace(r_start + Lr + dr / 2, r_start + Lr + dr / 2 + (ghost_nodes - 1) * dr, ghost_nodes)

# Physical domain in z-direction (top to bottom)
z_phys = np.linspace(Lz - dz / 2, dz / 2, Nz)
# Top and bottom ghost layers
z_ghost_top = np.linspace(Lz + (ghost_nodes - 1) * dz + dz / 2, Lz + dz / 2, ghost_nodes)
z_ghost_bot = np.linspace(0 - dz / 2, -ghost_nodes * dz + dz / 2, ghost_nodes)

# Combine full coordinates
r_all = np.concatenate([r_ghost_left, r_phys, r_ghost_right])
z_all = np.concatenate([z_ghost_top, z_phys, z_ghost_bot])

Nr_tot = len(r_all)
Nz_tot = len(z_all)

# Generate 2D meshgrid
Rmat, Zmat = np.meshgrid(r_all, z_all, indexing='xy')

# ------------------------
# Compute distance matrix and horizon mask
# ------------------------
r_flat = Rmat.flatten()
z_flat = Zmat.flatten()

# Compute pairwise distances between all points
dx_r = r_flat[:, None] - r_flat[None, :]
dx_z = z_flat[:, None] - z_flat[None, :]
distance_matrix = np.sqrt(dx_r ** 2 + dx_z ** 2)

# Mask: which points are within the horizon (excluding self)
horizon_mask = (distance_matrix > 0) & (distance_matrix <= delta + 1e-6)
true_indices = np.where(horizon_mask)  # Optional: for indexing optimization

# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat, true_indices)
partial_area_matrix = cf.compute_partial_area_matrix(r_flat, z_flat, dr, dz, delta, distance_matrix)
threshold_distance = np.sqrt(2) * dr
factor_mat = np.where(distance_matrix <= threshold_distance, 1.125, 1.0)  # Local adjustment factor

# ------------------------
# Temperature update function
# ------------------------
def update_temperature(Tcurr, Hcurr, Kmat):
    flux = Kmat @ Tcurr.flatten() * dt               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_tot, Nr_tot)              # Reshape back to 2D
    Hnew = Hcurr + flux                              # Update enthalpy
    Tnew = cf.get_temperature(Hnew, rho_mat, Cp_mat) # Convert to temperature
    Tnew = cf.apply_mixed_bc(Tnew, z_all, Nr_tot, Nz_tot, ghost_nodes)  # Apply BC
    return Tnew, Hnew

# ------------------------
# Initialization
# ------------------------
T = np.full(Rmat.shape, 200)  # Initial temperature field (uniform 200 K)
T = cf.apply_mixed_bc(T, z_all, Nr_tot, Nz_tot, ghost_nodes)  # Apply boundary conditions
H = cf.get_enthalpy(T, rho_mat, Cp_mat)  # Initial enthalpy
Kmat = cf.build_K_matrix(T, cf.compute_thermal_conductivity_matrix, factor_mat,
                         partial_area_matrix, shape_factor_matrix,
                         distance_matrix, horizon_mask, true_indices, r_flat,
                         k_mat, delta)  # Initial stiffness matrix

# ------------------------
# Simulation loop settings
# ------------------------
dt = 10                     # Time step (s)
total_time = 5 * 3600     # Total simulation time (10 hours)
nsteps = int(total_time / dt)
print_interval = int(10 / dt)  # Print progress every 10 seconds of simulated time

print(f"Total steps: {nsteps}")
start_time = time.time()

# ------------------------
# Time-stepping loop
# ------------------------
for step in range(nsteps):
    T, H = update_temperature(T, H, Kmat)
    if step % print_interval == 0:
        print(f"Step={step}, Simulated time={step * dt:.2f}s")

end_time = time.time()
print(f"Calculation finished, elapsed real time={end_time - start_time:.2f}s")

# ------------------------
# Post-processing: visualization
# ------------------------
plot.temperature(Rmat, Zmat, T, total_time, nsteps)
