
import numpy as np
import time
import core_funcs as cf
import plot_utils as plot
import bc_funcs
import area_matrix_calculator
import Physical_Field_Calculation as pfc  # Import module containing three field functions
import ADR
# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s, cs, ks = 1800.0, 1688.0, 1
rho_l, cl, kl = 1800.0, 4182.0, 0.6
Ts = 372.65
Tl = 373.65
L = 333

Lr, Lz = 0.1, 0.1        # Domain size in r and z directions (meters)
Nr, Nz = 40, 40         # Number of cells in r and z directions
dr, dz = Lr / Nr, Lz / Nz  # Cell size in r and z directions

E = 1e9                  # Elastic modulus [Pa]
nu = 0.33                # Poisson's ratio
K = 1.0                  # Thermal conductivity [W/(m·K)]
alpha = 1.8e-5           # Thermal expansion coefficient [1/K]

delta = 3 * dr           # Horizon radius for nonlocal interaction
ghost_nodes_x = 3        # Number of ghost cells in the x (or r) direction
ghost_nodes_z = 3        # Number of ghost cells in the z direction
h = 1

vmod = E / 3 / (1.0 - nu**2)
smod = E / (2.0 * (1.0 + nu))
# 体积能量系数 a1 以及热膨胀耦合系数 a2, a3
a1 = vmod + smod * (nu + 1)**2 / (9 * (2*nu - 1)**2)
a2 = 4.0 * alpha * a1
a3 = alpha * a2
b = 6.0 * smod / (np.pi * h * delta**4)
d = 2.0 / (np.pi * h * delta**3)

# ------------------------
# Construct grid coordinates (including ghost layers)
# ------------------------r
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
r_all = np.concatenate([r_ghost_left, r_phys,r_ghost_right])
Nr_tot = len(r_all)

# ==========================
#   z direction
# ==========================
z_phys = np.linspace(Lz - dz / 2, dz / 2, Nz)
z_ghost_top = np.linspace(Lz + (ghost_nodes_z - 1) * dz + dz / 2, Lz + dz / 2, ghost_nodes_z)
z_ghost_bot = np.linspace(0 - dz / 2, -ghost_nodes_z * dz + dz / 2, ghost_nodes_z)
z_all = np.concatenate([z_phys,z_ghost_bot])
Nz_tot = len(z_all)

# Meshgrid
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
sij = np.zeros_like(distance_matrix)

# Compute partial area overlap matrix
partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
    r_flat, z_flat, dr, dz, delta, distance_matrix, tolerance
)
horizon_mask = (distance_matrix > tolerance) & (partial_area_matrix != 0.0)
row_sum = np.sum(partial_area_matrix, axis=1)
matrix_sum = row_sum[:, None] + row_sum[None, :]
scr = 2 * np.pi * delta**2 / matrix_sum
scr = np.maximum(1.0, scr)

# ------------------------
# Temperature update function
# ------------------------
true_indices = np.where(horizon_mask)
shape_factor_matrix = cf.compute_shape_factor_matrix(Rmat, true_indices)

# ------------------------
# Initialize displacement, velocity, and acceleration fields
# ------------------------
Ur = np.zeros_like(r_flat)  # Radial displacement
Uz = np.zeros_like(r_flat)  # Axial displacement

Vr = np.zeros_like(r_flat) # Radial velocity
Vz = np.zeros_like(r_flat) # Axial velocity

Ar = np.zeros_like(r_flat) # Radial acceleration
Az = np.zeros_like(r_flat)  # Axial acceleration

n_r = np.zeros_like(distance_matrix, dtype=float)
n_z = np.zeros_like(distance_matrix, dtype=float)

# Get ghost node indices from bc_funcs
ghost_inds_top, interior_inds_top = bc_funcs.get_top_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_bottom, interior_inds_bottom = bc_funcs.get_bottom_ghost_indices(z_all, ghost_nodes_z)
ghost_inds_left, interior_inds_left = bc_funcs.get_left_ghost_indices(r_all, ghost_nodes_x)
ghost_inds_right, interior_inds_right = bc_funcs.get_right_ghost_indices(r_all, ghost_nodes_x)


# core function of thermal and mechanical
def compute_accelerated_velocity(n_x, n_y, horizon_mask, distance_matrix, partial_area_matrix,
                                 weight, dilation, nlength, alpha, rho, bz):
    """
    Use three functions in Physical_Field_Calculation to calculate total displacement field.
    """
    fx = np.zeros_like(distance_matrix, dtype=float)  # (N, N)
    fy = np.zeros_like(distance_matrix, dtype=float)
    dforce = np.zeros_like(distance_matrix, dtype=float)

    theta_i = dilation[:, None]  # N×1
    theta_j = dilation[None, :]  # 1×N

    bulk = a1 * (theta_i + theta_j) - a2 * dtemp  # N×1 广播到 N×N

    dforce[horizon_mask] = (4.0 * weight[horizon_mask] * (
            d * dot_xy[horizon_mask]/distance_matrix[horizon_mask] / nlength[horizon_mask] * bulk[horizon_mask] +
            b * (nlength[horizon_mask] - distance_matrix[horizon_mask])- alpha * dtemp * distance_matrix[horizon_mask]) * scr[horizon_mask]
    )

    fx[horizon_mask] = 0.5 * n_x[horizon_mask] * dforce[horizon_mask] * partial_area_matrix[horizon_mask]
    fy[horizon_mask] = 0.5 * n_y[horizon_mask] * dforce[horizon_mask] * partial_area_matrix[horizon_mask]

    Fx_new = np.sum(fx, axis=1)
    Fy_new = np.sum(fy, axis=1) + bz   # 加上体力（如重力）

    return Fx_new, Fy_new

# Apply initial boundary conditions
bz = np.zeros_like(Rmat)
pressure = 1000e3/dr
bz[0, :] = -pressure
bz = bz.flatten()

dtemp = 0
#dt_m = np.sqrt((2 * rho_s) / (np.pi * delta**2 * c1)) * 0.1  # Time step in seconds
#dt_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks, partial_area_matrix, horizon_mask,distance_matrix,delta)

massvec = 0.25 * (np.pi * delta**2 * h) * (4 * delta * b / dr) * 1.1
mass = np.ones_like(Ur) * massvec  # 形状自动匹配 U

# ------------------------
# Simulation loop settings
# ------------------------

total_time = 1000  # Total simulation time (e.g., 5 hours)
nsteps = int(1000)
#print_interval = int(10 / dt_m)  # Print progress every 10 simulated seconds
start_time = time.time()

# ------------------------
# Time-stepping loop
# ------------------------
save_times = [2, 4, 6, 8, 10]  # Save snapshots (in hours)
#save_steps = [int(t * 3600 / dt_m) for t in save_times]
T_record = []  # Store temperature snapshots

for step in range(nsteps):

    r_cur = r_flat + Ur  # 当前 r 坐标，长度 N
    z_cur = z_flat + Uz  # 当前 z 坐标
    # 当前构形的坐标差 y_ij = y_j - y_i
    dy_r = r_cur[None, :] - r_cur[:, None]  # N×N
    dy_z = z_cur[None, :] - z_cur[:, None]  # N×N
    # 当前键长 |y_ij|
    nlength = np.sqrt(dy_r ** 2 + dy_z ** 2)  # N×N
    sij[horizon_mask] = (nlength[horizon_mask] - distance_matrix[horizon_mask]) / distance_matrix[horizon_mask]
    dot_xy = dx_r * dy_r + dx_z * dy_z  # N×N
    dist = distance_matrix.copy()
    dist[dist == 0] = 1.0  # 把所有 0 距离改成 1，避免除以 0
    weight = delta / dist  # 现在不会再有 divide by zero

    n_r[horizon_mask] = dy_r[horizon_mask] / nlength[horizon_mask]
    n_z[horizon_mask] = dy_z[horizon_mask] / nlength[horizon_mask]

    dilation = pfc.compute_dilation_2d(distance_matrix, partial_area_matrix, horizon_mask, nlength, dot_xy, sij,
                                            weight, alpha, dtemp, d) * 2*(2*nu-1)/(nu-1)
    if step == 0:
        Fr_0, Fz_0 = compute_accelerated_velocity(n_r, n_z, horizon_mask, distance_matrix, partial_area_matrix,
                                 weight, dilation, nlength, alpha, rho_s, bz)
        Vr_half = (1 / 2) * (Fr_0 / mass)
        Vz_half = (1 / 2) * (Fz_0 / mass)
        Ur = Vr_half * 1 + Ur
        Uz = Vz_half * 1 + Uz

    else:
        previous_Ur = Ur
        previous_Uz = Uz
        Fr, Fz = compute_accelerated_velocity(n_r, n_z, horizon_mask, distance_matrix, partial_area_matrix,
                                                  weight, dilation, nlength, alpha, rho_s, bz)

        cr_n = ADR.compute_local_damping_coefficient(Fr, Fr_0, Vr_half, mass, Ur, 1)
        cz_n = ADR.compute_local_damping_coefficient(Fz, Fz_0, Vz_half, mass, Uz, 1)
        Fr_0 = Fr
        Fz_0 = Fz
        Vr_half, Ur = ADR.adr_update_velocity_displacement(Ur, Vr_half, Fr, cr_n, mass, 1)
        Vz_half, Uz = ADR.adr_update_velocity_displacement(Uz, Vz_half, Fz, cz_n, mass, 1)

        Ur_grid = Ur.reshape(Rmat.shape)
        Uz_grid = Uz.reshape(Rmat.shape)

        Ur_grid[:, ghost_inds_left] = 0
        Ur_grid[:, ghost_inds_right] = 0
        Ur_grid[ghost_inds_bottom, :] = 0

        Uz_grid[:, ghost_inds_left] = 0
        Uz_grid[:, ghost_inds_right] = 0
        Uz_grid[ghost_inds_bottom, :] = 0

        Ur_flat = Ur_grid.ravel()
        Uz_flat = Uz_grid.ravel()

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
Ur = Ur.reshape(Rmat.shape)
Uz = Uz.reshape(Rmat.shape)
# ------------------------
# Post-processing: visualization
# ------------------------
mask = np.ones(Rmat.shape, dtype=bool)

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
