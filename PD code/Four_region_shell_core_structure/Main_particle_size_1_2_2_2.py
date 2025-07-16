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
import grid_generator_rectangle as gg
import matplotlib
matplotlib.use('TkAgg')  # 或尝试 'QtAgg' 如果 TkAgg 报错

# ------------------------
# Physical and simulation parameters
# ------------------------

rho_s, cs, ks = 2681.0, 959.11, 93
rho_l, cl, kl = 2365.0, 1085.99, 70
rho_shell, cshell, kshell = 3970.0, 919.38, 10
E_core, alpha_core = 70e9, 2.1e-5
E_shell, alpha_shell = 70e9, 8.2e-6
Ts, Tl, = 933.15, 934.15
L = 395.60

nu = 0.25
G0 = 0.2   # J/m²
G1 = 10000   # J/m²

ghost_nodes_x = 3        # Number of ghost cells in the x (or r) direction
ghost_nodes_z = 3        # Number of ghost cells in the z direction
h = 1

tolerance = 1e-12
Tini = 303.15
Tult = 450.15
Tpre_avg = 303.15 # initial value of temperature
T_increment = 500#(Tult - Tini) / 1000 + Tini
# ==========================
#  Mechanical field coordinates
# Domain size, cell numbers
# ----------------------------

Lr1, Lz1, Nr1, Nz1 = 30e-6, 30e-6, 15, 15
Lr2, Lz2, Nr2, Nz2 = 30e-6, 10e-6, 30, 10
Lr3, Lz3, Nr3, Nz3 = 10e-6, 10e-6, 10, 10
Lr4, Lz4, Nr4, Nz4 = 10e-6, 30e-6, 10, 30
dr1, dz1 = Lr1 / Nr1, Lz1 / Nz1
dr2, dz2 = Lr2 / Nr2, Lz2 / Nz2
dr3, dz3 = Lr3 / Nr3, Lz3 / Nz3
dr4, dz4 = Lr4 / Nr4, Lz4 / Nz4
if not np.isclose(dr1, dz1):
    raise ValueError("dr1 and dz1 must be equal for region 1")
if not np.isclose(dr2, dz2):
    raise ValueError("dr2 and dz2 must be equal for region 2")
if not np.isclose(dr3, dz3):
    raise ValueError("dr3 and dz3 must be equal for region 3")
if not np.isclose(dr4, dz4):
    raise ValueError("dr4 and dz4 must be equal for region 4")

r1_start, z1_start = 0.0, 0.0
r2_start, z2_start = 0.0, z1_start + Lz1
r3_start, z3_start = r1_start + Lr1, z1_start + Lz1
r4_start, z4_start = r1_start + Lr1, 0.0  # ← 新区域右边紧接 r3

delta1, delta2, delta3, delta4 = 3 * dr1, 3 * dr2, 3 * dr3, 3 * dr4

s0_1 = np.sqrt(5 * np.pi * G0 / (18 * E_core  * delta1))
s0_2 = np.sqrt(5 * np.pi * G0 / (18 * E_shell * delta2))
s0_3 = np.sqrt(5 * np.pi * G0 / (18 * E_shell * delta3))
s0_4 = np.sqrt(5 * np.pi * G0 / (18 * E_shell * delta4))
# ---------- Region 1 ----------
r1_ghost_left_m,  r1_ghost_right_m  = True, True
z1_ghost_top_m,   z1_ghost_bot_m    = True, True
r1_ghost_left_th, r1_ghost_right_th = True, True
z1_ghost_top_th,  z1_ghost_bot_th   = True, True
# ---------- Region 2 ----------
r2_ghost_left_m,  r2_ghost_right_m  = True, True
z2_ghost_top_m,   z2_ghost_bot_m    = False, True
r2_ghost_left_th, r2_ghost_right_th = True, True
z2_ghost_top_th,  z2_ghost_bot_th   = True, True
# ---------- Region 3 ----------
r3_ghost_left_m,  r3_ghost_right_m  = True, False
z3_ghost_top_m,   z3_ghost_bot_m    = False, True
r3_ghost_left_th, r3_ghost_right_th = True, True
z3_ghost_top_th,  z3_ghost_bot_th   = True, True
# ---------- Region 4 ----------
r4_ghost_left_m,  r4_ghost_right_m  = True, False
z4_ghost_top_m,   z4_ghost_bot_m    = True, True
r4_ghost_left_th, r4_ghost_right_th = True, True
z4_ghost_top_th,  z4_ghost_bot_th   = True, True
r1_all_m, z1_all_m, Nr1_tot_m, Nz1_tot_m = gc.generate_coordinates(r1_start, z1_start, dr1, dz1, Lr1, Lz1, Nr1, Nz1, ghost_nodes_x, ghost_nodes_z, r1_ghost_left_m, r1_ghost_right_m, z1_ghost_top_m, z1_ghost_bot_m)
r1_all_th, z1_all_th, Nr1_tot_th, Nz1_tot_th = gc.generate_coordinates(r1_start, z1_start, dr1, dz1, Lr1, Lz1, Nr1, Nz1, ghost_nodes_x, ghost_nodes_z, r1_ghost_left_th, r1_ghost_right_th, z1_ghost_top_th, z1_ghost_bot_th)
r2_all_m, z2_all_m, Nr2_tot_m, Nz2_tot_m = gc.generate_coordinates(r2_start, z2_start, dr2, dz2, Lr2, Lz2, Nr2, Nz2, ghost_nodes_x, ghost_nodes_z, r2_ghost_left_m, r2_ghost_right_m, z2_ghost_top_m, z2_ghost_bot_m)
r2_all_th, z2_all_th, Nr2_tot_th, Nz2_tot_th = gc.generate_coordinates(r2_start, z2_start, dr2, dz2, Lr2, Lz2, Nr2, Nz2, ghost_nodes_x, ghost_nodes_z, r2_ghost_left_th, r2_ghost_right_th, z2_ghost_top_th, z2_ghost_bot_th)
r3_all_m, z3_all_m, Nr3_tot_m, Nz3_tot_m = gc.generate_coordinates(r3_start, z3_start, dr3, dz3, Lr3, Lz3, Nr3, Nz3, ghost_nodes_x, ghost_nodes_z, r3_ghost_left_m, r3_ghost_right_m, z3_ghost_top_m, z3_ghost_bot_m)
r3_all_th, z3_all_th, Nr3_tot_th, Nz3_tot_th = gc.generate_coordinates(r3_start, z3_start, dr3, dz3, Lr3, Lz3, Nr3, Nz3, ghost_nodes_x, ghost_nodes_z, r3_ghost_left_th, r3_ghost_right_th, z3_ghost_top_th, z3_ghost_bot_th)
r4_all_m, z4_all_m, Nr4_tot_m, Nz4_tot_m = gc.generate_coordinates(r4_start, z4_start, dr4, dz4, Lr4, Lz4, Nr4, Nz4, ghost_nodes_x, ghost_nodes_z, r4_ghost_left_m, r4_ghost_right_m, z4_ghost_top_m, z4_ghost_bot_m)
r4_all_th, z4_all_th, Nr4_tot_th, Nz4_tot_th = gc.generate_coordinates(r4_start, z4_start, dr4, dz4, Lr4, Lz4, Nr4, Nz4, ghost_nodes_x, ghost_nodes_z, r4_ghost_left_th, r4_ghost_right_th, z4_ghost_top_th, z4_ghost_bot_th)

R1mat_m, Z1mat_m = np.meshgrid(r1_all_m, z1_all_m, indexing='xy')
R2mat_m, Z2mat_m = np.meshgrid(r2_all_m, z2_all_m, indexing='xy')
R3mat_m, Z3mat_m = np.meshgrid(r3_all_m, z3_all_m, indexing='xy')
R4mat_m, Z4mat_m = np.meshgrid(r4_all_m, z4_all_m, indexing='xy')
R1mat_th, Z1mat_th = np.meshgrid(r1_all_th, z1_all_th, indexing='xy')
R2mat_th, Z2mat_th = np.meshgrid(r2_all_th, z2_all_th, indexing='xy')
R3mat_th, Z3mat_th = np.meshgrid(r3_all_th, z3_all_th, indexing='xy')
R4mat_th, Z4mat_th = np.meshgrid(r4_all_th, z4_all_th, indexing='xy')

r1_flat_m, z1_flat_m, r1_flat_th, z1_flat_th = R1mat_m.flatten(), Z1mat_m.flatten(), R1mat_th.flatten(), Z1mat_th.flatten()
r2_flat_m, z2_flat_m, r2_flat_th, z2_flat_th = R2mat_m.flatten(), Z2mat_m.flatten(), R2mat_th.flatten(), Z2mat_th.flatten()
r3_flat_m, z3_flat_m, r3_flat_th, z3_flat_th = R3mat_m.flatten(), Z3mat_m.flatten(), R3mat_th.flatten(), Z3mat_th.flatten()
r4_flat_m, z4_flat_m, r4_flat_th, z4_flat_th = R4mat_m.flatten(), Z4mat_m.flatten(), R4mat_th.flatten(), Z4mat_th.flatten()

dx1_r_m, dx1_z_m, dx1_r_th, dx1_z_th = r1_flat_m[None,:]-r1_flat_m[:,None], z1_flat_m[None,:]-z1_flat_m[:,None], r1_flat_th[None,:]-r1_flat_th[:,None], z1_flat_th[None,:]-z1_flat_th[:,None]
dx2_r_m, dx2_z_m, dx2_r_th, dx2_z_th = r2_flat_m[None,:]-r2_flat_m[:,None], z2_flat_m[None,:]-z2_flat_m[:,None], r2_flat_th[None,:]-r2_flat_th[:,None], z2_flat_th[None,:]-z2_flat_th[:,None]
dx3_r_m, dx3_z_m, dx3_r_th, dx3_z_th = r3_flat_m[None,:]-r3_flat_m[:,None], z3_flat_m[None,:]-z3_flat_m[:,None], r3_flat_th[None,:]-r3_flat_th[:,None], z3_flat_th[None,:]-z3_flat_th[:,None]
dx4_r_m, dx4_z_m, dx4_r_th, dx4_z_th = r4_flat_m[None,:]-r4_flat_m[:,None], z4_flat_m[None,:]-z4_flat_m[:,None], r4_flat_th[None,:]-r4_flat_th[:,None], z4_flat_th[None,:]-z4_flat_th[:,None]

distance_matrix1_m, distance_matrix1_th = np.sqrt(dx1_r_m**2 + dx1_z_m**2), np.sqrt(dx1_r_th**2 + dx1_z_th**2)
distance_matrix2_m, distance_matrix2_th = np.sqrt(dx2_r_m**2 + dx2_z_m**2), np.sqrt(dx2_r_th**2 + dx2_z_th**2)
distance_matrix3_m, distance_matrix3_th = np.sqrt(dx3_r_m**2 + dx3_z_m**2), np.sqrt(dx3_r_th**2 + dx3_z_th**2)
distance_matrix4_m, distance_matrix4_th = np.sqrt(dx4_r_m**2 + dx4_z_m**2), np.sqrt(dx4_r_th**2 + dx4_z_th**2)

partial_area_matrix1_m = area_matrix_calculator.compute_partial_area_matrix(r1_flat_m, z1_flat_m, dr1, dz1, delta1, distance_matrix1_m, tolerance)
partial_area_matrix1_th = area_matrix_calculator.compute_partial_area_matrix(r1_flat_th, z1_flat_th, dr1, dz1, delta1, distance_matrix1_th, tolerance)
partial_area_matrix2_m = area_matrix_calculator.compute_partial_area_matrix(r2_flat_m, z2_flat_m, dr2, dz2, delta2, distance_matrix2_m, tolerance)
partial_area_matrix2_th = area_matrix_calculator.compute_partial_area_matrix(r2_flat_th, z2_flat_th, dr2, dz2, delta2, distance_matrix2_th, tolerance)
partial_area_matrix3_m = area_matrix_calculator.compute_partial_area_matrix(r3_flat_m, z3_flat_m, dr3, dz3, delta3, distance_matrix3_m, tolerance)
partial_area_matrix3_th = area_matrix_calculator.compute_partial_area_matrix(r3_flat_th, z3_flat_th, dr3, dz3, delta3, distance_matrix3_th, tolerance)
partial_area_matrix4_m = area_matrix_calculator.compute_partial_area_matrix(r4_flat_m, z4_flat_m, dr4, dz4, delta4, distance_matrix4_m, tolerance)
partial_area_matrix4_th = area_matrix_calculator.compute_partial_area_matrix(r4_flat_th, z4_flat_th, dr4, dz4, delta4, distance_matrix4_th, tolerance)

horizon_mask1_m = (distance_matrix1_m > tolerance) & (partial_area_matrix1_m != 0.0)
horizon_mask1_th = (distance_matrix1_th > tolerance) & (partial_area_matrix1_th != 0.0)
horizon_mask2_m = (distance_matrix2_m > tolerance) & (partial_area_matrix2_m != 0.0)
horizon_mask2_th = (distance_matrix2_th > tolerance) & (partial_area_matrix2_th != 0.0)
horizon_mask3_m = (distance_matrix3_m > tolerance) & (partial_area_matrix3_m != 0.0)
horizon_mask3_th = (distance_matrix3_th > tolerance) & (partial_area_matrix3_th != 0.0)
horizon_mask4_m = (distance_matrix4_m > tolerance) & (partial_area_matrix4_m != 0.0)
horizon_mask4_th = (distance_matrix4_th > tolerance) & (partial_area_matrix4_th != 0.0)

true_indices1_th, true_indices2_th, true_indices3_th, true_indices4_th = (np.where(horizon_mask1_th),np.where(horizon_mask2_th),np.where(horizon_mask3_th),np.where(horizon_mask4_th))
mu1 = np.zeros_like(horizon_mask1_m, dtype=int)
mu2 = np.zeros_like(horizon_mask2_m, dtype=int)
mu3 = np.zeros_like(horizon_mask3_m, dtype=int)
mu4 = np.zeros_like(horizon_mask4_m, dtype=int)
mu1[horizon_mask1_m], mu2[horizon_mask2_m], mu3[horizon_mask3_m], mu4[horizon_mask4_m] = 1, 1, 1, 1


# ------------------------
# Preprocessing: shape factors, area weights, correction factors
# ------------------------
threshold_distance1, threshold_distance2, threshold_distance3, threshold_distance4 = (np.sqrt(2)*dr1, np.sqrt(2)*dr2, np.sqrt(2)*dr3, np.sqrt(2)*dr4)
factor_mat1 = np.where(distance_matrix1_th <= threshold_distance1 + tolerance, 1.125, 1.0)
factor_mat2 = np.where(distance_matrix2_th <= threshold_distance2 + tolerance, 1.125, 1.0)
factor_mat3 = np.where(distance_matrix3_th <= threshold_distance3 + tolerance, 1.125, 1.0)
factor_mat4 = np.where(distance_matrix4_th <= threshold_distance4 + tolerance, 1.125, 1.0)
true_indices1_th = np.where(horizon_mask1_th)
true_indices2_th = np.where(horizon_mask2_th)
true_indices3_th = np.where(horizon_mask3_th)
true_indices4_th = np.where(horizon_mask4_th)
shape_factor_matrix1_th = cf.compute_shape_factor_matrix(R1mat_th, true_indices1_th)
shape_factor_matrix2_th = cf.compute_shape_factor_matrix(R2mat_th, true_indices2_th)
shape_factor_matrix3_th = cf.compute_shape_factor_matrix(R3mat_th, true_indices3_th)
shape_factor_matrix4_th = cf.compute_shape_factor_matrix(R4mat_th, true_indices4_th)

row_sum1, row_sum2, row_sum3, row_sum4 = (partial_area_matrix1_m.sum(axis=1),partial_area_matrix2_m.sum(axis=1),partial_area_matrix3_m.sum(axis=1),partial_area_matrix4_m.sum(axis=1))
matrix_sum1 = row_sum1[:, None] + row_sum1[None, :]
matrix_sum2 = row_sum2[:, None] + row_sum2[None, :]
matrix_sum3 = row_sum3[:, None] + row_sum3[None, :]
matrix_sum4 = row_sum4[:, None] + row_sum4[None, :]

c1 = (6 * E_core)  / (np.pi * delta1**4 * h * (1 - 2 * nu) * (1 + nu))
c2 = (6 * E_shell) / (np.pi * delta2**4 * h * (1 - 2 * nu) * (1 + nu))
c3 = (6 * E_shell) / (np.pi * delta3**4 * h * (1 - 2 * nu) * (1 + nu))
c4 = (6 * E_shell) / (np.pi * delta4**4 * h * (1 - 2 * nu) * (1 + nu))
c_matrix1 = 2 * np.pi * delta1**2 / matrix_sum1 * c1
c_matrix2 = 2 * np.pi * delta2**2 / matrix_sum2 * c2
c_matrix3 = 2 * np.pi * delta3**2 / matrix_sum3 * c3
c_matrix4 = 2 * np.pi * delta4**2 / matrix_sum4 * c4

# ------------------------
# Initialize displacement, velocity, and acceleration fields, T
# ------------------------
Ur1 = np.zeros_like(R1mat_m).flatten()
Ur2 = np.zeros_like(R2mat_m).flatten()
Ur3 = np.zeros_like(R3mat_m).flatten()
Ur4 = np.zeros_like(R4mat_m).flatten()
Uz1 = np.zeros_like(Z1mat_m).flatten()
Uz2 = np.zeros_like(Z2mat_m).flatten()
Uz3 = np.zeros_like(Z3mat_m).flatten()
Uz4 = np.zeros_like(Z4mat_m).flatten()
Fr1 = np.zeros_like(R1mat_m).flatten()
Fr2 = np.zeros_like(R2mat_m).flatten()
Fr3 = np.zeros_like(R3mat_m).flatten()
Fr4 = np.zeros_like(R4mat_m).flatten()
Fz1 = np.zeros_like(Z1mat_m).flatten()
Fz2 = np.zeros_like(Z2mat_m).flatten()
Fz3 = np.zeros_like(Z3mat_m).flatten()
Fz4 = np.zeros_like(Z4mat_m).flatten()
br1, br2, br3, br4 = Ur1.copy(), Ur2.copy(), Ur3.copy(), Ur4.copy()
bz1, bz2, bz3, bz4 = Uz1.copy(), Uz2.copy(), Uz3.copy(), Uz4.copy()
T1, T2, T3, T4 = (np.full(R1mat_th.shape, Tini), np.full(R2mat_th.shape, Tini),np.full(R3mat_th.shape, Tini), np.full(R4mat_th.shape, Tini))

dt = 1
dt1_m = np.sqrt((2 * rho_s) / (np.pi * delta1**2 * c1)) * 0.2
dt2_m = np.sqrt((2 * rho_s) / (np.pi * delta2**2 * c2)) * 0.2
dt3_m = np.sqrt((2 * rho_s) / (np.pi * delta3**2 * c3)) * 0.2
dt4_m = np.sqrt((2 * rho_s) / (np.pi * delta4**2 * c4)) * 0.2

dt1_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks,partial_area_matrix1_th, horizon_mask1_th, distance_matrix1_th, delta1) * 0.8
dt2_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks,partial_area_matrix2_th, horizon_mask2_th, distance_matrix2_th, delta2) * 0.8
dt3_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks,partial_area_matrix3_th, horizon_mask3_th, distance_matrix3_th, delta3) * 0.8
dt4_th = cf.compute_dt_cr_th_solid_with_dist(rho_s, cs, ks,partial_area_matrix4_th, horizon_mask4_th, distance_matrix4_th, delta4) * 0.8
# 取最小值
dt_min = min(dt1_th, dt2_th, dt3_th, dt4_th)
# 所有时间步长统一为最小值
dt1_th = dt2_th = dt3_th = dt4_th = dt_min

# Region 1
ghost_inds_left1_m,  interior_inds_left1_m,  ghost_inds_left1_1d_m  = bc_funcs.get_left_ghost_indices(r1_all_m, ghost_nodes_x, Nz1_tot_m)
ghost_inds_right1_m, interior_inds_right1_m, ghost_inds_right1_1d_m = bc_funcs.get_right_ghost_indices(r1_all_m, ghost_nodes_x, Nz1_tot_m)
ghost_inds_top1_m,   interior_inds_top1_m,   ghost_inds_top1_1d_m    = bc_funcs.get_top_ghost_indices(z1_all_m, ghost_nodes_z, Nr1_tot_m)
ghost_inds_bot1_m,   interior_inds_bot1_m,   ghost_inds_bot1_1d_m    = bc_funcs.get_bottom_ghost_indices(z1_all_m, ghost_nodes_z, Nr1_tot_m)
ghost_inds_left1_th,  interior_inds_left1_th,  ghost_inds_left1_1d_th  = bc_funcs.get_left_ghost_indices(r1_all_th, ghost_nodes_x, Nz1_tot_th)
ghost_inds_right1_th, interior_inds_right1_th, ghost_inds_right1_1d_th = bc_funcs.get_right_ghost_indices(r1_all_th, ghost_nodes_x, Nz1_tot_th)
ghost_inds_top1_th,   interior_inds_top1_th,   ghost_inds_top1_1d_th   = bc_funcs.get_top_ghost_indices(z1_all_th, ghost_nodes_z, Nr1_tot_th)
ghost_inds_bot1_th,   interior_inds_bot1_th,   ghost_inds_bot1_1d_th   = bc_funcs.get_bottom_ghost_indices(z1_all_th, ghost_nodes_z, Nr1_tot_th)
# Region 2
ghost_inds_left2_m,  interior_inds_left2_m,  ghost_inds_left2_1d_m  = bc_funcs.get_left_ghost_indices(r2_all_m, ghost_nodes_x, Nz2_tot_m)
ghost_inds_right2_m, interior_inds_right2_m, ghost_inds_right2_1d_m = bc_funcs.get_right_ghost_indices(r2_all_m, ghost_nodes_x, Nz2_tot_m)
ghost_inds_top2_m,   interior_inds_top2_m,   ghost_inds_top2_1d_m    = bc_funcs.get_top_ghost_indices(z2_all_m, ghost_nodes_z, Nr2_tot_m)
ghost_inds_bot2_m,   interior_inds_bot2_m,   ghost_inds_bot2_1d_m    = bc_funcs.get_bottom_ghost_indices(z2_all_m, ghost_nodes_z, Nr2_tot_m)
ghost_inds_left2_th,  interior_inds_left2_th,  ghost_inds_left2_1d_th  = bc_funcs.get_left_ghost_indices(r2_all_th, ghost_nodes_x, Nz2_tot_th)
ghost_inds_right2_th, interior_inds_right2_th, ghost_inds_right2_1d_th = bc_funcs.get_right_ghost_indices(r2_all_th, ghost_nodes_x, Nz2_tot_th)
ghost_inds_top2_th,   interior_inds_top2_th,   ghost_inds_top2_1d_th   = bc_funcs.get_top_ghost_indices(z2_all_th, ghost_nodes_z, Nr2_tot_th)
ghost_inds_bot2_th,   interior_inds_bot2_th,   ghost_inds_bot2_1d_th   = bc_funcs.get_bottom_ghost_indices(z2_all_th, ghost_nodes_z, Nr2_tot_th)
# Region 3
ghost_inds_left3_m,  interior_inds_left3_m,  ghost_inds_left3_1d_m  = bc_funcs.get_left_ghost_indices(r3_all_m, ghost_nodes_x, Nz3_tot_m)
ghost_inds_right3_m, interior_inds_right3_m, ghost_inds_right3_1d_m = bc_funcs.get_right_ghost_indices(r3_all_m, ghost_nodes_x, Nz3_tot_m)
ghost_inds_top3_m,   interior_inds_top3_m,   ghost_inds_top3_1d_m    = bc_funcs.get_top_ghost_indices(z3_all_m, ghost_nodes_z, Nr3_tot_m)
ghost_inds_bot3_m,   interior_inds_bot3_m,   ghost_inds_bot3_1d_m    = bc_funcs.get_bottom_ghost_indices(z3_all_m, ghost_nodes_z, Nr3_tot_m)
ghost_inds_left3_th,  interior_inds_left3_th,  ghost_inds_left3_1d_th  = bc_funcs.get_left_ghost_indices(r3_all_th, ghost_nodes_x, Nz3_tot_th)
ghost_inds_right3_th, interior_inds_right3_th, ghost_inds_right3_1d_th = bc_funcs.get_right_ghost_indices(r3_all_th, ghost_nodes_x, Nz3_tot_th)
ghost_inds_top3_th,   interior_inds_top3_th,   ghost_inds_top3_1d_th   = bc_funcs.get_top_ghost_indices(z3_all_th, ghost_nodes_z, Nr3_tot_th)
ghost_inds_bot3_th,   interior_inds_bot3_th,   ghost_inds_bot3_1d_th   = bc_funcs.get_bottom_ghost_indices(z3_all_th, ghost_nodes_z, Nr3_tot_th)
# Region 4
ghost_inds_left4_m,  interior_inds_left4_m,  ghost_inds_left4_1d_m  = bc_funcs.get_left_ghost_indices(r4_all_m, ghost_nodes_x, Nz4_tot_m)
ghost_inds_right4_m, interior_inds_right4_m, ghost_inds_right4_1d_m = bc_funcs.get_right_ghost_indices(r4_all_m, ghost_nodes_x, Nz4_tot_m)
ghost_inds_top4_m,   interior_inds_top4_m,   ghost_inds_top4_1d_m    = bc_funcs.get_top_ghost_indices(z4_all_m, ghost_nodes_z, Nr4_tot_m)
ghost_inds_bot4_m,   interior_inds_bot4_m,   ghost_inds_bot4_1d_m    = bc_funcs.get_bottom_ghost_indices(z4_all_m, ghost_nodes_z, Nr4_tot_m)
ghost_inds_left4_th,  interior_inds_left4_th,  ghost_inds_left4_1d_th  = bc_funcs.get_left_ghost_indices(r4_all_th, ghost_nodes_x, Nz4_tot_th)
ghost_inds_right4_th, interior_inds_right4_th, ghost_inds_right4_1d_th = bc_funcs.get_right_ghost_indices(r4_all_th, ghost_nodes_x, Nz4_tot_th)
ghost_inds_top4_th,   interior_inds_top4_th,   ghost_inds_top4_1d_th   = bc_funcs.get_top_ghost_indices(z4_all_th, ghost_nodes_z, Nr4_tot_th)
ghost_inds_bot4_th,   interior_inds_bot4_th,   ghost_inds_bot4_1d_th   = bc_funcs.get_bottom_ghost_indices(z4_all_th, ghost_nodes_z, Nr4_tot_th)

coord_index4_top = gg.get_exact_neighbor_points_same_spacing(R4mat_th, Z4mat_th, ghost_inds_top4_th, r3_all_th, z3_all_th, 0, ghost_nodes_z, ghost_nodes_x)
coord_index4_left = gg.get_fine_neighbor_points(R4mat_th, Z4mat_th, ghost_inds_left4_th, dr4, dz4, R1mat_th, Z1mat_th, r1_all_th, z1_all_th, 1, ghost_nodes_z, ghost_nodes_x)
coord_index2_bot  = gg.get_fine_neighbor_points(R2mat_th, Z2mat_th, ghost_inds_bot2_th, dr2, dz2, R1mat_th, Z1mat_th, r1_all_th, z1_all_th, 0, ghost_nodes_z, ghost_nodes_x)
coord_index2_right = gg.get_exact_neighbor_points_same_spacing(R2mat_th, Z2mat_th, ghost_inds_right2_th, r3_all_th, z3_all_th, 1, ghost_nodes_z, ghost_nodes_x)
coord_index3_left = gg.get_exact_neighbor_points_same_spacing(R3mat_th, Z3mat_th, ghost_inds_left3_th, r2_all_th, z2_all_th, 1, ghost_nodes_z, ghost_nodes_x)
coord_index3_bot = gg.get_exact_neighbor_points_same_spacing(R3mat_th, Z3mat_th, ghost_inds_bot3_th, r4_all_th, z4_all_th, 0, ghost_nodes_z, ghost_nodes_x)
coord_index1_top   = gg.get_coarse_neighbor_points(R1mat_th, Z1mat_th, ghost_inds_top1_th, dr2, dz2, r2_all_th, z2_all_th, 0, ghost_nodes_z, ghost_nodes_x)
coord_index1_right = gg.get_coarse_neighbor_points(R1mat_th, Z1mat_th, ghost_inds_right1_th, dr4, dz4, r4_all_th, z4_all_th, 1, ghost_nodes_z, ghost_nodes_x)

coord_index4_top_m    = gg.get_exact_neighbor_points_same_spacing(R4mat_m, Z4mat_m, ghost_inds_top4_m, r3_all_m, z3_all_m, 0, ghost_nodes_z, ghost_nodes_x)
coord_index4_left_m   = gg.get_fine_neighbor_points(R4mat_m, Z4mat_m, ghost_inds_left4_m, dr4, dz4, R1mat_m, Z1mat_m, r1_all_m, z1_all_m, 1, ghost_nodes_z, ghost_nodes_x)
coord_index2_bot_m    = gg.get_fine_neighbor_points(R2mat_m, Z2mat_m, ghost_inds_bot2_m, dr2, dz2, R1mat_m, Z1mat_m, r1_all_m, z1_all_m, 0, ghost_nodes_z, ghost_nodes_x)
coord_index2_right_m  = gg.get_exact_neighbor_points_same_spacing(R2mat_m, Z2mat_m, ghost_inds_right2_m, r3_all_m, z3_all_m, 1, ghost_nodes_z, ghost_nodes_x)
coord_index3_left_m   = gg.get_exact_neighbor_points_same_spacing(R3mat_m, Z3mat_m, ghost_inds_left3_m, r2_all_m, z2_all_m, 1, ghost_nodes_z, ghost_nodes_x)
coord_index3_bot_m    = gg.get_exact_neighbor_points_same_spacing(R3mat_m, Z3mat_m, ghost_inds_bot3_m, r4_all_m, z4_all_m, 0, ghost_nodes_z, ghost_nodes_x)
coord_index1_top_m    = gg.get_coarse_neighbor_points(R1mat_m, Z1mat_m, ghost_inds_top1_m, dr2, dz2, r2_all_m, z2_all_m, 0, ghost_nodes_z, ghost_nodes_x)
coord_index1_right_m  = gg.get_coarse_neighbor_points(R1mat_m, Z1mat_m, ghost_inds_right1_m, dr4, dz4, r4_all_m, z4_all_m, 1, ghost_nodes_z, ghost_nodes_x)
# Apply initial conditions
# Region 4
T4 = bc_funcs.apply_bc_zero_flux(T4, ghost_inds_bot4_th, interior_inds_bot4_th, axis=0)
T4 = bc_funcs.apply_bc_dirichlet_mirror(T4, ghost_inds_right4_th, interior_inds_right4_th, T_increment, axis=1, z_mask=None, r_mask=None)
T4 = gg.interpolate_temperature_direct_match(T4, T3, coord_index4_top)
T4 = gg.interpolate_temperature_for_fine(T4, T1, coord_index4_left)

# Region 2
T2 = bc_funcs.apply_bc_zero_flux(T2, ghost_inds_left2_th, interior_inds_left2_th, axis=1)
T2 = bc_funcs.apply_bc_dirichlet_mirror(T2, ghost_inds_top2_th, interior_inds_top2_th, T_increment, axis=0, z_mask=None, r_mask=None)
T2 = gg.interpolate_temperature_for_fine(T2, T1, coord_index2_bot)
T2 = gg.interpolate_temperature_direct_match(T2, T3, coord_index2_right)

# Region 3
T3 = bc_funcs.apply_bc_dirichlet_mirror(T3, ghost_inds_right3_th, interior_inds_right3_th, T_increment, axis=1, z_mask=None, r_mask=None)
T3 = bc_funcs.apply_bc_dirichlet_mirror(T3, ghost_inds_top3_th, interior_inds_top3_th, T_increment, axis=0, z_mask=None, r_mask=None)
T3 = gg.interpolate_temperature_direct_match(T3, T2, coord_index3_left)
T3 = gg.interpolate_temperature_direct_match(T3, T4, coord_index3_bot)

# Region 1
T1 = bc_funcs.apply_bc_zero_flux(T1, ghost_inds_bot1_th, interior_inds_bot1_th, axis=0)
T1 = bc_funcs.apply_bc_zero_flux(T1, ghost_inds_left1_th, interior_inds_left1_th, axis=1)
T1 = gg.interpolate_temperature_for_coarse(T1, T2, coord_index1_top)
T1 = gg.interpolate_temperature_for_coarse(T1, T4, coord_index1_right)

dir_r1_m, dir_z1_m = pfc.compute_direction_matrix(r1_flat_m, z1_flat_m, Ur1, Uz1, horizon_mask1_m)
dir_r2_m, dir_z2_m = pfc.compute_direction_matrix(r2_flat_m, z2_flat_m, Ur2, Uz2, horizon_mask2_m)
dir_r3_m, dir_z3_m = pfc.compute_direction_matrix(r3_flat_m, z3_flat_m, Ur3, Uz3, horizon_mask3_m)
dir_r4_m, dir_z4_m = pfc.compute_direction_matrix(r4_flat_m, z4_flat_m, Ur4, Uz4, horizon_mask4_m)

# core function of thermal and mechanical
def compute_accelerated_velocity(Ur_curr, Uz_curr,r_flat, z_flat,horizon_mask_m,
    dir_r, dir_z, c_matrix, partial_area_matrix_m, T_curr, Tpre_avg, mu_curr, distance_matrix, s0, alpha, CorrList_T):
    """
    Calculate the updated acceleration (force density) and failure matrix (mu).
    """
    Ur_new = Ur_curr
    Uz_new = Uz_curr

    T_curr = T_curr.flatten()
    T_m = pfc.filter_array_by_indices_keep_only(T_curr, CorrList_T)
    Tavg = pfc.compute_delta_temperature(T_m, Tpre_avg)
    Relative_elongation = pfc.compute_s_matrix(r_flat, z_flat, Ur_new, Uz_new, horizon_mask_m, distance_matrix)
    mu_new = pfc.update_mu_by_failure(mu_curr, Relative_elongation, s0)

    thermal_term = alpha * (1 + nu) * Tavg
    Fr_new = mu_new * c_matrix * dir_r * (Relative_elongation - thermal_term) * partial_area_matrix_m
    Fz_new = mu_new * c_matrix * dir_z * (Relative_elongation - thermal_term) * partial_area_matrix_m

    Fr_new = np.sum(Fr_new, axis=1)
    Fz_new = np.sum(Fz_new, axis=1)

    return Fr_new, Fz_new, mu_new

# Temperature update function
def update_temperature(
    Tcurr, Hcurr, Kcurr, mask_core, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell,
    factor_mat, partial_area_matrix_th, shape_factor_matrix, distance_matrix_th, horizon_mask_th,
    ks, kl, delta, dt_th, kshell, Nz_tot_th, Nr_tot_th,true_indices
):

    flux = Kcurr @ Tcurr.flatten()               # Apply nonlocal heat flux
    flux = flux.reshape(Nz_tot_th, Nr_tot_th)
    Hnew = Hcurr + flux                          # Update enthalpy
    Tnew = cf.get_temperature(Hnew, mask_core, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell)   # Convert to temperature

    Knew = cf.build_K_matrix(Tnew, cf.compute_thermal_conductivity_matrix, factor_mat,
        partial_area_matrix_th, shape_factor_matrix,
        distance_matrix_th, horizon_mask_th,
        ks, kl, Ts, Tl, delta, dt_th, true_indices,mask_core, kshell)

    return Tnew, Hnew, Knew

CorrList_T1 = pfc.shrink_Tth_by_matching_coords(R1mat_m, Z1mat_m, R1mat_th, Z1mat_th)
CorrList_T2 = pfc.shrink_Tth_by_matching_coords(R2mat_m, Z2mat_m, R2mat_th, Z2mat_th)
CorrList_T3 = pfc.shrink_Tth_by_matching_coords(R3mat_m, Z3mat_m, R3mat_th, Z3mat_th)
CorrList_T4 = pfc.shrink_Tth_by_matching_coords(R4mat_m, Z4mat_m, R4mat_th, Z4mat_th)

lambda_diag_matrix1 = ADR.compute_lambda_diag_matrix(partial_area_matrix1_m, distance_matrix1_m, c_matrix1, horizon_mask1_m, dt, dx1_r_m, dx1_z_m)
lambda_diag_matrix2 = ADR.compute_lambda_diag_matrix(partial_area_matrix2_m, distance_matrix2_m, c_matrix2, horizon_mask2_m, dt, dx2_r_m, dx2_z_m)
lambda_diag_matrix3 = ADR.compute_lambda_diag_matrix(partial_area_matrix3_m, distance_matrix3_m, c_matrix3, horizon_mask3_m, dt, dx3_r_m, dx3_z_m)
lambda_diag_matrix4 = ADR.compute_lambda_diag_matrix(partial_area_matrix4_m, distance_matrix4_m, c_matrix4, horizon_mask4_m, dt, dx4_r_m, dx4_z_m)

"""Get the average value between each bond: """
T1_m = pfc.filter_array_by_indices_keep_only(T1.flatten(), CorrList_T1)
T2_m = pfc.filter_array_by_indices_keep_only(T2.flatten(), CorrList_T2)
T3_m = pfc.filter_array_by_indices_keep_only(T3.flatten(), CorrList_T3)
T4_m = pfc.filter_array_by_indices_keep_only(T4.flatten(), CorrList_T4)
mask_core = True
mask_shell = ~mask_core
H1 = cf.get_enthalpy(T1, mask_core,  rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell)
H2 = cf.get_enthalpy(T2, mask_shell,  rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell)
H3 = cf.get_enthalpy(T3, mask_shell, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell)
H4 = cf.get_enthalpy(T4, mask_shell, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell)

 # Initial enthalpy
K1 = cf.build_K_matrix(T1, cf.compute_thermal_conductivity_matrix, factor_mat1,
                       partial_area_matrix1_th, shape_factor_matrix1_th,
                       distance_matrix1_th, horizon_mask1_th,
                       ks, kl, Ts, Tl, delta1, dt1_th, true_indices1_th,mask_core, kshell)
K2 = cf.build_K_matrix(T2, cf.compute_thermal_conductivity_matrix, factor_mat2,
                       partial_area_matrix2_th, shape_factor_matrix2_th,
                       distance_matrix2_th, horizon_mask2_th,
                       ks, kl, Ts, Tl, delta2, dt2_th, true_indices2_th,mask_shell, kshell)
K3 = cf.build_K_matrix(T3, cf.compute_thermal_conductivity_matrix, factor_mat3,
                       partial_area_matrix3_th, shape_factor_matrix3_th,
                       distance_matrix3_th, horizon_mask3_th,
                       ks, kl, Ts, Tl, delta3, dt3_th, true_indices3_th, mask_shell, kshell)
K4 = cf.build_K_matrix(T4, cf.compute_thermal_conductivity_matrix, factor_mat4,
                       partial_area_matrix4_th, shape_factor_matrix4_th,
                       distance_matrix4_th, horizon_mask4_th,
                       ks, kl, Ts, Tl, delta4, dt4_th, true_indices4_th,mask_shell, kshell)
area_matrix1_m = partial_area_matrix1_m.copy(); np.fill_diagonal(area_matrix1_m, 0)
phi1 = 1 - np.sum(mu1 * area_matrix1_m, axis=1) / np.sum(area_matrix1_m, axis=1)
phi1 = phi1.reshape(R1mat_m.shape)

area_matrix2_m = partial_area_matrix2_m.copy(); np.fill_diagonal(area_matrix2_m, 0)
phi2 = 1 - np.sum(mu2 * area_matrix2_m, axis=1) / np.sum(area_matrix2_m, axis=1)
phi2 = phi2.reshape(R2mat_m.shape)

area_matrix3_m = partial_area_matrix3_m.copy(); np.fill_diagonal(area_matrix3_m, 0)
phi3 = 1 - np.sum(mu3 * area_matrix3_m, axis=1) / np.sum(area_matrix3_m, axis=1)
phi3 = phi3.reshape(R3mat_m.shape)

area_matrix4_m = partial_area_matrix4_m.copy(); np.fill_diagonal(area_matrix4_m, 0)
phi4 = 1 - np.sum(mu4 * area_matrix4_m, axis=1) / np.sum(area_matrix4_m, axis=1)
phi4 = phi4.reshape(R4mat_m.shape)

# ------------------------
# Simulation loop settings
# ------------------------
total_time = 1e-6/dt1_th # Total simulation time (e.g., 5 hours)
print(f"Total simulation time: {total_time:.6f} s")

nsteps_th = int(total_time)
nsteps_m = int(10000)
start_time = time.time()
# ------------------------
# Time-stepping loop
# ------------------------
for step1 in range(nsteps_th):
    if step1  == 10:
        print(f"Step {step1} / total = {total_time*dt1_th:.6f} s")
    T_increment = (Tult - Tini) / nsteps_th * (step1 + 1) + Tini
    T1, H1, K1 = update_temperature(
        T1, H1, K1, mask_core, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell,
        factor_mat1, partial_area_matrix1_th, shape_factor_matrix1_th, distance_matrix1_th, horizon_mask1_th,
        ks, kl, delta1, dt1_th, kshell, Nz1_tot_th, Nr1_tot_th, true_indices1_th
    )
    T2, H2, K2 = update_temperature(
        T2, H2, K2, mask_shell, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell,
        factor_mat2, partial_area_matrix2_th, shape_factor_matrix2_th, distance_matrix2_th, horizon_mask2_th,
        ks, kl, delta2, dt2_th, kshell, Nz2_tot_th, Nr2_tot_th, true_indices2_th
    )
    T3, H3, K3 = update_temperature(
        T3, H3, K3, mask_shell, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell,
        factor_mat3, partial_area_matrix3_th, shape_factor_matrix3_th, distance_matrix3_th, horizon_mask3_th,
        ks, kl, delta3, dt3_th, kshell, Nz3_tot_th, Nr3_tot_th, true_indices3_th
    )
    T4, H4, K4 = update_temperature(
        T4, H4, K4, mask_shell, rho_s, rho_l, cs, cl, L, Ts, Tl, rho_shell, cshell,
        factor_mat4, partial_area_matrix4_th, shape_factor_matrix4_th, distance_matrix4_th, horizon_mask4_th,
        ks, kl, delta4, dt4_th, kshell, Nz4_tot_th, Nr4_tot_th, true_indices4_th
    )

    # Region 4
    T4 = gg.interpolate_temperature_direct_match(T4, T3, coord_index4_top)
    T4 = gg.interpolate_temperature_for_fine(T4, T1, coord_index4_left)
    T4 = bc_funcs.apply_bc_zero_flux(T4, ghost_inds_bot4_th, interior_inds_bot4_th, axis=0)
    T4 = bc_funcs.apply_bc_dirichlet_mirror(T4, ghost_inds_right4_th, interior_inds_right4_th, T_increment, axis=1,z_mask=None, r_mask=None)

    # Region 2
    T2 = gg.interpolate_temperature_for_fine(T2, T1, coord_index2_bot)
    T2 = gg.interpolate_temperature_direct_match(T2, T3, coord_index2_right)
    T2 = bc_funcs.apply_bc_zero_flux(T2, ghost_inds_left2_th, interior_inds_left2_th, axis=1)
    T2 = bc_funcs.apply_bc_dirichlet_mirror(T2, ghost_inds_top2_th, interior_inds_top2_th, T_increment, axis=0,z_mask=None, r_mask=None)

    # Region 3
    T3 = gg.interpolate_temperature_direct_match(T3, T2, coord_index3_left)
    T3 = gg.interpolate_temperature_direct_match(T3, T4, coord_index3_bot)
    T3 = bc_funcs.apply_bc_dirichlet_mirror(T3, ghost_inds_right3_th, interior_inds_right3_th, T_increment, axis=1,z_mask=None, r_mask=None)
    T3 = bc_funcs.apply_bc_dirichlet_mirror(T3, ghost_inds_top3_th, interior_inds_top3_th, T_increment, axis=0,z_mask=None, r_mask=None)

    # Region 1
    T1 = gg.interpolate_temperature_for_coarse(T1, T2, coord_index1_top)
    T1 = gg.interpolate_temperature_for_coarse(T1, T4, coord_index1_right)
    T1 = bc_funcs.apply_bc_zero_flux(T1, ghost_inds_bot1_th, interior_inds_bot1_th, axis=0)
    T1 = bc_funcs.apply_bc_zero_flux(T1, ghost_inds_left1_th, interior_inds_left1_th, axis=1)


    phi1 = 1 - np.sum(mu1 * area_matrix1_m, axis=1) / np.sum(area_matrix1_m, axis=1)
    phi1 = phi1.reshape(R1mat_m.shape)
    phi2 = 1 - np.sum(mu2 * area_matrix2_m, axis=1) / np.sum(area_matrix2_m, axis=1)
    phi2 = phi2.reshape(R2mat_m.shape)
    phi3 = 1 - np.sum(mu3 * area_matrix3_m, axis=1) / np.sum(area_matrix3_m, axis=1)
    phi3 = phi3.reshape(R3mat_m.shape)
    phi4 = 1 - np.sum(mu4 * area_matrix4_m, axis=1) / np.sum(area_matrix4_m, axis=1)
    phi4 = phi4.reshape(R4mat_m.shape)

    if step1 == 0:
        # Region 1
        Fr1, Fz1, mu1 = compute_accelerated_velocity(Ur1, Uz1, r1_flat_m, z1_flat_m, horizon_mask1_m, dir_r1_m,dir_z1_m, c_matrix1,
                                                     partial_area_matrix1_m, T1, Tpre_avg, mu1, distance_matrix1_m, s0_1, alpha_core, CorrList_T1)
        lambda_diag_matrix1 = ADR.compute_lambda_diag_matrix(partial_area_matrix1_m, distance_matrix1_m, c_matrix1,
                                                             horizon_mask1_m, dt, dx1_r_m, dx1_z_m)
        Vr1_half = Fr1 / lambda_diag_matrix1 * dt; Ur1 += Vr1_half * dt
        Vz1_half = Fz1 / lambda_diag_matrix1 * dt; Uz1 += Vz1_half * dt

        # Region 2
        Fr2, Fz2, mu2 = compute_accelerated_velocity(Ur2, Uz2, r2_flat_m, z2_flat_m, horizon_mask2_m, dir_r2_m,dir_z2_m, c_matrix2,
                                                     partial_area_matrix2_m, T2, Tpre_avg, mu2, distance_matrix2_m, s0_2, alpha_shell, CorrList_T2)
        lambda_diag_matrix2 = ADR.compute_lambda_diag_matrix(partial_area_matrix2_m, distance_matrix2_m, c_matrix2,
                                                             horizon_mask2_m, dt, dx2_r_m, dx2_z_m)
        Vr2_half = Fr2 / lambda_diag_matrix2 * dt; Ur2 += Vr2_half * dt
        Vz2_half = Fz2 / lambda_diag_matrix2 * dt; Uz2 += Vz2_half * dt
        # Region 3
        Fr3, Fz3, mu3 = compute_accelerated_velocity(Ur3, Uz3, r3_flat_m, z3_flat_m, horizon_mask3_m, dir_r3_m,dir_z3_m, c_matrix3,
                                                     partial_area_matrix3_m, T3, Tpre_avg, mu3, distance_matrix3_m, s0_3, alpha_shell, CorrList_T3)
        lambda_diag_matrix3 = ADR.compute_lambda_diag_matrix(partial_area_matrix3_m, distance_matrix3_m, c_matrix3,
                                                             horizon_mask3_m, dt, dx3_r_m, dx3_z_m)
        Vr3_half = Fr3 / lambda_diag_matrix3 * dt; Ur3 += Vr3_half * dt
        Vz3_half = Fz3 / lambda_diag_matrix3 * dt; Uz3 += Vz3_half * dt
        # Region 4
        Fr4, Fz4, mu4 = compute_accelerated_velocity(Ur4, Uz4, r4_flat_m, z4_flat_m, horizon_mask4_m, dir_r4_m,dir_z4_m, c_matrix4,
                                                     partial_area_matrix4_m, T4, Tpre_avg, mu4, distance_matrix4_m, s0_4, alpha_shell, CorrList_T4)
        lambda_diag_matrix4 = ADR.compute_lambda_diag_matrix(partial_area_matrix4_m, distance_matrix4_m, c_matrix4,
                                                             horizon_mask4_m, dt, dx4_r_m, dx4_z_m)
        Vr4_half = Fr4 / lambda_diag_matrix4 * dt; Ur4 += Vr4_half * dt
        Vz4_half = Fz4 / lambda_diag_matrix4 * dt; Uz4 += Vz4_half * dt
        Uz1[ghost_inds_bot1_1d_m] = 0
        Ur1[ghost_inds_left1_1d_m] = 0
        Ur2[ghost_inds_left2_1d_m] = 0
        Uz4[ghost_inds_bot4_1d_m] = 0
        Ur1 = Ur1.reshape(R1mat_m.shape); Uz1 = Uz1.reshape(R1mat_m.shape)
        Ur2 = Ur2.reshape(R2mat_m.shape); Uz2 = Uz2.reshape(R2mat_m.shape)
        Ur3 = Ur3.reshape(R3mat_m.shape); Uz3 = Uz3.reshape(R3mat_m.shape)
        Ur4 = Ur4.reshape(R4mat_m.shape); Uz4 = Uz4.reshape(R4mat_m.shape)
        Ur1 = gg.interpolate_temperature_for_coarse(Ur1, Ur2, coord_index1_top_m)
        Uz1 = gg.interpolate_temperature_for_coarse(Uz1, Uz2, coord_index1_top_m)
        Ur1 = gg.interpolate_temperature_for_coarse(Ur1, Ur4, coord_index1_right_m)
        Uz1 = gg.interpolate_temperature_for_coarse(Uz1, Uz4, coord_index1_right_m)
        Ur2 = gg.interpolate_temperature_for_fine(Ur2, Ur1, coord_index2_bot_m)
        Uz2 = gg.interpolate_temperature_for_fine(Uz2, Uz1, coord_index2_bot_m)
        Ur2 = gg.interpolate_temperature_direct_match(Ur2, Ur3, coord_index2_right_m)
        Uz2 = gg.interpolate_temperature_direct_match(Uz2, Uz3, coord_index2_right_m)
        Ur3 = gg.interpolate_temperature_direct_match(Ur3, Ur2, coord_index3_left_m)
        Uz3 = gg.interpolate_temperature_direct_match(Uz3, Uz2, coord_index3_left_m)
        Ur3 = gg.interpolate_temperature_direct_match(Ur3, Ur4, coord_index3_bot_m)
        Uz3 = gg.interpolate_temperature_direct_match(Uz3, Uz4, coord_index3_bot_m)
        Ur4 = gg.interpolate_temperature_for_fine(Ur4, Ur1, coord_index4_left_m)
        Uz4 = gg.interpolate_temperature_for_fine(Uz4, Uz1, coord_index4_left_m)
        Ur4 = gg.interpolate_temperature_direct_match(Ur4, Ur3, coord_index4_top_m)
        Uz4 = gg.interpolate_temperature_direct_match(Uz4, Uz3, coord_index4_top_m)
        Ur1 = Ur1.flatten();Uz1 = Uz1.flatten()
        Ur2 = Ur2.flatten();Uz2 = Uz2.flatten()
        Ur3 = Ur3.flatten();Uz3 = Uz3.flatten()
        Ur4 = Ur4.flatten();Uz4 = Uz4.flatten()
        Fr1_0 = Fr1;
        Fz1_0 = Fz1
        Fr2_0 = Fr2;
        Fz2_0 = Fz2
        Fr3_0 = Fr3;
        Fz3_0 = Fz3
        Fr4_0 = Fr4;
        Fz4_0 = Fz4

    for step2 in range(nsteps_m):
        if step2 % 10 == 0:
            print(f"Step {step2}")
        # 这里是你的主计算逻辑

            # 这里是你的主计算逻辑
        previous_Ur1, previous_Uz1 = Ur1.copy(), Uz1.copy()
        previous_Ur2, previous_Uz2 = Ur2.copy(), Uz2.copy()
        previous_Ur3, previous_Uz3 = Ur3.copy(), Uz3.copy()
        previous_Ur4, previous_Uz4 = Ur4.copy(), Uz4.copy()
        Fr1, Fz1, mu1 = compute_accelerated_velocity(Ur1, Uz1, r1_flat_m, z1_flat_m, horizon_mask1_m,
                                                     dir_r1_m, dir_z1_m, c_matrix1, partial_area_matrix1_m, T1,
                                                     Tpre_avg, mu1, distance_matrix1_m, s0_1, alpha_core,
                                                     CorrList_T1)
        Fr2, Fz2, mu2 = compute_accelerated_velocity(Ur2, Uz2, r2_flat_m, z2_flat_m, horizon_mask2_m,
                                                     dir_r2_m, dir_z2_m, c_matrix2, partial_area_matrix2_m, T2,
                                                     Tpre_avg, mu2, distance_matrix2_m, s0_2, alpha_shell,
                                                     CorrList_T2)
        Fr3, Fz3, mu3 = compute_accelerated_velocity(Ur3, Uz3, r3_flat_m, z3_flat_m, horizon_mask3_m,
                                                     dir_r3_m, dir_z3_m, c_matrix3, partial_area_matrix3_m, T3,
                                                     Tpre_avg, mu3, distance_matrix3_m, s0_3, alpha_shell,
                                                     CorrList_T3)
        Fr4, Fz4, mu4 = compute_accelerated_velocity(Ur4, Uz4, r4_flat_m, z4_flat_m, horizon_mask4_m,
                                                     dir_r4_m, dir_z4_m, c_matrix4, partial_area_matrix4_m, T4,
                                                     Tpre_avg, mu4, distance_matrix4_m, s0_4, alpha_shell,
                                                     CorrList_T4)
        cr1_n = ADR.compute_local_damping_coefficient(Fr1, Fr1_0, Vr1_half, lambda_diag_matrix1, Ur1, dt)
        cz1_n = ADR.compute_local_damping_coefficient(Fz1, Fz1_0, Vz1_half, lambda_diag_matrix1, Uz1, dt)
        cr2_n = ADR.compute_local_damping_coefficient(Fr2, Fr2_0, Vr2_half, lambda_diag_matrix2, Ur2, dt)
        cz2_n = ADR.compute_local_damping_coefficient(Fz2, Fz2_0, Vz2_half, lambda_diag_matrix2, Uz2, dt)
        cr3_n = ADR.compute_local_damping_coefficient(Fr3, Fr3_0, Vr3_half, lambda_diag_matrix3, Ur3, dt)
        cz3_n = ADR.compute_local_damping_coefficient(Fz3, Fz3_0, Vz3_half, lambda_diag_matrix3, Uz3, dt)
        cr4_n = ADR.compute_local_damping_coefficient(Fr4, Fr4_0, Vr4_half, lambda_diag_matrix4, Ur4, dt)
        cz4_n = ADR.compute_local_damping_coefficient(Fz4, Fz4_0, Vz4_half, lambda_diag_matrix4, Uz4, dt)

        Fr1_0, Fz1_0 = Fr1.copy(), Fz1.copy()
        Fr2_0, Fz2_0 = Fr2.copy(), Fz2.copy()
        Fr3_0, Fz3_0 = Fr3.copy(), Fz3.copy()
        Fr4_0, Fz4_0 = Fr4.copy(), Fz4.copy()

        Vr1_half, Ur1 = ADR.adr_update_velocity_displacement(Ur1, Vr1_half, Fr1, cr1_n, lambda_diag_matrix1, dt)
        Vz1_half, Uz1 = ADR.adr_update_velocity_displacement(Uz1, Vz1_half, Fz1, cz1_n, lambda_diag_matrix1, dt)
        Vr2_half, Ur2 = ADR.adr_update_velocity_displacement(Ur2, Vr2_half, Fr2, cr2_n, lambda_diag_matrix2, dt)
        Vz2_half, Uz2 = ADR.adr_update_velocity_displacement(Uz2, Vz2_half, Fz2, cz2_n, lambda_diag_matrix2, dt)
        Vr3_half, Ur3 = ADR.adr_update_velocity_displacement(Ur3, Vr3_half, Fr3, cr3_n, lambda_diag_matrix3, dt)
        Vz3_half, Uz3 = ADR.adr_update_velocity_displacement(Uz3, Vz3_half, Fz3, cz3_n, lambda_diag_matrix3, dt)
        Vr4_half, Ur4 = ADR.adr_update_velocity_displacement(Ur4, Vr4_half, Fr4, cr4_n, lambda_diag_matrix4, dt)
        Vz4_half, Uz4 = ADR.adr_update_velocity_displacement(Uz4, Vz4_half, Fz4, cz4_n, lambda_diag_matrix4, dt)

        Uz1[ghost_inds_bot1_1d_m] = 0
        Ur1[ghost_inds_left1_1d_m] = 0
        Ur2[ghost_inds_left2_1d_m] = 0
        Uz4[ghost_inds_bot4_1d_m] = 0

        Ur1 = Ur1.reshape(R1mat_m.shape);
        Uz1 = Uz1.reshape(R1mat_m.shape)
        Ur2 = Ur2.reshape(R2mat_m.shape);
        Uz2 = Uz2.reshape(R2mat_m.shape)
        Ur3 = Ur3.reshape(R3mat_m.shape);
        Uz3 = Uz3.reshape(R3mat_m.shape)
        Ur4 = Ur4.reshape(R4mat_m.shape);
        Uz4 = Uz4.reshape(R4mat_m.shape)
        Ur1 = gg.interpolate_temperature_for_coarse(Ur1, Ur2, coord_index1_top_m)
        Uz1 = gg.interpolate_temperature_for_coarse(Uz1, Uz2, coord_index1_top_m)
        Ur1 = gg.interpolate_temperature_for_coarse(Ur1, Ur4, coord_index1_right_m)
        Uz1 = gg.interpolate_temperature_for_coarse(Uz1, Uz4, coord_index1_right_m)
        Ur2 = gg.interpolate_temperature_for_fine(Ur2, Ur1, coord_index2_bot_m)
        Uz2 = gg.interpolate_temperature_for_fine(Uz2, Uz1, coord_index2_bot_m)
        Ur2 = gg.interpolate_temperature_direct_match(Ur2, Ur3, coord_index2_right_m)
        Uz2 = gg.interpolate_temperature_direct_match(Uz2, Uz3, coord_index2_right_m)
        Ur3 = gg.interpolate_temperature_direct_match(Ur3, Ur2, coord_index3_left_m)
        Uz3 = gg.interpolate_temperature_direct_match(Uz3, Uz2, coord_index3_left_m)
        Ur3 = gg.interpolate_temperature_direct_match(Ur3, Ur4, coord_index3_bot_m)
        Uz3 = gg.interpolate_temperature_direct_match(Uz3, Uz4, coord_index3_bot_m)
        Ur4 = gg.interpolate_temperature_for_fine(Ur4, Ur1, coord_index4_left_m)
        Uz4 = gg.interpolate_temperature_for_fine(Uz4, Uz1, coord_index4_left_m)
        Ur4 = gg.interpolate_temperature_direct_match(Ur4, Ur3, coord_index4_top_m)
        Uz4 = gg.interpolate_temperature_direct_match(Uz4, Uz3, coord_index4_top_m)
        Ur1 = Ur1.flatten();
        Uz1 = Uz1.flatten()
        Ur2 = Ur2.flatten();
        Uz2 = Uz2.flatten()
        Ur3 = Ur3.flatten();
        Uz3 = Uz3.flatten()
        Ur4 = Ur4.flatten();
        Uz4 = Uz4.flatten()
        # dir_r, dir_z = pfc.compute_direction_matrix(r_flat_m, z_flat_m, Ur, Uz, horizon_mask_m)
        # 计算当前位移增量的RMS
        U1 = np.sqrt(Ur1 ** 2 + Uz1 ** 2);
        U1_prev = np.sqrt(previous_Ur1 ** 2 + previous_Uz1 ** 2)
        U2 = np.sqrt(Ur2 ** 2 + Uz2 ** 2);
        U2_prev = np.sqrt(previous_Ur2 ** 2 + previous_Uz2 ** 2)
        U3 = np.sqrt(Ur3 ** 2 + Uz3 ** 2);
        U3_prev = np.sqrt(previous_Ur3 ** 2 + previous_Uz3 ** 2)
        U4 = np.sqrt(Ur4 ** 2 + Uz4 ** 2);
        U4_prev = np.sqrt(previous_Ur4 ** 2 + previous_Uz4 ** 2)
        Ru1 = np.sqrt(np.sum((U1 - U1_prev) ** 2) / len(U1))
        Ru2 = np.sqrt(np.sum((U2 - U2_prev) ** 2) / len(U2))
        Ru3 = np.sqrt(np.sum((U3 - U3_prev) ** 2) / len(U3))
        Ru4 = np.sqrt(np.sum((U4 - U4_prev) ** 2) / len(U4))
        if all(Ru < 1e-9 for Ru in (Ru1, Ru2, Ru3, Ru4)):
            print(
                f"Convergence reached at step {step2} with RMS: Ru1={Ru1:.2e}, Ru2={Ru2:.2e}, Ru3={Ru3:.2e}, Ru4={Ru4:.2e}")
            break
end_time = time.time()
print(f"Calculation finished, elapsed real time = {end_time - start_time:.2f}s")
time = end_time - start_time

# ------------------------
# Post-processing: visualization
# ------------------------
#mask_m and mask_th Used to exclude displacement and temperature of boundary particles
mask1_th = np.ones(R1mat_th.shape, dtype=bool)
mask1_th[ghost_inds_top1_th, :] = False
mask1_th[:, ghost_inds_right1_th] = False

mask2_th = np.ones(R2mat_th.shape, dtype=bool)
mask2_th[ghost_inds_bot2_th, :] = False
mask2_th[:, ghost_inds_right2_th] = False

mask3_th = np.ones(R3mat_th.shape, dtype=bool)
mask3_th[ghost_inds_bot3_th, :] = False
mask3_th[:, ghost_inds_left3_th] = False

mask4_th = np.ones(R4mat_th.shape, dtype=bool)
mask4_th[ghost_inds_top4_th, :] = False
mask4_th[:, ghost_inds_left4_th] = False

levels = np.linspace(np.min([T1.min(), T2.min(), T3.min(), T4.min()]),
                     np.max([T1.max(), T2.max(), T3.max(), T4.max()]),
                     50)
plot.temperature_combined_four_regions(
    R1mat_th, Z1mat_th, T1, mask1_th,
    R2mat_th, Z2mat_th, T2, mask2_th,
    R3mat_th, Z3mat_th, T3, mask3_th,
    R4mat_th, Z4mat_th, T4, mask4_th,
    Lr1, Lr4, Lz1, Lz2,
    total_time, nsteps_th,levels
)

# Region 1 (mask1_m)
mask1_m = np.ones(R1mat_m.shape, dtype=bool)
mask1_m[ghost_inds_top1_m, :] = False
mask1_m[:, ghost_inds_right1_m] = False

# Region 2 (mask2_m)
mask2_m = np.ones(R2mat_m.shape, dtype=bool)
mask2_m[ghost_inds_bot2_m, :] = False
mask2_m[:, ghost_inds_right2_m] = False

# Region 3 (mask3_m)
mask3_m = np.ones(R3mat_m.shape, dtype=bool)
mask3_m[ghost_inds_bot3_m, :] = False
mask3_m[:, ghost_inds_left3_m] = False

# Region 4 (mask4_m)
mask4_m = np.ones(R4mat_m.shape, dtype=bool)
mask4_m[ghost_inds_top4_m, :] = False
mask4_m[:, ghost_inds_left4_m] = False

plot.displacement_combined_four_regions(
    R1mat_m, Z1mat_m, Ur1, Uz1, mask1_m,
    R2mat_m, Z2mat_m, Ur2, Uz2, mask2_m,
    R3mat_m, Z3mat_m, Ur3, Uz3, mask3_m,
    R4mat_m, Z4mat_m, Ur4, Uz4, mask4_m,
    Lr1, Lr4, Lz1, Lz2,
    total_time=nsteps_m*dt1_m,
    nsteps=nsteps_m,
    levels=np.linspace(-1e-6, 1e-6, 100)  # 或根据最大位移自动设置
)


"""
U = np.sqrt(Ur**2 + Uz**2)

area_matrix_m = partial_area_matrix_m.copy()  # 避免影响原数组
np.fill_diagonal(area_matrix_m, 0)
phi = 1 - np.sum(mu * area_matrix_m, axis=1) / np.sum(area_matrix_m, axis=1)
phi = phi.reshape(Rmat_m.shape)
plot.plot_mu_field(Rmat_m, Zmat_m, phi, mask_m, Lr, Lz, title_prefix="Bond Failure State", save=False)

mask_th = np.ones(T.shape, dtype=bool)
mask_th[ghost_inds_top_m, :] = True
mask_th[ghost_inds_bottom_m, :] = False
mask_th[:, ghost_inds_left_m] = False
mask_th[:, ghost_inds_right_m] = True
plot.temperature(Rmat_th, Zmat_th, T, total_time, nsteps_th, dr, dz, time, mask_th, Lr,Lz)

"""