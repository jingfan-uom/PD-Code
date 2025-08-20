# region_utils.py

import area_matrix_calculator
import core_funcs as cf
import numpy as np
import bc_funcs as bc

def compute_region_matrices(args):
    coords, dr, delta, tolerance, slice_id = args

    r_flat = coords[:, 0]
    z_flat = coords[:, 1]

    N = len(r_flat)
    distance_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        dx_r = r_flat[i] - r_flat  # shape (N,)
        dx_z = z_flat[i] - z_flat
        distance_matrix[i, :] = np.sqrt(dx_r ** 2 + dx_z ** 2)

    partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
        r_flat, z_flat, dr, dr, delta, distance_matrix, tolerance
    )

    horizon_mask = ((distance_matrix > tolerance) & (partial_area_matrix != 0.0))
    true_indices = np.where(horizon_mask)

    # Save as npz file
    np.savez_compressed(f"matrix_slice_{slice_id}.npz",
                        distance=distance_matrix,
                        area=partial_area_matrix,
                        mask=horizon_mask,
                        indices=true_indices)
    return f"matrix_slice_{slice_id}.npz"

# Multiprocess_task_function.py

def update_temperature_for_region(args):
    (
        region_id,
        Tcurr,
        Hcurr,
        Kmat,
        factor_mat,
        partial_area_matrix,
        shape_factor_matrix,
        distance_matrix,
        horizon_mask,
        true_indices,
        delta,
        dt,
        rho_s, cs, cl, L, Ts, Tl, ks, kl 
    ) = args

    # 1. Non-local flux update enthalpy
    flux = Kmat @ Tcurr
    Hnew = Hcurr + flux
    # 2. Enthalpy conversion temperature
    Tnew = cf.get_temperature(Hnew, rho_s, cs, cl, L, Ts, Tl)
    # 3. Build a new thermal conductivity matrix
    Knew = cf.build_K_matrix(
        Tnew,
        cf.compute_thermal_conductivity_matrix,
        factor_mat,
        partial_area_matrix,
        shape_factor_matrix,
        distance_matrix,
        horizon_mask,
        true_indices,
        ks, kl, Ts, Tl, delta, dt
    )

    return region_id, Tnew, Hnew, Knew

# mechanical_calculations.py

import Physical_Field_Calculation as pfc  # Make sure the import path is correct

def compute_accelerated_velocity_initial(args):
    """
    Compute updated acceleration for radial and axial directions (mechanical field).

    Parameters in args:
    - coords_all: (N, 2) array, combined physical + ghost coords (radial, axial)
    - Ur_curr: radial displacement array (N,)
    - Uz_curr: axial displacement array (N,)
    - horizon_mask: (N, N) boolean mask
    - dir_r, dir_z: direction matrices (N, N)
    - c: stiffness coefficient
    - partial_area_matrix: (N, N) area overlap matrix
    - rho: density
    - T_curr: current temperature array (N,)
    - T_prev: previous temperature array (N,)
    - nu: Poisson's ratio
    - alpha: thermal expansion coefficient
    """

    (coords_all, Ur_curr, Uz_curr, horizon_mask, dir_r, dir_z,
     c, partial_area_matrix, rho, T_curr, T_prev, nu, alpha, CorrList_T) = args

    # Build coordinate matrices from coords_all

    Ur_new = Ur_curr
    Uz_new = Uz_curr
    # Average temperature change
    T_curr =  pfc.filter_array_by_indices_keep_only(T_curr, CorrList_T)
    Tavg = pfc.compute_delta_temperature(T_curr, T_prev)
    # Relative elongation
    Relative_elongation = pfc.compute_s_matrix(coords_all, Ur_new, Uz_new, horizon_mask)

    # Acceleration computation (radial + axial)
    Ar_new = dir_r * c * (Relative_elongation - (1 + nu) * alpha * Tavg) * partial_area_matrix / rho
    Az_new = dir_z * c * (Relative_elongation - (1 + nu) * alpha * Tavg) * partial_area_matrix / rho

    # Sum over neighbor contributions
    Ar_new = np.sum(Ar_new, axis=1).reshape(Ur_curr.shape)
    Az_new = np.sum(Az_new, axis=1).reshape(Uz_curr.shape)

    return Ar_new, Az_new

def compute_mechanical_step(args):

    (coords_all, Ur_curr, Uz_curr,
     horizon_mask, dir_r, dir_z,
     c, partial_area_matrix, rho,
     T_curr, T_prev, nu, alpha, CorrList_T,
     Fr_0, Fz_0, Vr_half, Vz_half, lambda_diag_matrix, ADR) = args

    # --- Step 1: Filter temperatures and compute average temperature change ---
    T_curr_filtered = pfc.filter_array_by_indices_keep_only(T_curr, CorrList_T)
    Tavg = pfc.compute_delta_temperature(T_curr_filtered, T_prev)

    # --- Step 2: Compute relative bond elongations ---
    Relative_elongation = pfc.compute_s_matrix(coords_all, Ur_curr, Uz_curr, horizon_mask)

    # --- Step 3: Compute accelerations ---
    Ar_new = dir_r * c * (Relative_elongation - (1 + nu) * alpha * Tavg) * partial_area_matrix / rho
    Az_new = dir_z * c * (Relative_elongation - (1 + nu) * alpha * Tavg) * partial_area_matrix / rho

    Ar_new = np.sum(Ar_new, axis=1).reshape(Ur_curr.shape)
    Az_new = np.sum(Az_new, axis=1).reshape(Uz_curr.shape)

    # --- Step 4: Compute forces ---
    Fr_new = Ar_new * rho
    Fz_new = Az_new * rho

    # --- Step 5: Compute damping coefficients ---
    cr_n = ADR.compute_local_damping_coefficient(Fr_new, Fr_0, Vr_half, lambda_diag_matrix, Ur_curr, 1)
    cz_n = ADR.compute_local_damping_coefficient(Fz_new, Fz_0, Vz_half, lambda_diag_matrix, Uz_curr, 1)

    # --- Step 6: Update historical forces ---
    Fr_0 = Fr_new
    Fz_0 = Fz_new

    # --- Step 7: Update velocities and displacements ---
    Vr_half, Ur_curr = ADR.adr_update_velocity_displacement(Ur_curr, Vr_half, Fr_new, cr_n, lambda_diag_matrix, 1)
    Vz_half, Uz_curr = ADR.adr_update_velocity_displacement(Uz_curr, Vz_half, Fz_new, cz_n, lambda_diag_matrix, 1)

    return Ur_curr, Uz_curr, Fr_0, Fz_0, Vr_half, Vz_half
