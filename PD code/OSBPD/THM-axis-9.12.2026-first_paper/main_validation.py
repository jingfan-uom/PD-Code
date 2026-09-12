import generate_coordinates as gc
import ADR
import time
import Multiprocess_task_function as mt
import bc_funcs as bc
import core_funcs as cf
import Physical_Field_Calculation as pfc
import geometry_utils as geom
import plot_utils as plot
import shell_only_functions as sof
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
import os


# NaNO3@polyimide(P84) material parameters from THM-2.
phase_volume_expansion_fraction = float(
    os.environ.get("THM_PHASE_VOLUME_EXPANSION_FRACTION", "0.097")
)
rho_s, cs, ks = 2260.0, 1330.0, 0.46
rho_l, cl, kl = rho_s / (1.0 + phase_volume_expansion_fraction), 1330.0, 0.46
rho_air, cair, kair = 1, 1010, 0.0263
rho_shell = 1400.0
c_shell = float(os.environ.get("THM_P84_HEAT_CAPACITY", "1090.0"))
k_shell = float(os.environ.get("THM_P84_CONDUCTIVITY", "0.22"))
nu_core, nu_shell = 0.30, 0.34
E_core, alpha_core_l, alpha_core_s = 32.7e9, 7.7e-4, 7.7e-3
E_shell, alpha_shell = 3.581e9, 4.1e-6
comp_s, comp_l = 3.0 * (1.0 - 2.0 * nu_core) / E_core, 1.86e-10  # Pa^-1
sigmat = 140e6

stop_after_first_pressure_check = os.environ.get("THM_STOP_AFTER_FIRST_PRESSURE_CHECK", "1") != "0"
enable_mechanical_thermal_strain = (
    os.environ.get("THM_ENABLE_MECH_THERMAL_STRAIN", "0") == "1"
)
direct_pressure_load_env = os.environ.get("THM_DIRECT_PRESSURE_LOAD")
direct_pressure_load = (
    None if direct_pressure_load_env is None else float(direct_pressure_load_env)
)
nsteps_m = int(os.environ.get("THM_NSTEPS_M", "10000"))
shell_transfer_conversion_steps = 100

T_phase_center = float(os.environ.get("THM_PHASE_CENTER_K", str(307.0 + 273.15)))
phase_interval = 5.0
Ts = T_phase_center - 0.5 * phase_interval
Tl = T_phase_center + 0.5 * phase_interval
L = 172000.0
tolerance = 1e-14
Tsurr = 600.65
Tinit = 560.65
Tpre_avg = Tinit

"""Initialization of the single region and temperatures."""

r = 100e-6
dshell = 20e-6
r_core = r - dshell
modify_coordinates = False
coordinate_x_scale = 1.0
enable_cracking = False
enable_crack_pressure_force = False
damage_threshold = 0.5
size = 1e-6

generate_core = True
dr = 2 * size
r_start = 0
ghost_nodes_r = 3  # Number of ghost cells in the x (or r) direction.
n_slices = 1

dt_ADR = 1
total_time = 0.02
core_shell_rms = float(os.environ.get("THM_CORE_SHELL_RMS", "5.0e-11"))
shell_only_rms = float(os.environ.get("THM_SHELL_ONLY_RMS", "5.0e-12"))
graph_point_t = False
graph_point_m = False
graph_shell_inner_surface_points = False
pressure_plot = os.environ.get("THM_PRESSURE_PLOT", "1") != "0"
use_boundary_ramp = os.environ.get("THM_Upressure_ramp_steps SE_BOUNDARY_RAMP", "1") != "0"
core_shell_timemagnify = 1.2
shell_only_timemagnify = 1.2
thermal_stepper = "euler"
shell_transfer_match_rel_tol = 0.02
pressure_ramp_steps = 1


if __name__ == "__main__":
    start_time2 = time.time()

    """1. Definition of coordinates for the single region."""

    r_inner = float(r_core)
    if modify_coordinates:
        outer_axis_x = coordinate_x_scale * r
        outer_axis_z = r
        inner_axis_x = outer_axis_x - dshell
        inner_axis_z = outer_axis_z - dshell
    else:
        outer_axis_x = r
        outer_axis_z = r
        inner_axis_x = r_inner
        inner_axis_z = r_inner

    Nr = int(outer_axis_x / dr + 1e-12)
    delta = ghost_nodes_r * dr
    length = 2 * r
    dr_i = dr
    delta_i = delta
    coordinate_common = {
        "dr": dr,
        "length": length,
        "r_inner": r_inner,
        "outer_axis_x": outer_axis_x,
        "outer_axis_z": outer_axis_z,
        "inner_axis_x": inner_axis_x,
        "inner_axis_z": inner_axis_z,
    }
    core_shell_coordinates = dict(coordinate_common, generate_core=generate_core)
    shell_only_coordinates = dict(coordinate_common, generate_core=False)

    print(
        f"Single region: dr = {dr:.2e}, delta = {delta:.2e}, "
        f"length = {length:.2e}, Nr = {Nr}"
    )
    print("[Grid] thermal and mechanical core-shell grids include the core region")
    print(
        f"[Geometry] outer ellipse axes = ({outer_axis_x:.6e}, {outer_axis_z:.6e}) m; "
        f"inner ellipse axes = ({inner_axis_x:.6e}, {inner_axis_z:.6e}) m"
    )
    print(
        f"[THM-5 settings] mechanical thermal strain enabled={enable_mechanical_thermal_strain}, "
        f"pressure model=phase_and_core_thermal_volume, "
        f"direct pressure load={direct_pressure_load}, "
        f"shell transfer conversion steps={shell_transfer_conversion_steps}, "
        f"pressure ramp steps={pressure_ramp_steps}"
    )

    # ---------------- Temperature field coordinates ----------------
    phys_coords_t, ghost_coords_t, n_points_t, ghost_dict_t = gc.generate_one_slice_coordinates(
        r, Nr, ghost_nodes_r, core_shell_coordinates,
        False, True, graph_point_t, r_start,
        modify_coordinates, coordinate_x_scale,
    )
    coords_t = np.vstack([phys_coords_t, ghost_coords_t])
    coords_phy_t = phys_coords_t
    coords_all_t = coords_t
    coords_all_t_list = [coords_t]

    print(f"[Temperature] Number of particles: {n_points_t}")

    # ---------------- Mechanical field coordinates ----------------
    phys_coords_m, ghost_coords_m, n_points_m, ghost_dict_m = gc.generate_one_slice_coordinates(
        r, Nr, ghost_nodes_r, core_shell_coordinates,
        False, False, graph_point_m, r_start,
        modify_coordinates, coordinate_x_scale,
    )
    coords_m = np.vstack([phys_coords_m, ghost_coords_m])
    coords_phy_m = phys_coords_m
    coords_all_m = coords_m
    coords_all_m_list = [coords_m]

    print(f"[Mechanical] Number of particles: {n_points_m}")

    # ---------------- Shell-only temperature field coordinates ----------------
    phys_coords_shell_only_t, ghost_coords_shell_only_t, n_points_shell_only_t, ghost_dict_shell_only_t = (
        gc.generate_one_slice_coordinates(
            r, Nr, ghost_nodes_r, shell_only_coordinates,
            False, True, graph_point_t, r_start,
            modify_coordinates, coordinate_x_scale,
        )
    )
    coords_shell_only_t = np.vstack([phys_coords_shell_only_t, ghost_coords_shell_only_t])
    coords_phy_shell_only_t = phys_coords_shell_only_t
    coords_all_shell_only_t = coords_shell_only_t
    coords_all_shell_only_t_list = [coords_shell_only_t]

    print(f"[Temperature shell-only] Number of particles: {n_points_shell_only_t}")

    # ---------------- Shell-only mechanical field coordinates ----------------
    phys_coords_shell_only_m, ghost_coords_shell_only_m, n_points_shell_only_m, ghost_dict_shell_only_m = (
        gc.generate_one_slice_coordinates(
            r, Nr, ghost_nodes_r, shell_only_coordinates,
            False, False, graph_point_m, r_start,
            modify_coordinates, coordinate_x_scale,
        )
    )
    coords_shell_only_m = np.vstack([phys_coords_shell_only_m, ghost_coords_shell_only_m])
    coords_phy_shell_only_m = phys_coords_shell_only_m
    coords_all_shell_only_m = coords_shell_only_m
    coords_all_shell_only_m_list = [coords_shell_only_m]

    print(f"[Mechanical shell-only] Number of particles: {n_points_shell_only_m}")
    print(f"[Coordinates] Generation finished, elapsed real time = {time.time() - start_time2:.2f}s")

    # ---------------- Temperature field CSR matrices ----------------
    start_time_t = time.time()
    csr_indptr_t, csr_indices_t, csr_dist_t, csr_area_t = mt.compute_region_matrices(
        (coords_t, dr_i, delta_i, tolerance)
    )
    print(
        "[Temperature] Calculation of partial_area_matrices finished, "
        f"elapsed real time = {time.time() - start_time_t:.2f}s"
    )

    # ---------------- Mechanical field CSR matrices ----------------
    start_time_m = time.time()
    csr_indptr_m, csr_indices_m, csr_dist_m, csr_area_m = mt.compute_region_matrices(
        (coords_m, dr_i, delta_i, tolerance)
    )
    print(
        "[Mechanical] Calculation of partial_area_matrices finished, "
        f"elapsed real time = {time.time() - start_time_m:.2f}s"
    )

    # ---------------- Mechanical core-shell edge properties ----------------
    start_time_edge_m = time.time()
    core_level_m = (
        ((coords_m[:, 0] - r_start) / inner_axis_x) ** 2
        + ((coords_m[:, 1] - r) / inner_axis_z) ** 2
    )
    mask_core_m = core_level_m < 1.0

    indptr = csr_indptr_m
    indices = csr_indices_m
    area = csr_area_m
    N = len(indptr) - 1
    nnz = len(indices)
    eps = 1e-20
    edge_i = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))

    L_core_edge, L_shell_edge, L_total_edge = pfc.precompute_edge_core_shell_lengths_ellipse(
        coords_all=coords_all_m[:, :2],
        edge_i=edge_i,
        edge_j=indices,
        core_axis_x=inner_axis_x,
        core_axis_z=inner_axis_z,
        r_center=r_start,
        z_center=r,
    )

    nu_node = np.where(mask_core_m, nu_core, nu_shell).astype(np.float64)
    E_node = np.where(mask_core_m, E_core, E_shell).astype(np.float64)
    alpha_node = np.where(mask_core_m, alpha_core_s, alpha_shell).astype(np.float64)
    if not enable_mechanical_thermal_strain:
        alpha_node = np.zeros_like(alpha_node)
    lamda0_node = (E_node * nu_node) / ((1.0 - 2.0 * nu_node) * (1.0 + nu_node) + eps)
    miu_node = E_node / (2.0 * (1.0 + nu_node) + eps)
    lamda_node = lamda0_node - miu_node
    kprime_node = (E_node / (1.0 - 2.0 * nu_node + eps)) * alpha_node

    E_edge = pfc.build_edge_property_harmonic_from_lengths(
        E_node, mask_core_m, edge_i, indices,
        L_core_edge, L_shell_edge, L_total_edge,
    )
    nu_edge = pfc.build_edge_property_harmonic_from_lengths(
        nu_node, mask_core_m, edge_i, indices,
        L_core_edge, L_shell_edge, L_total_edge,
    )
    alpha_edge = pfc.build_edge_property_harmonic_from_lengths(
        alpha_node, mask_core_m, edge_i, indices,
        L_core_edge, L_shell_edge, L_total_edge,
    )
    miu_edge = pfc.build_edge_property_harmonic_from_lengths(
        miu_node, mask_core_m, edge_i, indices,
        L_core_edge, L_shell_edge, L_total_edge,
    )
    lamda_edge = pfc.build_edge_property_harmonic_from_lengths(
        lamda_node, mask_core_m, edge_i, indices,
        L_core_edge, L_shell_edge, L_total_edge,
    )
    kprime_edge = pfc.build_edge_property_harmonic_from_lengths(
        kprime_node, mask_core_m, edge_i, indices,
        L_core_edge, L_shell_edge, L_total_edge,
    )

    mask_self = indices == edge_i
    if np.any(mask_self):
        csr_area_m = csr_area_m.copy()
        csr_area_m[mask_self] = 0.0
        area = csr_area_m

    actual_area_node = np.bincount(edge_i, weights=area, minlength=N).astype(np.float64)
    theoretical_area_node = np.full(N, np.pi * delta_i ** 2, dtype=np.float64)
    numerator = theoretical_area_node[edge_i] + theoretical_area_node[indices]
    denom = actual_area_node[edge_i] + actual_area_node[indices]
    csr = np.divide(
        numerator,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=np.abs(denom) > eps,
    )

    mixed_edge_count = int(np.count_nonzero((L_core_edge > eps) & (L_shell_edge > eps)))
    print(
        f"[Mechanical] Core-shell edge properties finished, edges={nnz}, "
        f"mixed_edges={mixed_edge_count}, elapsed real time = {time.time() - start_time_edge_m:.2f}s"
    )

    # ---------------- Shell-only temperature field CSR matrices ----------------
    start_time_shell_only_t = time.time()
    (
        csr_indptr_shell_only_t,
        csr_indices_shell_only_t,
        csr_dist_shell_only_t,
        csr_area_shell_only_t,
    ) = mt.compute_region_matrices((coords_shell_only_t, dr_i, delta_i, tolerance))
    print(
        "[Temperature shell-only] Calculation of partial_area_matrices finished, "
        f"elapsed real time = {time.time() - start_time_shell_only_t:.2f}s"
    )

    # ---------------- Shell-only mechanical field CSR matrices ----------------
    start_time_shell_only_m = time.time()
    (
        csr_indptr_shell_only_m,
        csr_indices_shell_only_m,
        csr_dist_shell_only_m,
        csr_area_shell_only_m,
    ) = mt.compute_region_matrices((coords_shell_only_m, dr_i, delta_i, tolerance))
    print(
        "[Mechanical shell-only] Calculation of partial_area_matrices finished, "
        f"elapsed real time = {time.time() - start_time_shell_only_m:.2f}s"
    )

    """2. Definition of right boundary particle information."""
    ghostcoords_right_t = ghost_dict_t["right"][:, :2]
    coords_t_pltb = np.vstack([
        phys_coords_t[:, :2],
        ghost_dict_t["left"][:, :2],
        ghost_dict_t["top"][:, :2],
        ghost_dict_t["bot"][:, :2],
    ])
    ghost_idx_right_t, phys_idx_right_t = bc.find_circle_mirror_pairs_multilayer(
        ghostcoords_right_t,
        coords_t_pltb,
        dr_i,
        r,
        r_start,
        modify_coordinates,
        coordinate_x_scale,
    )
    boundary_neighbors_t = {
        "right": {
            "ghost_indices": ghost_idx_right_t,
            "phys_indices": phys_idx_right_t,
        }
    }

    """3. Definition of temperature field."""
    rho_void = 0
    Cp_void = 0
    k_void = 0

    core_level_t = (
        ((coords_t[:, 0] - r_start) / inner_axis_x) ** 2
        + ((coords_t[:, 1] - r) / inner_axis_z) ** 2
    )
    mask_core_t = core_level_t < 1.0

    threshold_distance = np.sqrt(2) * dr_i + tolerance
    factor_data_t = np.where(
        csr_dist_t <= threshold_distance,
        1.125,
        1.0,
    ).astype(np.float64)

    r_node_t = coords_t[:, 0]
    N_t = csr_indptr_t.size - 1
    edge_i_t = np.repeat(np.arange(N_t, dtype=np.int64), np.diff(csr_indptr_t))
    shape_factor_t = cf.compute_shape_factor_edge_csr(
        r_node_t,
        edge_i_t,
        csr_indices_t,
        mechanical=False,
    )

    dt_th_core = cf.compute_dt_cr_th_solid_with_csr(
        rho_s,
        cs,
        ks,
        csr_indptr_t,
        csr_dist_t,
        csr_area_t,
        delta_i,
    )
    dt_th_shell = cf.compute_dt_cr_th_solid_with_csr(
        rho_shell,
        c_shell,
        k_shell,
        csr_indptr_t,
        csr_dist_t,
        csr_area_t,
        delta_i,
    )
    dt_th_explicit = min(dt_th_core, dt_th_shell)
    dt_th = dt_th_explicit * core_shell_timemagnify
    dt_th_shell_only = dt_th_explicit * shell_only_timemagnify

    def exact_dt_to_total_time(base_dt, remaining_time):
        n_remaining = max(1, int(np.ceil(remaining_time / base_dt)))
        return float(remaining_time / n_remaining), int(n_remaining)

    _, nsteps_th_estimated = exact_dt_to_total_time(dt_th, total_time)
    max_th_steps = os.environ.get("THM_MAX_TH_STEPS")
    max_th_steps_limit = None
    if max_th_steps:
        max_th_steps_limit = max(1, int(max_th_steps))
    nsteps_th = min(nsteps_th_estimated, max_th_steps_limit) if max_th_steps_limit else nsteps_th_estimated

    T_increment = Tinit + (Tsurr - Tinit) / nsteps_th * 2
    T_phys = np.full(phys_coords_t.shape[0], Tinit)
    T_left = np.full(ghost_dict_t["left"].shape[0], Tinit)
    T_right = np.full(ghost_dict_t["right"].shape[0], Tsurr)
    T_top = np.full(ghost_dict_t["top"].shape[0], Tinit)
    T_bot = np.full(ghost_dict_t["bot"].shape[0], Tinit)
    T = np.concatenate([T_phys, T_left, T_right, T_top, T_bot])

    corr_t_to_m = pfc.shrink_Tth_by_matching_coords(coords_m, coords_t)
    corr_t_to_shell_only_m = pfc.shrink_Tth_by_matching_coords(coords_shell_only_m, coords_t)
    corr_shell_only_t_to_t = pfc.shrink_shell_only_by_matching_coords(coords_shell_only_t, coords_t)
    corr_shell_only_m_to_m = pfc.shrink_shell_only_by_matching_coords(coords_shell_only_m, coords_m)
    corr_shell_only_m_to_shell_only_t = pfc.shrink_Tth_by_matching_coords(
        coords_shell_only_m,
        coords_shell_only_t,
    )
    T_shell_only = T[corr_shell_only_t_to_t]

    """4. Definition of enthalpy for temperature field."""
    H = cf.get_enthalpy(
        T,
        mask_core_t,
        rho_s,
        rho_l,
        rho_shell,
        cs,
        cl,
        c_shell,
        L,
        Ts,
        Tl,
        rho_void,
        Cp_void,
    )

    print(
        f"[Temperature] dt_explicit={dt_th_explicit:.6e}, dt={dt_th:.6e}, "
        f"nsteps={nsteps_th}, right_boundary_pairs={ghost_idx_right_t.size}"
    )

    """5. Definition of Mechanical field."""

    """5.1 Definition of shell-core Mechanical field."""
    T_m = T[corr_t_to_m]
    edge_j = csr_indices_m
    crack = np.ones(nnz, dtype=np.int8)
    s0_shell = sigmat / E_shell
    s0_edge = np.full(nnz, s0_shell, dtype=np.float64)
    core_edge_mask = mask_core_m[edge_i] | mask_core_m[edge_j]
    s0_edge[core_edge_mask] = 1.0
    rho_node = np.where(mask_core_m, rho_s, rho_shell).astype(np.float64)
    dir_r_m, dir_z_m = pfc.compute_direction_edges_csr_numba(
        coords_m,
        edge_i,
        edge_j,
        csr_dist_m,
    )
    dx0_edge_m = dir_r_m * csr_dist_m
    dz0_edge_m = dir_z_m * csr_dist_m

    n_phys_m = phys_coords_m.shape[0]
    r_all_m = coords_m[:, 0]
    axis_mask = (r_all_m > r_start) & (r_all_m < r_start + dr_i - eps)
    r_flat = coords_m[:, 0].astype(np.float64)
    z_flat = coords_m[:, 1].astype(np.float64)
    shape_edge_m = cf.compute_shape_factor_edge_csr(
        r_flat,
        edge_i,
        csr_indices_m,
        mechanical=True,
    )
    damage_bond_mask = (
        (edge_j != edge_i)
        & (csr_area_m > 0.0)
        & (mask_core_m[edge_i] == mask_core_m[edge_j])
        & (~mask_core_m[edge_i])
        & (~mask_core_m[edge_j])
    )
    outer_surface_node_mask = geom.outer_surface_mask_from_coords(
        coords_m[:, :2],
        r,
        dr_i,
        dshell,
        r_start,
        r,
        modify_coordinates,
        coordinate_x_scale,
    )
    outer_surface_node_mask[n_phys_m:] = False

    coeff = 3.0 / (np.pi * delta_i ** 3)
    w_maxe_edge = np.maximum(np.abs(dir_r_m), np.abs(dir_z_m)).astype(np.float64)
    q = 4.0 / 3.0 * np.pi ** 2 * delta_i ** 4
    lambda_diag = ADR.compute_lambda_diag_matrix_axsy(
        csr_indptr_m,
        csr_indices_m,
        csr_area_m,
        csr_dist_m,
        w_maxe_edge,
        shape_edge_m,
        lamda_edge,
        miu_edge,
        r_flat,
        delta_i,
        q,
    )

    Ur = np.zeros(coords_m.shape[0])
    Uz = np.zeros(coords_m.shape[0])
    Ar = np.zeros(coords_m.shape[0])
    Az = np.zeros(coords_m.shape[0])
    br = np.zeros(coords_m.shape[0])
    bz = np.zeros(coords_m.shape[0])
    Fr = np.zeros_like(Ur)
    Fz = np.zeros_like(Uz)
    Fr_0 = np.zeros_like(Ur)
    Fz_0 = np.zeros_like(Uz)
    Vr_half = np.zeros_like(Ur)
    Vz_half = np.zeros_like(Uz)
    damage_phi = np.zeros_like(Ur)

    print(
        f"[Mechanical shell-core] field initialized, nodes={coords_m.shape[0]}, "
        f"damage_bonds={np.count_nonzero(damage_bond_mask)}, "
        f"outer_surface_nodes={np.count_nonzero(outer_surface_node_mask)}"
    )

    """5.2 Definition of shell-only Mechanical field."""
    indptr_shell_only = csr_indptr_shell_only_m
    indices_shell_only = csr_indices_shell_only_m
    area_shell_only = csr_area_shell_only_m
    N_shell_only = len(indptr_shell_only) - 1
    nnz_shell_only = len(indices_shell_only)
    edge_i_shell_only = np.repeat(np.arange(N_shell_only, dtype=np.int64), np.diff(indptr_shell_only))
    edge_j_shell_only = indices_shell_only

    mask_self_shell_only = edge_j_shell_only == edge_i_shell_only
    if np.any(mask_self_shell_only):
        csr_area_shell_only_m = csr_area_shell_only_m.copy()
        area_shell_only = csr_area_shell_only_m
        area_shell_only[mask_self_shell_only] = 0.0

    actual_area_node_shell_only = np.bincount(
        edge_i_shell_only,
        weights=area_shell_only,
        minlength=N_shell_only,
    ).astype(np.float64)
    theoretical_area_node_shell_only = np.full(N_shell_only, np.pi * delta_i ** 2, dtype=np.float64)
    numerator_shell_only = (
        theoretical_area_node_shell_only[edge_i_shell_only]
        + theoretical_area_node_shell_only[edge_j_shell_only]
    )
    denom_shell_only = (
        actual_area_node_shell_only[edge_i_shell_only]
        + actual_area_node_shell_only[edge_j_shell_only]
    )
    csr_shell_only = np.divide(
        numerator_shell_only,
        denom_shell_only,
        out=np.zeros_like(denom_shell_only, dtype=np.float64),
        where=np.abs(denom_shell_only) > eps,
    )

    mask_core_shell_only_m = np.zeros(N_shell_only, dtype=bool)
    L_total_edge_shell_only = csr_dist_shell_only_m.copy()
    L_core_edge_shell_only = np.zeros_like(L_total_edge_shell_only)
    L_shell_edge_shell_only = L_total_edge_shell_only.copy()

    miu_shell = E_shell / (2.0 * (1.0 + nu_shell) + eps)
    lamda0_shell = (E_shell * nu_shell) / ((1.0 - 2.0 * nu_shell) * (1.0 + nu_shell) + eps)
    lamda_shell = lamda0_shell - miu_shell
    mechanical_alpha_shell = alpha_shell if enable_mechanical_thermal_strain else 0.0
    kprime_shell = (E_shell / (1.0 - 2.0 * nu_shell + eps)) * mechanical_alpha_shell

    E_node_shell_only = np.full(N_shell_only, E_shell, dtype=np.float64)
    nu_node_shell_only = np.full(N_shell_only, nu_shell, dtype=np.float64)
    alpha_node_shell_only = np.full(N_shell_only, mechanical_alpha_shell, dtype=np.float64)
    lamda_node_shell_only = np.full(N_shell_only, lamda_shell, dtype=np.float64)
    miu_node_shell_only = np.full(N_shell_only, miu_shell, dtype=np.float64)
    kprime_node_shell_only = np.full(N_shell_only, kprime_shell, dtype=np.float64)
    E_edge_shell_only = np.full(nnz_shell_only, E_shell, dtype=np.float64)
    nu_edge_shell_only = np.full(nnz_shell_only, nu_shell, dtype=np.float64)
    alpha_edge_shell_only = np.full(nnz_shell_only, mechanical_alpha_shell, dtype=np.float64)
    lamda_edge_shell_only = np.full(nnz_shell_only, lamda_shell, dtype=np.float64)
    miu_edge_shell_only = np.full(nnz_shell_only, miu_shell, dtype=np.float64)
    kprime_edge_shell_only = np.full(nnz_shell_only, kprime_shell, dtype=np.float64)

    T_shell_only_m = T_shell_only[corr_shell_only_m_to_shell_only_t]
    crack_shell_only = np.ones(nnz_shell_only, dtype=np.int8)
    s0_edge_shell_only = np.full(nnz_shell_only, s0_shell, dtype=np.float64)
    rho_node_shell_only = np.full(N_shell_only, rho_shell, dtype=np.float64)
    dir_r_shell_only_m, dir_z_shell_only_m = pfc.compute_direction_edges_csr_numba(
        coords_shell_only_m,
        edge_i_shell_only,
        edge_j_shell_only,
        csr_dist_shell_only_m,
    )
    dx0_edge_shell_only_m = dir_r_shell_only_m * csr_dist_shell_only_m
    dz0_edge_shell_only_m = dir_z_shell_only_m * csr_dist_shell_only_m

    n_phys_shell_only_m = phys_coords_shell_only_m.shape[0]
    surface_info_shell_only = pfc.find_inner_surface_layer(
        phys_coords_shell_only_m,
        r,
        dshell,
        dr_i,
        center_r=r_start,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
    )
    inner_surface_arc_correction_shell_only = pfc.compute_inner_surface_arc_length_correction(
        phys_coords_shell_only_m,
        surface_info_shell_only["indices"],
        r_core,
        dr_i,
        r_start,
        r,
        surface_info_shell_only["unit_outward"],
        -0.5 * np.pi,
        0.5 * np.pi,
        "angular",
        r,
        modify_coordinates,
        coordinate_x_scale,
    )
    idx_inner_shell_only = surface_info_shell_only["indices"]
    inner_surface_node_mask_core_shell = np.zeros(coords_m.shape[0], dtype=bool)
    inner_surface_core_shell_normal = np.zeros((coords_m.shape[0], 2), dtype=np.float64)
    inner_surface_core_shell_arc_length = np.zeros(coords_m.shape[0], dtype=np.float64)
    inner_surface_core_shell_indices = np.empty(0, dtype=np.int64)
    inner_surface_core_shell_match_max_distance = 0.0
    inner_surface_core_shell_match_bad_count = 0
    if idx_inner_shell_only.size:
        candidate_core_shell_indices = corr_shell_only_m_to_m[idx_inner_shell_only].astype(np.int64)
        shell_inner_coords = phys_coords_shell_only_m[idx_inner_shell_only, :2]
        matched_core_shell_coords = coords_m[candidate_core_shell_indices, :2]
        match_dist = np.sqrt(np.sum((matched_core_shell_coords - shell_inner_coords) ** 2, axis=1))
        inner_surface_core_shell_match_max_distance = float(np.max(match_dist))
        match_tol = max(0.51 * dr_i, 10.0 * tolerance)
        valid_match = (
            np.isfinite(match_dist)
            & (match_dist <= match_tol)
            & (~mask_core_m[candidate_core_shell_indices])
            & (candidate_core_shell_indices < n_phys_m)
        )
        inner_surface_core_shell_match_bad_count = int(np.count_nonzero(~valid_match))
        for match_pos in np.where(valid_match)[0]:
            node_idx = int(candidate_core_shell_indices[match_pos])
            arc_len = float(inner_surface_arc_correction_shell_only["arc_length"][match_pos])
            normal_vec = surface_info_shell_only["unit_outward"][match_pos]
            if not np.isfinite(arc_len) or arc_len <= eps:
                continue
            inner_surface_core_shell_arc_length[node_idx] += arc_len
            inner_surface_core_shell_normal[node_idx, 0] += arc_len * normal_vec[0]
            inner_surface_core_shell_normal[node_idx, 1] += arc_len * normal_vec[1]

        mapped_inner_nodes = np.where(inner_surface_core_shell_arc_length > eps)[0]
        for node_idx in mapped_inner_nodes:
            normal_norm = np.sqrt(
                inner_surface_core_shell_normal[node_idx, 0] ** 2
                + inner_surface_core_shell_normal[node_idx, 1] ** 2
            )
            if normal_norm > eps:
                inner_surface_core_shell_normal[node_idx, 0] /= normal_norm
                inner_surface_core_shell_normal[node_idx, 1] /= normal_norm
            else:
                inner_surface_core_shell_arc_length[node_idx] = 0.0

        inner_surface_core_shell_indices = np.where(
            inner_surface_core_shell_arc_length > eps
        )[0].astype(np.int64)
        inner_surface_node_mask_core_shell[inner_surface_core_shell_indices] = True

    inner_surface_shell_core_bond_mask = (
        inner_surface_node_mask_core_shell[edge_i]
        & (~mask_core_m[edge_i])
        & mask_core_m[edge_j]
        & (edge_i != edge_j)
        & (csr_area_m > eps)
    )
    inner_surface_shell_core_bond_count = int(np.count_nonzero(inner_surface_shell_core_bond_mask))
    unit_transfer_pressure_br_shell_only, unit_transfer_pressure_bz_shell_only = (
        sof.build_inner_pressure_body_force(
            coords_shell_only_m,
            surface_info_shell_only,
            inner_surface_arc_correction_shell_only,
            1.0,
            dr_i,
            nu_shell,
        )
    )

    r_all_shell_only_m = coords_shell_only_m[:, 0]
    axis_mask_shell_only = (r_all_shell_only_m > r_start) & (r_all_shell_only_m < r_start + dr_i - eps)
    r_flat_shell_only = coords_shell_only_m[:, 0].astype(np.float64)
    z_flat_shell_only = coords_shell_only_m[:, 1].astype(np.float64)
    shape_edge_shell_only_m = cf.compute_shape_factor_edge_csr(
        r_flat_shell_only,
        edge_i_shell_only,
        edge_j_shell_only,
        mechanical=True,
    )
    damage_bond_mask_shell_only = (
        (edge_j_shell_only != edge_i_shell_only)
        & (csr_area_shell_only_m > 0.0)
    )
    outer_surface_node_mask_shell_only = geom.outer_surface_mask_from_coords(
        coords_shell_only_m[:, :2],
        r,
        dr_i,
        dshell,
        r_start,
        r,
        modify_coordinates,
        coordinate_x_scale,
    )
    outer_surface_node_mask_shell_only[n_phys_shell_only_m:] = False

    skip_plots = os.environ.get("THM_SKIP_PLOTS", "0") == "1"
    show_shell_inner_surface_plot = (not skip_plots) and graph_shell_inner_surface_points
    save_shell_inner_surface_plot = os.environ.get("THM_SAVE_SHELL_INNER_SURFACE_PLOT", "0") != "0"
    sof.plot_shell_only_inner_surface_points(
        phys_coords_shell_only_m,
        surface_info_shell_only,
        outer_surface_node_mask_shell_only,
        r_start,
        r,
        outer_axis_x,
        outer_axis_z,
        inner_axis_x,
        inner_axis_z,
        show_shell_inner_surface_plot,
        save_shell_inner_surface_plot,
        os.environ.get("THM_SHELL_INNER_SURFACE_PLOT_PATH"),
    )

    w_maxe_edge_shell_only = np.maximum(
        np.abs(dir_r_shell_only_m),
        np.abs(dir_z_shell_only_m),
    ).astype(np.float64)
    lambda_diag_shell_only = ADR.compute_lambda_diag_matrix_axsy(
        indptr_shell_only,
        indices_shell_only,
        csr_area_shell_only_m,
        csr_dist_shell_only_m,
        w_maxe_edge_shell_only,
        shape_edge_shell_only_m,
        lamda_edge_shell_only,
        miu_edge_shell_only,
        r_flat_shell_only,
        delta_i,
        q,
    )

    Ur_shell_only = np.zeros(coords_shell_only_m.shape[0])
    Uz_shell_only = np.zeros(coords_shell_only_m.shape[0])
    Ar_shell_only = np.zeros(coords_shell_only_m.shape[0])
    Az_shell_only = np.zeros(coords_shell_only_m.shape[0])
    br_shell_only = np.zeros(coords_shell_only_m.shape[0])
    bz_shell_only = np.zeros(coords_shell_only_m.shape[0])
    Fr_shell_only = np.zeros_like(Ur_shell_only)
    Fz_shell_only = np.zeros_like(Uz_shell_only)
    Fr_0_shell_only = np.zeros_like(Ur_shell_only)
    Fz_0_shell_only = np.zeros_like(Uz_shell_only)
    Vr_half_shell_only = np.zeros_like(Ur_shell_only)
    Vz_half_shell_only = np.zeros_like(Uz_shell_only)
    damage_phi_shell_only = np.zeros_like(Ur_shell_only)
    T_shell_reference_m = np.full_like(T_shell_only_m, Tpre_avg)
    shell_transfer_br_shell_only = np.zeros_like(Ur_shell_only)
    shell_transfer_bz_shell_only = np.zeros_like(Uz_shell_only)

    print(
        f"[Mechanical shell-only] field initialized, nodes={coords_shell_only_m.shape[0]}, "
        f"inner_points={surface_info_shell_only['indices'].size}, "
        f"core_shell_inner_nodes={inner_surface_core_shell_indices.size}, "
        f"core_shell_inner_bonds={inner_surface_shell_core_bond_count}, "
        f"inner_match_max_dist={inner_surface_core_shell_match_max_distance:.3e}, "
        f"inner_match_bad={inner_surface_core_shell_match_bad_count}, "
        f"damage_bonds={np.count_nonzero(damage_bond_mask_shell_only)}, "
        f"outer_surface_nodes={np.count_nonzero(outer_surface_node_mask_shell_only)}"
    )

    # ------------------------
    # Time-stepping loop
    # ------------------------
    p = 0
    print_interval = max(1, int(10 / dt_th))
    print(f"Estimated thermal steps before phase switch: {nsteps_th}")
    print(
        f"core-shell thermal: factor={core_shell_timemagnify}, "
        f"dt={dt_th}, explicit base={dt_th_explicit}"
    )
    print(
        f"shell-only thermal: factor={shell_only_timemagnify}, "
        f"dt={dt_th_shell_only}"
    )

    start_time = time.time()

    rho_void_old, Cp_void_old = rho_air, cair
    rho_void_new, Cp_void_new = rho_air, cair

    shell_transfer_body_force_initialized = False
    shell_transfer_max_body_force = 0.0
    shell_transfer_match_info = None
    shell_transfer_br = np.zeros_like(Ur)
    shell_transfer_bz = np.zeros_like(Uz)
    shell_transfer_retained_pressure = 0.0
    shell_transfer_reference_point_pressure = None
    shell_transfer_reference_point_details = None
    pressure_ramp_counter = 0
    phase_pressure_applied = 0.0
    stop_time_loop = False
    stop_reason = None
    pressure_sample_point_label = "(rshell, r)"
    pressure_sample_point_target = np.array(
        [
            float(os.environ.get("THM_PRESSURE_POINT_R", str(r_start + inner_axis_x))),
            float(os.environ.get("THM_PRESSURE_POINT_Z", str(r))),
        ],
        dtype=np.float64,
    )
    pressure_local_source = "lame_displacement_before_shell_only_conversion"

    point_projection_kwargs = {
        "center_r": r_start,
        "center_z": r,
        "modify_coordinates": modify_coordinates,
        "inner_axis_x": inner_axis_x,
        "inner_axis_z": inner_axis_z,
        "outer_axis_x": outer_axis_x,
        "outer_axis_z": outer_axis_z,
    }
    state_displacement_kwargs = {
        **point_projection_kwargs,
        "tolerance": tolerance,
    }

    target_points = [
        ("(rshell, r)", r_start + inner_axis_x, r),
        ("(r, r)", r_start + outer_axis_x, r),
        ("(shell-mid, r)", r_start + 0.5 * (inner_axis_x + outer_axis_x), r),
        ("(0, 2r-dshell)", r_start, 2.0 * r - dshell),
    ]
    shell_phys_mask = ~mask_core_m[:phys_coords_m.shape[0]]
    tracked_points = plot.build_tracked_points(
        phys_coords_m,
        target_points,
        shell_phys_mask=shell_phys_mask,
        shell_labels=("(0, 2r-dshell)",),
        center_z=r,
        tolerance=tolerance,
        verbose=False,
    )
    shell_switch_ur_reference = next(
        (pt for pt in tracked_points if pt["label"] == "(rshell, r)"),
        None,
    )
    shell_switch_uz_reference = next(
        (pt for pt in tracked_points if pt["label"] == "(0, 2r-dshell)"),
        None,
    )

    """7，温度场-物理场更新"""
    time_history = []
    mechanical_mode_history = []
    mechanical_step_history = []
    mechanical_rms_history = []
    pressure_history = []
    pressure_formula_history = []
    pressure_load_factor_history = []
    pressure_active_history = []
    pressure_integral_history = []
    pressure_volume_formula_history = []
    pressure_applied_history = []
    pressure_source_history = []
    pressure_point_detail_history = []
    pressure_volume_detail_history = []
    core_temp_max_history = []
    core_temp_avg_history = []
    core_temp_min_history = []
    shell_temp_max_history = []
    thermal_dt_history = []
    thermal_stepper_history = []
    thermal_stage_history = []
    ur_histories = {pt["label"]: [] for pt in tracked_points}
    uz_histories = {pt["label"]: [] for pt in tracked_points}
    T_histories = {pt["label"]: [] for pt in tracked_points}

    n_phys_t = T_phys.size
    n_left_t = T_left.size
    n_right_t = T_right.size
    n_top_t = T_top.size
    n_bot_t = T_bot.size
    sl_phys_t = slice(0, n_phys_t)
    sl_left_t = slice(sl_phys_t.stop, sl_phys_t.stop + n_left_t)
    sl_right_t = slice(sl_left_t.stop, sl_left_t.stop + n_right_t)
    sl_top_t = slice(sl_right_t.stop, sl_right_t.stop + n_top_t)
    sl_bot_t = slice(sl_top_t.stop, sl_top_t.stop + n_bot_t)
    mask_core_phy_t = mask_core_t[sl_phys_t]
    cell_volume_node_t = 2.0 * np.pi * phys_coords_t[:, 0] * dr_i * dr_i
    core_volume_reference = max(float(np.sum(cell_volume_node_t[mask_core_phy_t])), 1.0e-30)

    def compute_pressure_volume_details_for_state(T_phys_state, T_shell_only_m_state):
        idx_inner = surface_info_shell_only["indices"]
        if idx_inner.size:
            T_inner_avg = float(np.mean(T_shell_only_m_state[idx_inner]))
        else:
            T_inner_avg = Tpre_avg
        deltaT_inner = T_inner_avg - Tpre_avg
        details = sof.compute_fixed_phase_pressure_volume_details(
            T_phys_state,
            mask_core_phy_t,
            phys_coords_t,
            dr_i,
            core_volume_reference,
            deltaT_inner,
            Tpre_avg,
            phase_volume_expansion_fraction,
            alpha_core_s,
            alpha_core_l,
            Ts,
            Tl,
            E_shell,
            nu_shell,
            r,
            r_core,
            alpha_shell,
            rho_s,
            rho_l,
            comp_l,
        )
        p_formula = float(details.get("pressure_formula", details.get("pressure", 0.0)))
        p_calculated_raw = float(details.get("pressure", p_formula))
        if not np.isfinite(p_formula):
            p_formula = 0.0
        if not np.isfinite(p_calculated_raw):
            p_calculated_raw = 0.0
        p_calculated = p_calculated_raw
        details["pressure_formula"] = p_formula
        details["pressure_calculated_raw"] = p_calculated_raw
        details["pressure_calculated"] = p_calculated
        details["pressure_phase_target"] = p_calculated
        details["pressure_activation"] = "core_temp_max_ge_Ts"
        details["pressure_ramp_steps"] = int(pressure_ramp_steps)
        return details

    def finite_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def compute_pressure_sample_lame_for_current_core_shell_state():
        idx_inner_pressure = surface_info_shell_only["indices"]
        if idx_inner_pressure.size:
            shell_deltaT_pressure = float(
                np.mean(T_shell_only_m[idx_inner_pressure]) - Tpre_avg
            )
        else:
            shell_deltaT_pressure = 0.0

        r_node_pressure = coords_m[:, 0]
        z_node_pressure = coords_m[:, 1]
        candidate_mask = (
            inner_surface_node_mask_core_shell
            & (~mask_core_m)
            & np.isfinite(inner_surface_core_shell_arc_length)
            & (inner_surface_core_shell_arc_length > 1.0e-30)
        )
        candidates = np.where(candidate_mask)[0]
        if candidates.size == 0:
            pressure_value = 0.0
            pressure_info = {
                "pressure": 0.0,
                "source": "lame_displacement",
                "pressure_model": "lame_displacement",
                "valid": False,
                "reason": "no_inner_surface_candidates",
            }
        else:
            dist2 = (
                (r_node_pressure[candidates] - float(pressure_sample_point_target[0])) ** 2
                + (z_node_pressure[candidates] - float(pressure_sample_point_target[1])) ** 2
            )
            best_pos = int(np.argmin(dist2))
            selected_node = int(candidates[best_pos])
            target_distance = float(np.sqrt(dist2[best_pos]))
            selected_r = float(r_node_pressure[selected_node])
            selected_z = float(z_node_pressure[selected_node])
            arc_i = float(inner_surface_core_shell_arc_length[selected_node])
            normal_i = inner_surface_core_shell_normal[selected_node]
            normal_len = float(np.sqrt(normal_i[0] * normal_i[0] + normal_i[1] * normal_i[1]))
            if (
                    not np.isfinite(arc_i)
                    or arc_i <= 1.0e-30
                    or not np.isfinite(normal_len)
                    or normal_len <= 1.0e-30
            ):
                pressure_value = 0.0
                pressure_info = {
                    "pressure": 0.0,
                    "source": "lame_displacement",
                    "pressure_model": "lame_displacement",
                    "valid": False,
                    "reason": "invalid_selected_surface_geometry",
                    "selected_node_index": selected_node,
                    "selected_node_coord": [selected_r, selected_z],
                    "target_distance": target_distance,
                }
            else:
                nr_i = float(normal_i[0] / normal_len)
                nz_i = float(normal_i[1] / normal_len)
                r_face = max(selected_r - 0.5 * dr_i * nr_i, 0.0)
                represented_area = 2.0 * np.pi * r_face * arc_i
                cell_volume_i = 2.0 * np.pi * max(selected_r, 0.0) * dr_i * dr_i
                center_r_value = float(r_start)
                center_z_value = float(r)
                radial_vector_r = selected_r - center_r_value
                radial_vector_z = selected_z - center_z_value
                radial_position_raw = float(
                    np.sqrt(radial_vector_r * radial_vector_r + radial_vector_z * radial_vector_z)
                )
                if radial_position_raw > 1.0e-30:
                    radial_unit_r = radial_vector_r / radial_position_raw
                    radial_unit_z = radial_vector_z / radial_position_raw
                else:
                    radial_unit_r = nr_i
                    radial_unit_z = nz_i
                radial_position = min(
                    max(radial_position_raw, float(inner_axis_x)),
                    float(outer_axis_x),
                )
                radial_position_clamped = bool(
                    abs(radial_position - radial_position_raw)
                    > 1.0e-12 * max(1.0, abs(radial_position_raw))
                )
                selected_uz = float(Uz[selected_node])
                radial_displacement = float(
                    Ur[selected_node] * radial_unit_r
                    + selected_uz * radial_unit_z
                )
                surface_normal_displacement = float(
                    Ur[selected_node] * nr_i
                    + selected_uz * nz_i
                )
                pressure_value = sof.lame_internal_pressure_from_shell_radial_displacement(
                    E_shell,
                    nu_shell,
                    outer_axis_x,
                    inner_axis_x,
                    radial_position,
                    radial_displacement,
                    alpha=alpha_shell,
                    deltaT=shell_deltaT_pressure,
                )
                shell_span = float(outer_axis_x) ** 3 - float(inner_axis_x) ** 3
                compliance = (
                    (1.0 - 2.0 * float(nu_shell))
                    * float(inner_axis_x) ** 3
                    * radial_position
                    + 0.5
                    * (1.0 + float(nu_shell))
                    * float(inner_axis_x) ** 3
                    * float(outer_axis_x) ** 3
                    / (radial_position ** 2)
                ) / (float(E_shell) * shell_span)
                thermal_displacement = float(alpha_shell) * shell_deltaT_pressure * radial_position
                pressure_info = {
                    "pressure": float(pressure_value),
                    "source": "lame_displacement",
                    "pressure_model": "lame_displacement",
                    "valid": bool(np.isfinite(pressure_value)),
                    "target_point": [
                        float(pressure_sample_point_target[0]),
                        float(pressure_sample_point_target[1]),
                    ],
                    "target_distance": target_distance,
                    "selected_node_index": selected_node,
                    "selected_node_coord": [selected_r, selected_z],
                    "surface_normal": [float(nr_i), float(nz_i)],
                    "lame_radial_unit": [float(radial_unit_r), float(radial_unit_z)],
                    "surface_arc_length": float(arc_i),
                    "represented_area": float(represented_area),
                    "cell_volume": float(cell_volume_i),
                    "radial_position": float(radial_position),
                    "radial_position_raw": float(radial_position_raw),
                    "radial_position_clamped_to_shell": radial_position_clamped,
                    "inner_radius": float(inner_axis_x),
                    "outer_radius": float(outer_axis_x),
                    "center_r": float(center_r_value),
                    "center_z": float(center_z_value),
                    "radial_displacement": float(radial_displacement),
                    "surface_normal_displacement": float(surface_normal_displacement),
                    "uz_supplied": True,
                    "thermal_displacement": float(thermal_displacement),
                    "mechanical_displacement": float(radial_displacement - thermal_displacement),
                    "lame_displacement_compliance": float(compliance),
                    "E_shell": float(E_shell),
                    "nu_shell": float(nu_shell),
                    "alpha_shell": float(alpha_shell),
                    "shell_deltaT": float(shell_deltaT_pressure),
                    "pressure_sign": "positive_core_to_shell",
                    "area_model": "not_used_lame_point_displacement",
                    "direction_domain": "not_used_lame_point_displacement",
                    "bond_force_model": "not_used_lame_point_displacement",
                    "analytic_assumptions": (
                        "linear_elastic_isotropic_spherical_shell_uniform_equivalent_internal_pressure"
                    ),
                }
        pressure_raw = float(pressure_value)
        if not np.isfinite(pressure_raw):
            pressure_value = 0.0
            pressure_info["valid"] = False
            pressure_info["reason"] = "non_finite_pressure"
            pressure_raw = 0.0
        else:
            pressure_value = pressure_raw
        pressure_info["pressure_raw"] = pressure_raw
        pressure_info["pressure_clamped_positive"] = max(0.0, pressure_raw)
        pressure_info["pressure"] = float(pressure_value)
        pressure_info["label"] = pressure_sample_point_label
        pressure_info["target_r"] = float(pressure_sample_point_target[0])
        pressure_info["target_z"] = float(pressure_sample_point_target[1])
        pressure_info["source"] = pressure_local_source
        pressure_info["time_scope"] = "before_shell_only_conversion"
        return float(pressure_value), pressure_info

    initial_pressure_details = compute_pressure_volume_details_for_state(
        T[sl_phys_t],
        T_shell_only_m,
    )
    initial_integral_pressure, initial_pressure_point_details = (
        compute_pressure_sample_lame_for_current_core_shell_state()
    )
    initial_volume_formula_pressure = float(initial_pressure_details["pressure_calculated"])
    initial_pressure_details["pressure"] = float(initial_integral_pressure)
    initial_pressure_details["pressure_source"] = pressure_local_source
    initial_pressure_details["pressure_integral_at_point"] = float(initial_integral_pressure)
    initial_pressure_details["pressure_volume_formula"] = float(initial_volume_formula_pressure)
    initial_pressure_details["pressure_applied"] = 0.0
    initial_pressure_details["pressure_phase"] = 0.0
    initial_pressure_details["pressure_phase_factor"] = 0.0
    initial_pressure_details["pressure_phase_ramp_mode"] = "not_started"
    initial_pressure_details["pressure_load_factor"] = 0.0
    initial_pressure_details["pressure_ramp_completed_steps"] = 0
    initial_pressure_details["pressure_transfer"] = 0.0
    initial_pressure_details["pressure_transfer_applied"] = 0.0
    initial_pressure_details["pressure_active"] = False
    initial_pressure_details["mechanical_mode"] = "core_shell_initial"
    time_history.append(0.0)
    mechanical_mode_history.append("core_shell_initial")
    mechanical_step_history.append(0)
    mechanical_rms_history.append(0.0)
    pressure_history.append(float(initial_integral_pressure))
    pressure_formula_history.append(float(initial_volume_formula_pressure))
    pressure_load_factor_history.append(0.0)
    pressure_active_history.append(False)
    pressure_integral_history.append(float(initial_integral_pressure))
    pressure_volume_formula_history.append(float(initial_volume_formula_pressure))
    pressure_applied_history.append(0.0)
    pressure_source_history.append(pressure_local_source)
    pressure_point_detail_history.append(initial_pressure_point_details)
    pressure_volume_detail_history.append(initial_pressure_details)
    core_temp_min_history.append(float(Tinit))
    core_temp_max_history.append(float(Tinit))
    core_temp_avg_history.append(float(Tinit))
    shell_temp_max_history.append(float(Tinit))
    thermal_dt_history.append(0.0)
    thermal_stepper_history.append("initial")
    thermal_stage_history.append(0)
    pressure_sample_point_info = {
        "label": pressure_sample_point_label,
        "target_r": float(pressure_sample_point_target[0]),
        "target_z": float(pressure_sample_point_target[1]),
        "selected_node_index": initial_pressure_point_details.get("selected_node_index"),
        "selected_node_coord": initial_pressure_point_details.get("selected_node_coord"),
        "target_distance": initial_pressure_point_details.get("target_distance"),
        "source_before_shell_only": "lame_displacement",
        "source_after_shell_only": "volume_formula",
    }
    print(
        f"[Pressure point] {pressure_sample_point_label} target="
        f"({pressure_sample_point_target[0]:.6e}, {pressure_sample_point_target[1]:.6e}) m, "
        f"selected={pressure_sample_point_info['selected_node_coord']}, "
        f"distance={pressure_sample_point_info['target_distance']}"
    )
    for pt in tracked_points:
        i_local = pt["local_idx"]
        mirror_local_idx = pt["mirror_local_idx"]
        if i_local is None:
            current_ur = np.nan
            current_uz = np.nan
            current_T = np.nan
        elif pt.get("tracking_mode") == "midline_symmetric_average" and mirror_local_idx is not None:
            current_ur = 0.5 * (Ur[i_local] + Ur[mirror_local_idx])
            current_uz = 0.5 * (Uz[i_local] + Uz[mirror_local_idx])
            current_T = 0.5 * (T_m[i_local] + T_m[mirror_local_idx])
        else:
            current_ur = Ur[i_local]
            current_uz = Uz[i_local]
            current_T = T_m[i_local]

        ur_histories[pt["label"]].append(float(current_ur))
        uz_histories[pt["label"]].append(float(current_uz))
        T_histories[pt["label"]].append(float(current_T))

    current_time = 0.0
    for step1 in range(nsteps_th):
        remaining_time = total_time - current_time
        if remaining_time <= 1.0e-30:
            break

        dt_base_this = dt_th_shell_only if shell_transfer_body_force_initialized else dt_th
        thermal_dt_this, thermal_steps_remaining_this = exact_dt_to_total_time(
            dt_base_this,
            remaining_time,
        )

        if use_boundary_ramp:
            boundary_fraction = float(
                np.clip(2.0 * (current_time + thermal_dt_this) / total_time, 0.0, 1.0)
            )
            T_increment = Tinit + (Tsurr - Tinit) * boundary_fraction
        else:
            T_increment = Tsurr

        T_work = np.array(T, copy=True)
        T_pltb = np.concatenate([
            T_work[sl_phys_t],
            T_work[sl_left_t],
            T_work[sl_top_t],
            T_work[sl_bot_t],
        ])
        T_right_work = T_work[sl_right_t].copy()
        T_right_work[ghost_idx_right_t] = 2.0 * T_increment - T_pltb[phys_idx_right_t]
        T_work[sl_right_t] = T_right_work

        H_start = cf.get_enthalpy(
            T_work,mask_core_t,
            rho_s,rho_l,rho_shell,
            cs,cl,c_shell,
            L,
            Ts,Tl,
            rho_void,Cp_void,
        )

        K_data_t, diag_t = cf.build_Kdata_and_rowsum_csr_numba(
            T_work,
            mask_core_t,
            factor_data_t,
            csr_area_t,
            shape_factor_t,
            csr_dist_t, csr_indptr_t,csr_indices_t,
            ks,kl,
            Ts,Tl,
            k_shell,
            delta_i,
            k_void,
            thermal_dt_this,
        )
        dH = cf.apply_K_with_diag_csr_numba(
            csr_indptr_t,
            csr_indices_t,
            K_data_t,
            diag_t,
            T_work,
            rho_void_old,
            Cp_void_old,
            rho_void_new,
            Cp_void_new,
        )
        H = H_start + dH
        T = cf.temperature_from_enthalpy_numba(
            H,
            mask_core_t,
            rho_s,
            rho_l,
            cs,
            cl,
            L,
            Ts,
            Tl,
            rho_shell,
            c_shell,
            rho_void,
            Cp_void,
        )

        T_pltb = np.concatenate([
            T[sl_phys_t],
            T[sl_left_t],
            T[sl_top_t],
            T[sl_bot_t],
        ])
        T_right_work = T[sl_right_t].copy()
        T_right_work[ghost_idx_right_t] = 2.0 * T_increment - T_pltb[phys_idx_right_t]
        T[sl_right_t] = T_right_work

        if not np.all(np.isfinite(T)):
            raise FloatingPointError(f"[Thermal] non-finite temperature at step {step1}")

        T_phys_current = T[sl_phys_t]
        core_temp_min = np.nan
        core_temp_max = np.nan
        core_temp_avg = np.nan
        shell_temp_max = np.nan
        if np.any(mask_core_phy_t):
            T_core_phy = T_phys_current[mask_core_phy_t]
            core_temp_min = float(np.min(T_core_phy))
            core_temp_max = float(np.max(T_core_phy))
            core_temp_avg = float(np.mean(T_core_phy))
        shell_mask_phy_t = ~mask_core_phy_t
        if np.any(shell_mask_phy_t):
            shell_temp_max = float(np.max(T_phys_current[shell_mask_phy_t]))

        T_m = T[corr_t_to_m]
        T_shell_only = T[corr_shell_only_t_to_t]
        T_shell_only_m = T_shell_only[corr_shell_only_m_to_shell_only_t]

        if thermal_steps_remaining_this == 1:
            thermal_time_after_update = total_time
        else:
            thermal_time_after_update = current_time + thermal_dt_this
        if step1 > 0:
            print(
                f"[Temperature] field updated at step {step1 + 1}/{nsteps_th}, "
                f"t = {thermal_time_after_update:.12e} / {total_time:.12e} s, "
                f"T_boundary = {T_increment:.6f} K, "
                f"T_core_min = {core_temp_min:.6f} K, "
                f"T_core_max = {core_temp_max:.6f} K",
                flush=True,
            )

        rho_void_old = rho_void
        Cp_void_old = Cp_void
        rho_void = 0
        Cp_void = 0
        k_void = 0
        rho_void_new = rho_void
        Cp_void_new = Cp_void


        if step1 == 0:
            rho_node = np.where(mask_core_m, rho_s, rho_shell).astype(np.float64)
            rho_edge = 0.5 * (rho_node[edge_i] + rho_node[edge_j])
            alpha_core_node = np.full_like(T_m, alpha_core_s, dtype=np.float64)
        else:
            rho_node, rho_edge = cf.get_density(
                T_m,
                mask_core_m,
                rho_s,
                rho_l,
                Ts,
                Tl,
                rho_shell,
                rho_void,
                edge_i,
                edge_j,
            )
            liquid_frac = np.clip((T_m - Ts) / (Tl - Ts + 1.0e-30), 0.0, 1.0)
            alpha_core_node = alpha_core_s + (alpha_core_l - alpha_core_s) * liquid_frac

        alpha_node = np.where(mask_core_m, alpha_core_node, alpha_shell).astype(np.float64)
        kprime_node = (E_node / (1.0 - 2.0 * nu_node + eps)) * alpha_node
        alpha_edge = pfc.build_edge_property_harmonic_from_lengths(
            alpha_node,
            mask_core_m,
            edge_i,
            edge_j,
            L_core_edge,
            L_shell_edge,
            L_total_edge,
        )
        kprime_edge = pfc.build_edge_property_harmonic_from_lengths(
            kprime_node,
            mask_core_m,
            edge_i,
            edge_j,
            L_core_edge,
            L_shell_edge,
            L_total_edge,
        )
        rho_node_shell_only.fill(rho_shell)

        mechanical_mode = "core_shell_initial"
        mechanical_converged_step = 0
        mechanical_rms = np.nan
        p = 0.0
        p_phase = 0.0
        just_initialized_shell_only = False
        phase_started = np.isfinite(core_temp_max) and core_temp_max >= Ts
        pressure_volume_details = compute_pressure_volume_details_for_state(
            T_phys_current,
            T_shell_only_m,
        )
        pressure_volume_formula = float(pressure_volume_details["pressure_calculated"])
        pressure_volume_details["pressure"] = 0.0
        pressure_volume_details["pressure_source"] = pressure_local_source
        pressure_volume_details["pressure_integral_at_point"] = None
        pressure_volume_details["pressure_volume_formula"] = float(pressure_volume_formula)
        pressure_volume_details["pressure_applied"] = 0.0
        pressure_volume_details["pressure_phase"] = 0.0
        pressure_volume_details["pressure_phase_factor"] = 0.0
        pressure_volume_details["pressure_phase_ramp_mode"] = "not_applied_before_shell_switch"
        pressure_volume_details["pressure_load_factor"] = 0.0
        pressure_volume_details["pressure_ramp_completed_steps"] = int(pressure_ramp_counter)
        pressure_volume_details["pressure_transfer"] = 0.0
        pressure_volume_details["pressure_transfer_applied"] = 0.0
        pressure_volume_details["pressure_active"] = bool(phase_started)
        pressure_volume_details["mechanical_mode"] = mechanical_mode

        if step1 == 0 and not phase_started:
            br.fill(0.0)
            bz.fill(0.0)

            dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr(
                r_flat,
                z_flat,
                Ur,
                Uz,
                edge_i,
                edge_j,
                csr_dist_m,
                csr_area_m,
                shape_edge_m,
                coeff,
                csr,
            )
            Ar, Az = mt.compute_accel_osbpd_axisym_csr_numba(
                csr_indptr_m,
                csr_indices_m,
                csr_area_m,
                eij_edge,
                n_r_edge,
                n_z_edge,
                lamda_edge,
                miu_edge,
                lamda_node,
                miu_node,
                coords_m[:, 0],
                Ur,
                dilation,
                rho_node,
                br,
                bz,
                delta_i,
                T_m,
                Tpre_avg,
                kprime_edge,
                kprime_node,
                0,
                csr,
            )

            Fr_0 = Ar * rho_node
            Fz_0 = Az * rho_node
            Vr_half = 0.5 * (Fr_0 / lambda_diag)
            Vz_half = 0.5 * (Fz_0 / lambda_diag)
            Ur = Ur + Vr_half * dt_ADR
            Uz = Uz + Vz_half * dt_ADR
            Ur[axis_mask] = 0.0
        else:
            use_core_shell_mechanics = (
                (not shell_transfer_body_force_initialized)
                and not phase_started
            )

            if use_core_shell_mechanics:
                mechanical_mode = "core_shell"
                br.fill(0.0)
                bz.fill(0.0)
                for step in range(nsteps_m):
                    Ur_prev = Ur.copy()
                    Uz_prev = Uz.copy()

                    damage_phi = mt.compute_damage_variable_axisym_csr(
                        csr_indptr_m,
                        csr_indices_m,
                        csr_area_m,
                        shape_edge_m,
                        csr,
                        crack,
                        damage_bond_mask,
                    )

                    Ur, Uz, Fr, Fz, Vr_half, Vz_half = mt.compute_mechanical_step_csr(
                        csr_indptr_m,csr_indices_m,csr_dist_m,csr_area_m,
                        r_flat,z_flat,
                        shape_edge_m,
                        coeff,
                        lamda_edge,
                        miu_edge,
                        crack,
                        s0_edge,
                        Ur,
                        Uz,
                        br,
                        bz,
                        Fr_0,
                        Fz_0,
                        Vr_half,
                        Vz_half,
                        lambda_diag,
                        dt_ADR,
                        delta_i,
                        T_m,
                        Tpre_avg,
                        kprime_edge,
                        alpha_edge,
                        csr,
                        enable_cracking=enable_cracking,
                        damage_node=damage_phi,
                        crack_pressure=0.0,
                        damage_threshold=damage_threshold,
                        enable_crack_pressure_force=False,
                        use_crack_in_dilation=False,
                    )
                    Fr_0 = Fr.copy()
                    Fz_0 = Fz.copy()
                    Ur[axis_mask] = 0.0

                    damage_phi = mt.compute_damage_variable_axisym_csr(
                        csr_indptr_m,
                        csr_indices_m,
                        csr_area_m,
                        shape_edge_m,
                        csr,
                        crack,
                        damage_bond_mask,
                    )
                    if mt.has_outer_surface_damage_node(
                        damage_phi,
                        outer_surface_node_mask,
                        damage_threshold,
                    ):
                        stop_time_loop = True
                        stop_reason = "outer_surface_damage"
                        mechanical_converged_step = step
                        print(
                            f"[Mechanical core-shell] stopped at step {step}: outer shell surface damage "
                            f"exceeded {damage_threshold:.3f}"
                        )
                        break

                    mechanical_rms = np.sqrt(np.mean((Ur - Ur_prev) ** 2 + (Uz - Uz_prev) ** 2))
                    mechanical_converged_step = step
                    if mechanical_rms < core_shell_rms:
                        print(
                            f"[Mechanical core-shell] converged at step {step} with RMS {mechanical_rms:.3e}, "
                            f"Tmax = {core_temp_max:.6f} K"
                        )
                        break

                    if step > 0 and step % 10 == 0:
                        print(f"[Mechanical core-shell] step {step}, RMS {mechanical_rms:.3e}")
            else:
                mechanical_mode = "shell_only"
                if not shell_transfer_body_force_initialized:
                    shell_transfer_reference_point_pressure, shell_transfer_reference_point_details = (
                        compute_pressure_sample_lame_for_current_core_shell_state()
                    )
                    dilation_transfer, eij_edge_transfer, n_r_edge_transfer, n_z_edge_transfer = (
                        pfc.compute_dilation_axisym_csr_rows_numba(
                            csr_indptr_m,
                            csr_indices_m,
                            r_flat,
                            z_flat,
                            Ur,
                            Uz,
                            csr_dist_m,
                            csr_area_m,
                            shape_edge_m,
                            coeff,
                            csr,
                        )
                    )
                    br_transfer, bz_transfer = sof.compute_core_shell_bond_force_to_shell_axisym_csr(
                        csr_indptr_m,
                        csr_indices_m,
                        csr_area_m,
                        shape_edge_m,
                        eij_edge_transfer,
                        n_r_edge_transfer,
                        n_z_edge_transfer,
                        crack,
                        lamda_edge,
                        miu_edge,
                        coords_m[:, 0],
                        coords_m[:, 1],
                        r_start,
                        r,
                        Ur,
                        dilation_transfer,
                        T_m,
                        Tpre_avg,
                        kprime_edge,
                        csr,
                        mask_core_m,
                        delta_i,
                        shell_source_node_mask=inner_surface_node_mask_core_shell,
                    )
                    br_transfer[mask_core_m] = 0.0
                    bz_transfer[mask_core_m] = 0.0
                    raw_shell_transfer_br = br_transfer[corr_shell_only_m_to_m].copy()
                    raw_shell_transfer_bz = bz_transfer[corr_shell_only_m_to_m].copy()
                    raw_transfer_pressure, raw_transfer_pressure_info = (
                        sof.project_transfer_force_to_uniform_pressure(
                            raw_shell_transfer_br,
                            raw_shell_transfer_bz,
                            unit_transfer_pressure_br_shell_only,
                            unit_transfer_pressure_bz_shell_only,
                            coords_all_m=coords_shell_only_m,
                            dr=dr_i,
                            center_r=r_start,
                            center_z=r,
                            inner_axis_x=inner_axis_x,
                            inner_axis_z=inner_axis_z,
                            n_phys=n_phys_shell_only_m,
                        )
                    )
                    if not np.isfinite(raw_transfer_pressure):
                        raw_transfer_pressure = 0.0
                    raw_surface_force_resultant_info = sof.integrate_axisym_body_force_resultant(
                        raw_shell_transfer_br,
                        raw_shell_transfer_bz,
                        coords_shell_only_m,
                        dr_i,
                        n_phys=n_phys_shell_only_m,
                    )
                    Ur_shell_start = Ur[corr_shell_only_m_to_m].copy()
                    Uz_shell_start = Uz[corr_shell_only_m_to_m].copy()
                    T_shell_reference_m = T_shell_only_m.copy()
                    T_shell_mech_m = Tpre_avg + (T_shell_only_m - T_shell_reference_m)

                    target_equiv_br, target_equiv_bz, target_equiv_info, target_equiv_crack = (
                        sof.build_displacement_target_equivalent_body_force(
                            Ur_shell_start,
                            Uz_shell_start,
                            csr_indptr_shell_only_m,
                            csr_indices_shell_only_m,
                            csr_dist_shell_only_m,
                            csr_area_shell_only_m,
                            r_flat_shell_only,
                            z_flat_shell_only,
                            shape_edge_shell_only_m,
                            coeff,
                            lamda_edge_shell_only,
                            miu_edge_shell_only,
                            crack_shell_only,
                            s0_edge_shell_only,
                            delta_i,
                            T_shell_mech_m,
                            Tpre_avg,
                            kprime_edge_shell_only,
                            alpha_edge_shell_only,
                            csr_shell_only,
                            axis_mask_shell_only=axis_mask_shell_only,
                            damage_bond_mask_shell_only=damage_bond_mask_shell_only,
                            damage_threshold=damage_threshold,
                            enable_cracking=enable_cracking,
                            use_crack_in_dilation=True,
                            n_phys=n_phys_shell_only_m,
                        )
                    )
                    matched_transfer_pressure, matched_transfer_pressure_info = (
                        sof.project_transfer_force_to_uniform_pressure(
                            target_equiv_br,
                            target_equiv_bz,
                            unit_transfer_pressure_br_shell_only,
                            unit_transfer_pressure_bz_shell_only,
                            coords_all_m=coords_shell_only_m,
                            dr=dr_i,
                            center_r=r_start,
                            center_z=r,
                            inner_axis_x=inner_axis_x,
                            inner_axis_z=inner_axis_z,
                            n_phys=n_phys_shell_only_m,
                        )
                    )
                    if not np.isfinite(matched_transfer_pressure):
                        matched_transfer_pressure = 0.0

                    shell_transfer_retained_pressure = float(
                        shell_transfer_reference_point_pressure
                    )
                    if not np.isfinite(shell_transfer_retained_pressure):
                        shell_transfer_retained_pressure = float(matched_transfer_pressure)
                    shell_transfer_br_shell_only = target_equiv_br.copy()
                    shell_transfer_bz_shell_only = target_equiv_bz.copy()
                    shell_transfer_br_shell_only = (
                        shell_transfer_br_shell_only
                        - shell_transfer_retained_pressure * unit_transfer_pressure_br_shell_only
                    )
                    shell_transfer_bz_shell_only = (
                        shell_transfer_bz_shell_only
                        - shell_transfer_retained_pressure * unit_transfer_pressure_bz_shell_only
                    )

                    transfer_only_br = (
                        shell_transfer_retained_pressure * unit_transfer_pressure_br_shell_only
                        + shell_transfer_br_shell_only
                    )
                    transfer_only_bz = (
                        shell_transfer_retained_pressure * unit_transfer_pressure_bz_shell_only
                        + shell_transfer_bz_shell_only
                    )
                    target_equiv_surface_force_resultant_info = sof.integrate_axisym_body_force_resultant(
                        transfer_only_br,
                        transfer_only_bz,
                        coords_shell_only_m,
                        dr_i,
                        n_phys=n_phys_shell_only_m,
                    )

                    initial_Ur_shell_only = Ur_shell_start.copy()
                    initial_Uz_shell_only = Uz_shell_start.copy()
                    initial_Ur_shell_only[axis_mask_shell_only] = 0.0
                    displacement_match_state = sof.solve_shell_body_force_equilibrium(
                        initial_Ur_shell_only,
                        initial_Uz_shell_only,
                        target_equiv_crack,
                        transfer_only_br,
                        transfer_only_bz,
                        csr_indptr_shell_only_m,
                        csr_indices_shell_only_m,
                        csr_dist_shell_only_m,
                        csr_area_shell_only_m,
                        r_flat_shell_only,
                        z_flat_shell_only,
                        shape_edge_shell_only_m,
                        coeff,
                        lamda_edge_shell_only,
                        miu_edge_shell_only,
                        s0_edge_shell_only,
                        lambda_diag_shell_only,
                        dt_ADR,
                        delta_i,
                        T_shell_mech_m,
                        Tpre_avg,
                        kprime_edge_shell_only,
                        alpha_edge_shell_only,
                        csr_shell_only,
                        axis_mask_shell_only,
                        damage_bond_mask_shell_only,
                        max(nsteps_m, shell_transfer_conversion_steps),
                        shell_only_rms,
                        enable_cracking,
                        damage_threshold,
                        crack_pressure=0.0,
                        enable_crack_pressure_force=False,
                        log_prefix="[Switch displacement target mech]",
                        min_steps=shell_transfer_conversion_steps,
                    )
                    Ur_shell_only = displacement_match_state["Ur"].copy()
                    Uz_shell_only = displacement_match_state["Uz"].copy()
                    Ur_shell_only[axis_mask_shell_only] = 0.0
                    Fr_0_shell_only = displacement_match_state["Fr_0"]
                    Fz_0_shell_only = displacement_match_state["Fz_0"]
                    Vr_half_shell_only = displacement_match_state["Vr_half"]
                    Vz_half_shell_only = displacement_match_state["Vz_half"]
                    crack_shell_only = displacement_match_state["crack"]
                    damage_phi_shell_only = mt.compute_damage_variable_axisym_csr(
                        csr_indptr_shell_only_m,
                        csr_indices_shell_only_m,
                        csr_area_shell_only_m,
                        shape_edge_shell_only_m,
                        csr_shell_only,
                        crack_shell_only,
                        damage_bond_mask_shell_only,
                    )
                    mechanical_converged_step = displacement_match_state["converged_step"]
                    mechanical_rms = displacement_match_state["rms"]

                    displacement_error = np.sqrt(
                        (Ur_shell_only[:n_phys_shell_only_m] - Ur_shell_start[:n_phys_shell_only_m]) ** 2
                        + (Uz_shell_only[:n_phys_shell_only_m] - Uz_shell_start[:n_phys_shell_only_m]) ** 2
                    )
                    target_displacement_mag = np.sqrt(
                        Ur_shell_start[:n_phys_shell_only_m] ** 2
                        + Uz_shell_start[:n_phys_shell_only_m] ** 2
                    )
                    displacement_error_rms = (
                        float(np.sqrt(np.mean(displacement_error ** 2)))
                        if displacement_error.size
                        else 0.0
                    )
                    displacement_error_max = (
                        float(np.max(displacement_error))
                        if displacement_error.size
                        else 0.0
                    )
                    target_displacement_rms = (
                        float(np.sqrt(np.mean(target_displacement_mag ** 2)))
                        if target_displacement_mag.size
                        else 0.0
                    )
                    max_relative_error = (
                        0.0
                        if displacement_error_max <= 1.0e-30 and target_displacement_rms <= 1.0e-30
                        else float(displacement_error_rms / max(target_displacement_rms, 1.0e-30))
                    )
                    target_ur, _ = mt.tracked_displacement_components(
                        shell_switch_ur_reference,
                        [Ur],
                        [Uz],
                    )
                    _, target_uz = mt.tracked_displacement_components(
                        shell_switch_uz_reference,
                        [Ur],
                        [Uz],
                    )
                    shell_match_state = {"phys_coords_list_m": [phys_coords_shell_only_m]}
                    final_ur, _, _ = mt.state_displacement_components_at_point(
                        shell_match_state,
                        [Ur_shell_only],
                        [Uz_shell_only],
                        shell_switch_ur_reference,
                        **state_displacement_kwargs,
                    )
                    _, final_uz, _ = mt.state_displacement_components_at_point(
                        shell_match_state,
                        [Ur_shell_only],
                        [Uz_shell_only],
                        shell_switch_uz_reference,
                        **state_displacement_kwargs,
                    )
                    initial_ur = float(target_ur)
                    initial_uz = float(target_uz)
                    displacement_match_steps_done = int(mechanical_converged_step + 1)

                    shell_transfer_match_info = {
                        "iterations": int(displacement_match_steps_done),
                        "actual_correction_iterations": 0,
                        "match_mode": "displacement_target_equivalent_body_force",
                        "displacement_split": "none_total_displacement_solved_directly",
                        "target_ur": float(target_ur),
                        "target_uz": float(target_uz),
                        "initial_ur": float(initial_ur),
                        "initial_uz": float(initial_uz),
                        "final_ur": float(final_ur),
                        "final_uz": float(final_uz),
                        "final_error_ur": float(target_ur - final_ur),
                        "final_error_uz": float(target_uz - final_uz),
                        "best_match_error_norm": float(displacement_error_rms),
                        "best_max_relative_error": float(max_relative_error),
                        "match_relative_tolerance": float(shell_transfer_match_rel_tol),
                        "displacement_error_rms": float(displacement_error_rms),
                        "displacement_error_max": float(displacement_error_max),
                        "target_displacement_rms": float(target_displacement_rms),
                        "solved_ur": float(final_ur),
                        "solved_uz": float(final_uz),
                        "solved_displacement_rms": float(target_displacement_rms),
                        "target_equivalent_body_force_info": target_equiv_info,
                        "temperature_reference_mode": "shell_only_increment_from_switch_temperature",
                        "temperature_reference_min": float(
                            np.min(T_shell_reference_m[:n_phys_shell_only_m])
                            if n_phys_shell_only_m > 0
                            else Tpre_avg
                        ),
                        "temperature_reference_max": float(
                            np.max(T_shell_reference_m[:n_phys_shell_only_m])
                            if n_phys_shell_only_m > 0
                            else Tpre_avg
                        ),
                        "temperature_reference_avg": float(
                            np.mean(T_shell_reference_m[:n_phys_shell_only_m])
                            if n_phys_shell_only_m > 0
                            else Tpre_avg
                        ),
                    }
                    shell_transfer_match_info["transfer_method"] = (
                        "displacement_target_equivalent_body_force"
                    )
                    shell_transfer_match_info["transfer_pressure_decomposition"] = (
                        "uniform_pressure_plus_residual_body_force"
                    )
                    shell_transfer_match_info["equivalent_body_force_kept_fixed"] = True
                    shell_transfer_match_info["pressure_matching_enforced"] = False
                    shell_transfer_match_info["raw_surface_force_resultant"] = (
                        raw_surface_force_resultant_info
                    )
                    shell_transfer_match_info["equivalent_body_force_info"] = target_equiv_info
                    shell_transfer_match_info["equivalent_body_force_resultant"] = (
                        target_equiv_surface_force_resultant_info
                    )
                    shell_transfer_match_info["raw_transfer_pressure"] = float(raw_transfer_pressure)
                    shell_transfer_match_info["raw_transfer_pressure_info"] = raw_transfer_pressure_info
                    shell_transfer_match_info["matched_transfer_pressure"] = float(matched_transfer_pressure)
                    shell_transfer_match_info["matched_transfer_pressure_info"] = matched_transfer_pressure_info
                    shell_transfer_match_info["transfer_pressure_source"] = (
                        "reference_point_lame_displacement"
                    )
                    shell_transfer_match_info["reference_point_pressure"] = float(
                        shell_transfer_reference_point_pressure
                    )
                    shell_transfer_match_info["reference_point_pressure_info"] = (
                        shell_transfer_reference_point_details
                    )
                    shell_transfer_match_info["projection_pressure_used_as_diagnostic_only"] = True
                    shell_transfer_match_info["core_shell_inner_surface_nodes"] = int(
                        inner_surface_core_shell_indices.size
                    )
                    shell_transfer_match_info["core_shell_inner_surface_bonds"] = int(
                        inner_surface_shell_core_bond_count
                    )
                    shell_transfer_match_info["core_shell_inner_surface_match_max_distance"] = float(
                        inner_surface_core_shell_match_max_distance
                    )
                    shell_transfer_match_info["core_shell_inner_surface_match_bad_count"] = int(
                        inner_surface_core_shell_match_bad_count
                    )
                    shell_transfer_match_info["retained_transfer_pressure"] = float(
                        shell_transfer_retained_pressure
                    )
                    shell_transfer_match_info["conversion_min_steps"] = int(
                        shell_transfer_conversion_steps
                    )
                    iterations_done = int(shell_transfer_match_info["iterations"])
                    shell_transfer_body_force_initialized = True
                    just_initialized_shell_only = True
                    transfer_mag = np.sqrt(
                        transfer_only_br[:n_phys_shell_only_m] ** 2
                        + transfer_only_bz[:n_phys_shell_only_m] ** 2
                    )
                    shell_transfer_max_body_force = (
                        float(np.nanmax(transfer_mag)) if transfer_mag.size else 0.0
                    )
                    residual_transfer_mag = np.sqrt(
                        shell_transfer_br_shell_only[:n_phys_shell_only_m] ** 2
                        + shell_transfer_bz_shell_only[:n_phys_shell_only_m] ** 2
                    )
                    shell_transfer_match_info["max_residual_body_force"] = (
                        float(np.nanmax(residual_transfer_mag)) if residual_transfer_mag.size else 0.0
                    )

                    for pt in tracked_points:
                        target_coord = np.array([pt["target_r"], pt["target_z"]], dtype=np.float64)
                        target_is_core = bool(geom.core_mask_from_coords(
                            target_coord.reshape(1, 2),
                            r,
                            dshell,
                            center_x=r_start,
                            center_z=r,
                            modify_coordinates=modify_coordinates,
                            coordinate_x_scale=coordinate_x_scale,
                        )[0])
                        if target_is_core:
                            pt["inactive_after_shell_switch"] = True
                            pt["local_idx"] = None
                            pt["mirror_local_idx"] = None
                            continue

                        best_slice, best_local_idx, best_coord = mt.find_closest_point_in_phys(
                            [phys_coords_shell_only_m],
                            pt["target_r"],
                            pt["target_z"],
                        )
                        mirror_local_idx = None
                        mirror_coord = None
                        tracking_mode = "nearest_particle"
                        track_coord = best_coord
                        if best_coord is not None and abs(pt["target_z"] - r) <= tolerance:
                            mirror_target = np.array(
                                [best_coord[0], 2.0 * pt["target_z"] - best_coord[1]],
                                dtype=np.float64,
                            )
                            _, mirror_local_idx, mirror_coord = mt.find_closest_point_to_coord(
                                [phys_coords_shell_only_m],
                                mirror_target,
                            )
                            if mirror_coord is not None and mirror_local_idx != best_local_idx:
                                tracking_mode = "midline_symmetric_average"
                                track_coord = 0.5 * (best_coord + mirror_coord)

                        pt.update({
                            "slice": best_slice,
                            "local_idx": best_local_idx,
                            "coord": track_coord,
                            "primary_coord": best_coord,
                            "mirror_slice": best_slice,
                            "mirror_local_idx": mirror_local_idx,
                            "mirror_coord": mirror_coord,
                            "tracking_mode": tracking_mode,
                            "inactive_after_shell_switch": False,
                        })

                    print(
                        f"[Switch] shell-only mechanics initialized at t={thermal_time_after_update:.6e} s; "
                        f"Tmax={core_temp_max:.6f} K, "
                        f"mode=displacement_target_equivalent_body_force, "
                        f"conversion mech steps={iterations_done}, "
                        f"Ur final={final_ur:.6e}, target={target_ur:.6e}, "
                        f"Uz final={final_uz:.6e}, target={target_uz:.6e}, "
                        f"transfer pressure={shell_transfer_retained_pressure:.6e} Pa, "
                    )

                T_shell_mech_m = Tpre_avg + (T_shell_only_m - T_shell_reference_m)
                p_phase_target = float(pressure_volume_details["pressure_calculated"])
                p_phase_start = float(phase_pressure_applied)
                pressure_substep_target_count = max(1, int(pressure_ramp_steps))
                if abs(p_phase_target - p_phase_start) <= 1.0e-30:
                    pressure_substep_target_count = 1
                pressure_substeps_completed_this_step = 0
                mechanical_iterations_this_pressure_ramp = 0
                pressure_substep_start_index = pressure_ramp_counter
                p_phase = p_phase_start
                p = p_phase + shell_transfer_retained_pressure

                for pressure_substep in range(pressure_substep_target_count):
                    pressure_fraction = float(
                        (pressure_substep + 1) / pressure_substep_target_count
                    )
                    phase_pressure_applied = (
                        p_phase_start
                        + (p_phase_target - p_phase_start) * pressure_fraction
                    )
                    pressure_ramp_counter += 1
                    pressure_substeps_completed_this_step += 1
                    p_phase = float(phase_pressure_applied)
                    p = float(p_phase + shell_transfer_retained_pressure)
                    phase_pressure_br, phase_pressure_bz = sof.build_inner_pressure_body_force(
                        coords_shell_only_m,
                        surface_info_shell_only,
                        inner_surface_arc_correction_shell_only,
                        p_phase,
                        dr_i,
                        nu_shell,
                    )
                    transfer_pressure_br = (
                        shell_transfer_retained_pressure * unit_transfer_pressure_br_shell_only
                    )
                    transfer_pressure_bz = (
                        shell_transfer_retained_pressure * unit_transfer_pressure_bz_shell_only
                    )
                    br_shell_only = (
                        phase_pressure_br
                        + transfer_pressure_br
                        + shell_transfer_br_shell_only
                    )
                    bz_shell_only = (
                        phase_pressure_bz
                        + transfer_pressure_bz
                        + shell_transfer_bz_shell_only
                    )

                    substep_mechanical_steps = 0
                    mechanical_rms = 0.0
                    for step in range(nsteps_m):
                        Ur_prev = Ur_shell_only.copy()
                        Uz_prev = Uz_shell_only.copy()
                        Ur_shell_only[axis_mask_shell_only] = 0.0

                        damage_phi_shell_only = mt.compute_damage_variable_axisym_csr(
                            csr_indptr_shell_only_m,
                            csr_indices_shell_only_m,
                            csr_area_shell_only_m,
                            shape_edge_shell_only_m,
                            csr_shell_only,
                            crack_shell_only,
                            damage_bond_mask_shell_only,
                        )

                        Ur_shell_only, Uz_shell_only, Fr_shell_only, Fz_shell_only, \
                            Vr_half_shell_only, Vz_half_shell_only = mt.compute_mechanical_step_csr(
                                csr_indptr_shell_only_m,
                                csr_indices_shell_only_m,
                                csr_dist_shell_only_m,
                                csr_area_shell_only_m,
                                r_flat_shell_only,
                                z_flat_shell_only,
                                shape_edge_shell_only_m,
                                coeff,
                                lamda_edge_shell_only,
                                miu_edge_shell_only,
                                crack_shell_only,
                                s0_edge_shell_only,
                                Ur_shell_only,
                                Uz_shell_only,
                                br_shell_only,
                                bz_shell_only,
                                Fr_0_shell_only,
                                Fz_0_shell_only,
                                Vr_half_shell_only,
                                Vz_half_shell_only,
                                lambda_diag_shell_only,
                                dt_ADR,
                                delta_i,
                                T_shell_mech_m,
                                Tpre_avg,
                                kprime_edge_shell_only,
                                alpha_edge_shell_only,
                                csr_shell_only,
                                enable_cracking=enable_cracking,
                                damage_node=damage_phi_shell_only,
                                crack_pressure=p,
                                damage_threshold=damage_threshold,
                                enable_crack_pressure_force=enable_crack_pressure_force,
                                use_crack_in_dilation=True,
                            )
                        Fr_0_shell_only = Fr_shell_only.copy()
                        Fz_0_shell_only = Fz_shell_only.copy()
                        Ur_shell_only[axis_mask_shell_only] = 0.0
                        substep_mechanical_steps = step + 1

                        damage_phi_shell_only = mt.compute_damage_variable_axisym_csr(
                            csr_indptr_shell_only_m,
                            csr_indices_shell_only_m,
                            csr_area_shell_only_m,
                            shape_edge_shell_only_m,
                            csr_shell_only,
                            crack_shell_only,
                            damage_bond_mask_shell_only,
                        )
                        mechanical_rms = np.sqrt(
                            np.mean(
                                (Ur_shell_only - Ur_prev) ** 2
                                + (Uz_shell_only - Uz_prev) ** 2
                            )
                        )
                        mechanical_converged_step = step
                        if mt.has_outer_surface_damage_node(
                            damage_phi_shell_only,
                            outer_surface_node_mask_shell_only,
                            damage_threshold,
                        ):
                            stop_time_loop = True
                            stop_reason = "outer_surface_damage"
                            print(
                                f"[Mechanical shell-only] stopped at pressure substep "
                                f"{pressure_substep + 1}/{pressure_substep_target_count}, "
                                f"step {step}: outer shell surface damage exceeded "
                                f"{damage_threshold:.3f}"
                            )
                            break

                        if mechanical_rms < shell_only_rms:
                            print(
                                f"[Mechanical shell-only] pressure substep "
                                f"{pressure_substep + 1}/{pressure_substep_target_count} "
                                f"converged at step {step} with RMS {mechanical_rms:.3e}, "
                                f"p = {p:.6g}"
                            )
                            break

                        if step > 0 and step % 10 == 0:
                            print(
                                f"[Mechanical shell-only] pressure substep "
                                f"{pressure_substep + 1}/{pressure_substep_target_count}, "
                                f"step {step}, RMS {mechanical_rms:.3e}"
                            )

                    mechanical_iterations_this_pressure_ramp += substep_mechanical_steps
                    if stop_time_loop:
                        break

                if mechanical_iterations_this_pressure_ramp > 0:
                    mechanical_converged_step = mechanical_iterations_this_pressure_ramp - 1

                phase_pressure_factor = (
                    0.0
                    if abs(p_phase_target) <= 1.0e-30
                    else float(p_phase / p_phase_target)
                )
                pressure_volume_details["pressure_phase_target"] = float(p_phase_target)
                pressure_volume_details["pressure_phase_factor"] = float(phase_pressure_factor)
                pressure_volume_details["pressure_phase_ramp_mode"] = (
                    "linear_substeps_within_current_thermal_step"
                )
                pressure_volume_details["pressure_load_factor"] = float(phase_pressure_factor)
                pressure_volume_details["pressure_ramp_steps"] = int(pressure_ramp_steps)
                pressure_volume_details["pressure_ramp_completed_steps"] = int(
                    pressure_ramp_counter
                )
                pressure_volume_details["pressure_substep_start_index"] = int(
                    pressure_substep_start_index
                )
                pressure_volume_details["pressure_substeps_this_thermal_step"] = int(
                    pressure_substeps_completed_this_step
                )
                pressure_volume_details["pressure_substep_target_count"] = int(
                    pressure_substep_target_count
                )
                pressure_volume_details["mechanical_iterations_this_pressure_ramp"] = int(
                    mechanical_iterations_this_pressure_ramp
                )
                pressure_volume_details["pressure_applied"] = float(p)
                pressure_volume_details["pressure_mechanical_load"] = float(p)
                pressure_volume_details["pressure_transfer"] = float(shell_transfer_retained_pressure)
                pressure_volume_details["pressure_transfer_applied"] = float(
                    shell_transfer_retained_pressure
                )
                pressure_volume_details["pressure_phase"] = float(p_phase)
                pressure_volume_details["pressure_active"] = True
                pressure_volume_details["mechanical_mode"] = mechanical_mode

        current_time = thermal_time_after_update
        pressure_volume_details["mechanical_mode"] = mechanical_mode
        pressure_volume_formula = float(pressure_volume_details["pressure_calculated"])
        if shell_transfer_body_force_initialized:
            pressure_integral_value = None
            if shell_transfer_reference_point_pressure is None:
                pressure_record_value = pressure_volume_formula
                pressure_source = "volume_formula_after_shell_only_conversion_no_point_reference"
                pressure_point_details = {
                    "label": pressure_sample_point_label,
                    "target_r": float(pressure_sample_point_target[0]),
                    "target_z": float(pressure_sample_point_target[1]),
                    "source": pressure_source,
                    "valid": False,
                    "reason": "missing_shell_transfer_reference_point_pressure",
                    "time_scope": "after_shell_only_conversion",
                }
            else:
                reference_point_pressure = float(shell_transfer_reference_point_pressure)
                pressure_record_value = reference_point_pressure + float(p_phase)
                pressure_source = "lame_displacement_reference_plus_phase_after_shell_only_conversion"
                pressure_point_details = dict(shell_transfer_reference_point_details or {})
                pressure_point_details.update({
                    "source": pressure_source,
                    "valid": True,
                    "time_scope": "after_shell_only_conversion",
                    "reference_pressure": reference_point_pressure,
                    "phase_pressure": float(p_phase),
                    "pressure": float(pressure_record_value),
                    "pressure_raw": float(pressure_record_value),
                    "pressure_volume_formula": float(pressure_volume_formula),
                    "pressure_mechanical_load": float(p),
                    "no_direct_core_shell_integral_after_conversion": True,
                })
            pressure_point_details.update({
                "label": pressure_sample_point_label,
                "target_r": float(pressure_sample_point_target[0]),
                "target_z": float(pressure_sample_point_target[1]),
            })
        else:
            pressure_integral_value, pressure_point_details = (
                compute_pressure_sample_lame_for_current_core_shell_state()
            )
            pressure_record_value = float(pressure_integral_value)
            pressure_source = pressure_local_source

        pressure_volume_details["pressure"] = float(pressure_record_value)
        pressure_volume_details["pressure_source"] = pressure_source
        pressure_volume_details["pressure_integral_at_point"] = finite_or_none(pressure_integral_value)
        pressure_volume_details["pressure_volume_formula"] = float(pressure_volume_formula)
        pressure_volume_details["pressure_applied"] = float(p)
        pressure_volume_details["pressure_sample_point"] = pressure_sample_point_label

        time_history.append(float(current_time))
        mechanical_mode_history.append(mechanical_mode)
        mechanical_step_history.append(int(mechanical_converged_step))
        mechanical_rms_history.append(float(mechanical_rms))
        pressure_history.append(float(pressure_record_value))
        pressure_formula_history.append(float(pressure_volume_formula))
        pressure_load_factor_history.append(float(pressure_volume_details["pressure_load_factor"]))
        pressure_active_history.append(bool(pressure_volume_details["pressure_active"]))
        pressure_integral_history.append(finite_or_none(pressure_integral_value))
        pressure_volume_formula_history.append(float(pressure_volume_formula))
        pressure_applied_history.append(float(p))
        pressure_source_history.append(pressure_source)
        pressure_point_detail_history.append(pressure_point_details)
        pressure_volume_detail_history.append(pressure_volume_details)
        core_temp_min_history.append(core_temp_min)
        core_temp_max_history.append(core_temp_max)
        core_temp_avg_history.append(core_temp_avg)
        shell_temp_max_history.append(shell_temp_max)
        thermal_dt_history.append(float(thermal_dt_this))
        thermal_stepper_history.append(thermal_stepper)
        thermal_stage_history.append(1)

        for pt in tracked_points:
            i_local = pt["local_idx"]
            mirror_local_idx = pt["mirror_local_idx"]
            if shell_transfer_body_force_initialized:
                Ur_track = Ur_shell_only
                Uz_track = Uz_shell_only
                T_track = T_shell_only_m
            else:
                Ur_track = Ur
                Uz_track = Uz
                T_track = T_m

            if pt.get("inactive_after_shell_switch") or i_local is None:
                current_ur = np.nan
                current_uz = np.nan
                current_T = np.nan
            elif pt.get("tracking_mode") == "midline_symmetric_average" and mirror_local_idx is not None:
                current_ur = 0.5 * (Ur_track[i_local] + Ur_track[mirror_local_idx])
                current_uz = 0.5 * (Uz_track[i_local] + Uz_track[mirror_local_idx])
                current_T = 0.5 * (T_track[i_local] + T_track[mirror_local_idx])
            else:
                current_ur = Ur_track[i_local]
                current_uz = Uz_track[i_local]
                current_T = T_track[i_local]

            ur_histories[pt["label"]].append(float(current_ur))
            uz_histories[pt["label"]].append(float(current_uz))
            T_histories[pt["label"]].append(float(current_T))

        if step1 == 0:
            print(
                f"[Temperature] step {step1 + 1}/{nsteps_th}, "
                f"t = {current_time:.12e} / {total_time:.12e} s, "
                f"T_boundary = {T_increment:.6f} K, "
                f"T_core_max = {core_temp_max:.6f} K"
            )

        if stop_time_loop:
            print(f"[Time loop] stopped at t={current_time:.12e} s due to {stop_reason}")
            break

    # ------------------------
    # Post-processing plots
    # ------------------------
    final_time = time_history[-1] if time_history else np.nan
    print("[Tracked final values]")
    tracked_final_rows = []
    for pt in tracked_points:
        label = pt["label"]
        final_ur = ur_histories[label][-1] if ur_histories[label] else np.nan
        final_uz = uz_histories[label][-1] if uz_histories[label] else np.nan
        final_t = T_histories[label][-1] if T_histories[label] else np.nan
        final_umag = np.sqrt(final_ur ** 2 + final_uz ** 2)
        tracked_final_rows.append({
            "label": label,
            "target_r": float(pt["target_r"]),
            "target_z": float(pt["target_z"]),
            "coord_r": float(pt["coord"][0]) if pt["coord"] is not None else np.nan,
            "coord_z": float(pt["coord"][1]) if pt["coord"] is not None else np.nan,
            "Ur_final_m": float(final_ur),
            "Uz_final_m": float(final_uz),
            "Umag_final_m": float(final_umag),
            "T_final_K": float(final_t),
        })
        print(
            f"  {label}: Ur={final_ur:.12e} m, Uz={final_uz:.12e} m, "
            f"Umag={final_umag:.12e} m, T={final_t:.6f} K"
        )

    if shell_transfer_body_force_initialized:
        plot_phys_coords_m = phys_coords_shell_only_m
        plot_Ur = Ur_shell_only[:n_phys_shell_only_m]
        plot_Uz = Uz_shell_only[:n_phys_shell_only_m]
        plot_damage = mt.compute_damage_variable_axisym_csr(
            csr_indptr_shell_only_m,
            csr_indices_shell_only_m,
            csr_area_shell_only_m,
            shape_edge_shell_only_m,
            csr_shell_only,
            crack_shell_only,
            damage_bond_mask_shell_only,
        )[:n_phys_shell_only_m]
        plot_outer_surface_mask = outer_surface_node_mask_shell_only[:n_phys_shell_only_m]
        plot_shell_only = True
        final_damage_bond_mask = damage_bond_mask_shell_only
        final_crack = crack_shell_only
    else:
        plot_phys_coords_m = phys_coords_m
        plot_Ur = Ur[:n_phys_m]
        plot_Uz = Uz[:n_phys_m]
        plot_damage = mt.compute_damage_variable_axisym_csr(
            csr_indptr_m,
            csr_indices_m,
            csr_area_m,
            shape_edge_m,
            csr,
            crack,
            damage_bond_mask,
        )[:n_phys_m]
        plot_outer_surface_mask = outer_surface_node_mask[:n_phys_m]
        plot_shell_only = False
        final_damage_bond_mask = damage_bond_mask
        final_crack = crack

    final_pressure = float(pressure_history[-1]) if pressure_history else 0.0
    broken_shell_bonds = int(np.count_nonzero(final_damage_bond_mask & (final_crack == 0)))
    total_shell_bonds = int(np.count_nonzero(final_damage_bond_mask))
    max_damage = float(np.nanmax(plot_damage)) if plot_damage.size else 0.0
    max_outer_surface_damage = (
        float(np.nanmax(plot_damage[plot_outer_surface_mask]))
        if np.any(plot_outer_surface_mask)
        else 0.0
    )
    print(
        f"[Crack check] Ps = {final_pressure:.12e} Pa, "
        f"broken_shell_bonds = {broken_shell_bonds}/{total_shell_bonds}, "
        f"max_damage = {max_damage:.6e}, "
        f"max_outer_surface_damage = {max_outer_surface_damage:.6e}"
    )

    pd_results_path = (
        os.environ.get("THM_PD_RESULTS_JSON_PATH")
        or os.environ.get("THM_METRICS_JSON_PATH")
    )
    if pd_results_path:
        pd_results_dir = os.path.dirname(pd_results_path)
        if pd_results_dir:
            os.makedirs(pd_results_dir, exist_ok=True)
        with open(pd_results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case": "nano3_p84_shell_pressure",
                    "materials": {
                        "nano3_solid": {
                            "rho": rho_s,
                            "specific_heat": cs,
                            "conductivity": ks,
                            "youngs_modulus": E_core,
                            "poisson_ratio": nu_core,
                            "thermal_expansion": alpha_core_s,
                        },
                        "nano3_liquid": {
                            "rho": rho_l,
                            "specific_heat": cl,
                            "conductivity": kl,
                            "thermal_expansion": alpha_core_l,
                            "compressibility": comp_l,
                        },
                        "p84_shell": {
                            "rho": rho_shell,
                            "specific_heat": c_shell,
                            "conductivity": k_shell,
                            "youngs_modulus": E_shell,
                            "poisson_ratio": nu_shell,
                            "thermal_expansion": alpha_shell,
                            "tensile_strength": sigmat,
                        },
                    },
                    "geometry": {
                        "radius": r,
                        "shell_thickness": dshell,
                        "core_radius": r_core,
                        "dr": dr_i,
                        "delta": delta_i,
                    },
                    "thermal_boundary": {
                        "initial_temperature": Tinit,
                        "surface_temperature": Tsurr,
                        "use_boundary_ramp": use_boundary_ramp,
                    },
                    "phase_change": {
                        "Ts": Ts,
                        "Tl": Tl,
                        "latent_heat": L,
                        "volume_expansion_fraction": phase_volume_expansion_fraction,
                        "pressure_model": "lame_displacement_before_shell_only_then_density_formula",
                    },
                    "final_time": float(final_time),
                    "final_pressure": final_pressure,
                    "tracked_final_values": tracked_final_rows,
                    "pressure_sample_point": pressure_sample_point_info,
                    "time_history": time_history,
                    "pressure_history": pressure_history,
                    "pressure_formula_history": pressure_formula_history,
                    "pressure_load_factor_history": pressure_load_factor_history,
                    "pressure_active_history": pressure_active_history,
                    "pressure_integral_history": pressure_integral_history,
                    "pressure_volume_formula_history": pressure_volume_formula_history,
                    "pressure_applied_history": pressure_applied_history,
                    "pressure_source_history": pressure_source_history,
                    "pressure_point_detail_history": pressure_point_detail_history,
                    "pressure_volume_detail_history": pressure_volume_detail_history,
                    "pressure_activation": "core_temp_max_ge_Ts",
                    "shell_transfer_match_info": shell_transfer_match_info,
                    "core_temp_min_history": core_temp_min_history,
                    "core_temp_max_history": core_temp_max_history,
                    "core_temp_avg_history": core_temp_avg_history,
                    "shell_temp_max_history": shell_temp_max_history,
                    "thermal_stepper_history": thermal_stepper_history,
                    "mechanical_mode_history": mechanical_mode_history,
                    "ur_histories": ur_histories,
                    "uz_histories": uz_histories,
                    "temperature_histories": T_histories,
                },
                f,
                indent=2,
            )
        print(f"[PD results] saved to {pd_results_path}")

    end_time2 = time.time()
    print("Execution time for this section:", end_time2 - start_time2)

    if not skip_plots:
        plot.plot_temperature_contour_in_circle(
            [phys_coords_t],
            dr_i,
            [T[sl_phys_t]],
            radius=r,
            cmap="viridis",
            title=r"$T$ (K)",
            shell_thickness=dshell,
            shell_only=False,
            r_start=r_start,
            modify_coordinates=modify_coordinates,
            coordinate_x_scale=coordinate_x_scale,
        )

        U_phys = {
            0: {
                "Ur": plot_Ur,
                "Uz": plot_Uz,
                "Umag": np.sqrt(plot_Ur ** 2 + plot_Uz ** 2),
            }
        }
        plot.plot_displacement_contours_in_circle(
            [plot_phys_coords_m],
            U_phys,
            r,
            dr_i,
            r_start,
            titles=("Ur (m)", "Uz (m)", "Umag (m)"),
            levels=10,
            shell_thickness=dshell,
            shell_only=plot_shell_only,
            modify_coordinates=modify_coordinates,
            coordinate_x_scale=coordinate_x_scale,
        )

        if plot_phys_coords_m.shape[0] > 0:
            coords_plot = plot_phys_coords_m[:, :2]
            damage_plot = np.clip(plot_damage, 0.0, 1.0)
            damage_vmax = max(
                float(np.nanmax(damage_plot)) if damage_plot.size else 0.0,
                damage_threshold,
            )
            grid_n = 300
            r_lin = np.linspace(r_start, r_start + outer_axis_x, grid_n)
            z_lin = np.linspace(0.0, 2.0 * r, grid_n)
            r_grid, z_grid = np.meshgrid(r_lin, z_lin)
            damage_grid = griddata(coords_plot, damage_plot, (r_grid, z_grid), method="linear")
            damage_nearest = griddata(coords_plot, damage_plot, (r_grid, z_grid), method="nearest")
            damage_grid = np.where(np.isfinite(damage_grid), damage_grid, damage_nearest)

            outer_level = geom.ellipse_level(
                r_grid,
                z_grid,
                r_start,
                r,
                outer_axis_x,
                outer_axis_z,
            )
            mask_domain = (
                (outer_level <= 1.0)
                & (r_grid >= r_start)
                & (z_grid >= 0.0)
                & (z_grid <= 2.0 * r)
            )
            inner_level = geom.ellipse_level(
                r_grid,
                z_grid,
                r_start,
                r,
                inner_axis_x,
                inner_axis_z,
            )
            if plot_shell_only:
                mask_domain &= inner_level >= 1.0
            damage_grid[~mask_domain] = np.nan

            theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 360)
            outer_r_plot = r_start + outer_axis_x * np.cos(theta)
            outer_z_plot = r + outer_axis_z * np.sin(theta)
            inner_r_plot = r_start + inner_axis_x * np.cos(theta)
            inner_z_plot = r + inner_axis_z * np.sin(theta)

            fig_damage, ax_damage = plt.subplots(figsize=(7.5, 7.0), facecolor="white")
            ax_damage.set_facecolor("white")
            damage_contour = ax_damage.contourf(
                r_grid * 1.0e6,
                z_grid * 1.0e6,
                damage_grid,
                levels=np.linspace(0.0, damage_vmax, 11),
                cmap="inferno",
                vmin=0.0,
                vmax=damage_vmax,
            )
            ax_damage.plot(outer_r_plot * 1.0e6, outer_z_plot * 1.0e6, color="black", linewidth=0.8)
            ax_damage.plot(
                inner_r_plot * 1.0e6,
                inner_z_plot * 1.0e6,
                color="black",
                linewidth=0.8,
                linestyle="--",
            )
            cbar = fig_damage.colorbar(damage_contour, ax=ax_damage, fraction=0.046, pad=0.04)
            cbar.set_label("Damage variable")
            ax_damage.set_aspect("equal", adjustable="box")
            ax_damage.set_xlabel("r (um)")
            ax_damage.set_ylabel("z (um)")
            ax_damage.set_title(f"Damage variable, Ps={final_pressure / 1.0e6:.1f} MPa")
            ax_damage.set_xlim(r_start * 1.0e6, (r_start + outer_axis_x) * 1.0e6)
            ax_damage.set_ylim(0.0, 2.0 * r * 1.0e6)
            fig_damage.tight_layout()

            damage_plot_path = os.environ.get("THM_DAMAGE_PLOT_PATH")
            if damage_plot_path:
                damage_plot_dir = os.path.dirname(damage_plot_path)
                if damage_plot_dir:
                    os.makedirs(damage_plot_dir, exist_ok=True)
                fig_damage.savefig(damage_plot_path, dpi=300)
                print(f"[Damage plot] saved to {damage_plot_path}")
            plt.show()

        if time_history:
            history_plot_dir = os.environ.get("THM_HISTORY_PLOT_DIR")
            if history_plot_dir:
                os.makedirs(history_plot_dir, exist_ok=True)
            time_arr = np.asarray(time_history, dtype=np.float64)
            for history_name, history_dict, ylabel, filename in (
                ("Tracked point Ur histories", ur_histories, "Ur (m)", "tracked_ur_history.png"),
                ("Tracked point Uz histories", uz_histories, "Uz (m)", "tracked_uz_history.png"),
                ("Tracked point temperature histories", T_histories, "T (K)", "tracked_temperature_history.png"),
            ):
                fig_hist, ax_hist = plt.subplots(figsize=(9.0, 5.0), facecolor="white")
                for pt in tracked_points:
                    label = pt["label"]
                    ax_hist.plot(time_arr, np.asarray(history_dict[label], dtype=np.float64), label=label)
                ax_hist.set_xlabel("Time (s)")
                ax_hist.set_ylabel(ylabel)
                ax_hist.set_title(history_name)
                ax_hist.grid(True, linestyle="--", alpha=0.35)
                ax_hist.legend(loc="best")
                fig_hist.tight_layout()
                if history_plot_dir:
                    history_path = os.path.join(history_plot_dir, filename)
                    fig_hist.savefig(history_path, dpi=300)
                    print(f"[Tracked plot] saved to {history_path}")
                plt.show()

        if pressure_plot:
            pressure_plot_path = (
                os.environ.get("THM_PRESSURE_PLOT_PATH")
                or os.path.join("figures", "pressure_history.png")
            )
            plot.plot_pressure_history(
                time_history,
                pressure_history,
                pressure_detail_history=pressure_volume_detail_history,
                mechanical_mode_history=mechanical_mode_history,
                title=r"$p_{(r_{\mathrm{shell}}, r)}$ history",
                line_label=r"$p_{(r_{\mathrm{shell}}, r)}$",
                save_path=pressure_plot_path,
                include_components=True,
            )
        else:
            print("[Pressure plot] disabled because THM_PRESSURE_PLOT=0")
    else:
        print("[Plots] skipped because THM_SKIP_PLOTS=1")
