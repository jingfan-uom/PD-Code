import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

import geometry_utils as geom
import Multiprocess_task_function as mt
import Physical_Field_Calculation as pfc


def shell_pressure_from_relative_volume_change(
        E,
        nu,
        R,
        r_m0,
        density_change,
        alpha=0.0,
        deltaT=0.0,
        eps=1e-30,
):
    """Pressure from PCM relative volume change; density_change is legacy dV/V input."""
    if R <= eps:
        raise ValueError("R must be positive")
    if r_m0 <= eps or r_m0 >= R:
        raise ValueError("r_m0 must satisfy 0 < r_m0 < R")
    density_change = float(density_change)
    if not np.isfinite(density_change):
        raise ValueError("density_change must be finite")

    shell_growth = 1.0 + float(alpha) * float(deltaT)
    core_growth = np.cbrt(max(0.0, 1.0 + density_change))
    denom = r_m0 ** 3 * (2.0 - 4.0 * nu) + R ** 3 * (1.0 + nu)
    if abs(denom) <= eps:
        raise ZeroDivisionError("pressure formula denominator is too small")

    return float(
        2.0 * (R ** 3 - r_m0 ** 3) * E
        * (core_growth - shell_growth)
        / denom
    )


def lame_internal_pressure_from_shell_radial_displacement(
        E,
        nu,
        outer_radius,
        inner_radius,
        radial_position,
        radial_displacement,
        alpha=0.0,
        deltaT=0.0,
        eps=1e-30,
):
    """
    Equivalent uniform internal pressure from Lame thick-spherical-shell displacement.

    The displacement is measured at radial_position in the shell. A uniform
    shell thermal strain alpha * deltaT is removed before solving for pressure.
    """
    E = float(E)
    nu = float(nu)
    outer_radius = float(outer_radius)
    inner_radius = float(inner_radius)
    radial_position = float(radial_position)
    radial_displacement = float(radial_displacement)
    if E <= eps:
        raise ValueError("E must be positive")
    if outer_radius <= eps:
        raise ValueError("outer_radius must be positive")
    if inner_radius <= eps or inner_radius >= outer_radius:
        raise ValueError("inner_radius must satisfy 0 < inner_radius < outer_radius")
    if radial_position <= eps:
        raise ValueError("radial_position must be positive")

    shell_span = outer_radius ** 3 - inner_radius ** 3
    if shell_span <= eps:
        raise ValueError("outer_radius and inner_radius are too close")

    compliance = (
        (1.0 - 2.0 * nu) * inner_radius ** 3 * radial_position
        + 0.5 * (1.0 + nu) * inner_radius ** 3 * outer_radius ** 3
        / (radial_position ** 2)
    ) / (E * shell_span)
    if abs(compliance) <= eps:
        raise ZeroDivisionError("Lame displacement compliance is too small")

    thermal_displacement = float(alpha) * float(deltaT) * radial_position
    mechanical_displacement = radial_displacement - thermal_displacement
    return float(mechanical_displacement / compliance)


def _core_pcm_density_change_terms(
        T,
        reference_temperature,
        Ts,
        Tl,
        rho_s,
        rho_l,
        alpha_s,
        alpha_l,
        pressure=0.0,
        liquid_compressibility=0.0,
        eps=1e-30,
):
    T = np.asarray(T, dtype=np.float64)
    T_ref = float(reference_temperature)
    Ts = float(Ts)
    Tl = float(Tl)
    phase_width = max(Tl - Ts, 1.0e-30)
    rho_s = float(rho_s)
    rho_l = float(rho_l)
    if rho_s <= eps or rho_l <= eps:
        raise ValueError("rho_s and rho_l must be positive")
    alpha_l = float(alpha_l)
    pressure = max(0.0, float(pressure))
    liquid_compressibility = max(0.0, float(liquid_compressibility))

    liquid_fraction = np.clip((T - Ts) / phase_width, 0.0, 1.0)
    liquid_fraction_ref = float(np.clip((T_ref - Ts) / phase_width, 0.0, 1.0))
    dT = T - T_ref

    phase_density_ratio = rho_s / rho_l - 1.0
    phase_density_change = (liquid_fraction - liquid_fraction_ref) * phase_density_ratio

    liquid_growth = np.maximum(1.0 + 3.0 * alpha_l * dT, eps)
    liquid_compression = np.maximum(1.0 - liquid_compressibility * pressure, eps)
    liquid_volume_factor = liquid_growth * liquid_compression
    solid_thermal_density_change = np.zeros_like(T, dtype=np.float64)
    liquid_thermal_density_change = (
        liquid_fraction * (rho_s / rho_l) * (liquid_growth - 1.0)
    )
    liquid_compression_density_change = (
        liquid_fraction
        * (rho_s / rho_l)
        * liquid_growth
        * (liquid_compression - 1.0)
    )
    thermal_density_change = solid_thermal_density_change + liquid_thermal_density_change
    total_density_change = (
        phase_density_change
        + thermal_density_change
        + liquid_compression_density_change
    )

    rho_s_eff = np.full_like(T, rho_s, dtype=np.float64)
    rho_l_eff = rho_l / liquid_volume_factor

    return {
        "liquid_fraction": liquid_fraction,
        "liquid_fraction_ref": liquid_fraction_ref,
        "phase_density_ratio": float(phase_density_ratio),
        "phase_density_change": phase_density_change,
        "solid_thermal_density_change": solid_thermal_density_change,
        "liquid_thermal_density_change": liquid_thermal_density_change,
        "liquid_compression_density_change": liquid_compression_density_change,
        "thermal_density_change": thermal_density_change,
        "total_density_change": total_density_change,
        "rho_s_eff": rho_s_eff,
        "rho_l_eff": rho_l_eff,
        "liquid_growth": liquid_growth,
        "liquid_compression": np.full_like(T, float(liquid_compression), dtype=np.float64),
        "pressure_for_density": float(pressure),
        "liquid_compressibility": float(liquid_compressibility),
    }


def _aggregate_core_pcm_density_change_terms(
        terms,
        cv_core,
        core_volume_ref,
        rho_s,
        rho_l,
        eps=1e-30,
):
    cv_core = np.asarray(cv_core, dtype=np.float64)
    core_volume_ref = max(float(core_volume_ref), eps)
    liquid_fraction = terms["liquid_fraction"]
    liquid_volume_equiv = float(np.sum(liquid_fraction * cv_core))
    solid_density_weight = float(np.sum((1.0 - liquid_fraction) * cv_core))
    liquid_density_weight = float(np.sum(liquid_fraction * cv_core))

    out = {
        "liquid_fraction": liquid_fraction,
        "liquid_volume_equiv": liquid_volume_equiv,
        "density_change_total": float(
            np.sum(terms["total_density_change"] * cv_core) / core_volume_ref
        ),
        "phase_density_change": float(
            np.sum(terms["phase_density_change"] * cv_core) / core_volume_ref
        ),
        "thermal_density_change": float(
            np.sum(terms["thermal_density_change"] * cv_core) / core_volume_ref
        ),
        "solid_thermal_density_change": float(
            np.sum(terms["solid_thermal_density_change"] * cv_core) / core_volume_ref
        ),
        "liquid_thermal_density_change": float(
            np.sum(terms["liquid_thermal_density_change"] * cv_core) / core_volume_ref
        ),
        "liquid_compression_density_change": float(
            np.sum(terms["liquid_compression_density_change"] * cv_core) / core_volume_ref
        ),
        "solid_density_weight": solid_density_weight,
        "liquid_density_weight": liquid_density_weight,
        "phase_density_ratio": float(terms["phase_density_ratio"]),
        "pressure_for_density": float(terms["pressure_for_density"]),
        "liquid_compressibility": float(terms["liquid_compressibility"]),
    }

    rho_s = float(rho_s)
    rho_l = float(rho_l)
    if solid_density_weight > eps:
        delta_rho_s_avg = float(
            np.sum(
                (terms["rho_s_eff"] - rho_s)
                * (1.0 - liquid_fraction)
                * cv_core
            ) / solid_density_weight
        )
        rho_s_eff_avg = float(rho_s + delta_rho_s_avg)
    else:
        delta_rho_s_avg = 0.0
        rho_s_eff_avg = rho_s

    if liquid_density_weight > eps:
        delta_rho_l_avg = float(
            np.sum((terms["rho_l_eff"] - rho_l) * liquid_fraction * cv_core)
            / liquid_density_weight
        )
        rho_l_eff_avg = float(rho_l + delta_rho_l_avg)
    else:
        delta_rho_l_avg = 0.0
        rho_l_eff_avg = rho_l

    out.update({
        "delta_rho_s_avg": float(delta_rho_s_avg),
        "delta_rho_l_avg": float(delta_rho_l_avg),
        "rho_s_eff_avg": float(rho_s_eff_avg),
        "rho_l_eff_avg": float(rho_l_eff_avg),
    })
    return out


def iterate_liquid_density_for_shell_pressure(
        T_core,
        cv_core,
        core_volume_ref,
        shell_deltaT,
        core_reference_temperature,
        rho_s,
        rho_l,
        alpha_core_l,
        Ts,
        Tl,
        E_shell,
        nu_shell,
        outer_radius,
        inner_radius,
        alpha_shell,
        liquid_compressibility,
        density_rel_tol=1.0e-8,
        max_iter=80,
        initial_pressure=0.0,
        eps=1e-30,
):
    """
    Fixed-point update of pressure and liquid density for one-way coupling.

    The mechanical state is not solved inside this routine. It only updates the
    formula pressure and the pressure-dependent liquid density until the
    liquid-density relative change is at most density_rel_tol.
    """
    T_core = np.asarray(T_core, dtype=np.float64)
    cv_core = np.asarray(cv_core, dtype=np.float64)
    if T_core.shape[0] != cv_core.shape[0]:
        raise ValueError("T_core and cv_core must have the same length")
    if density_rel_tol <= 0.0:
        raise ValueError("density_rel_tol must be positive")
    max_iter = max(1, int(max_iter))
    liquid_compressibility = max(0.0, float(liquid_compressibility))
    pressure_limit = (
        np.inf if liquid_compressibility <= 0.0
        else 0.95 / liquid_compressibility
    )

    def evaluate(pressure_for_density):
        terms = _core_pcm_density_change_terms(
            T_core,
            core_reference_temperature,
            Ts,
            Tl,
            rho_s,
            rho_l,
            0.0,
            alpha_core_l,
            pressure=pressure_for_density,
            liquid_compressibility=liquid_compressibility,
            eps=eps,
        )
        details = _aggregate_core_pcm_density_change_terms(
            terms,
            cv_core,
            core_volume_ref,
            rho_s,
            rho_l,
            eps=eps,
        )
        pressure_formula = shell_pressure_from_relative_volume_change(
            E_shell,
            nu_shell,
            outer_radius,
            inner_radius,
            details["density_change_total"],
            alpha=alpha_shell,
            deltaT=shell_deltaT,
            eps=eps,
        )
        details.update({
            "pressure_formula": float(pressure_formula),
            "pressure": float(pressure_formula),
            "pcm_growth": float(
                np.cbrt(max(0.0, 1.0 + details["density_change_total"]))
            ),
            "shell_growth": float(1.0 + float(alpha_shell) * float(shell_deltaT)),
            "shell_deltaT": float(shell_deltaT),
        })
        return details

    pressure_for_density = min(max(0.0, float(initial_pressure)), pressure_limit)
    previous_rho_l_eff_avg = None
    history = []
    final_details = evaluate(pressure_for_density)
    converged = liquid_compressibility <= 0.0
    density_rel_error = 0.0 if converged else np.inf

    if not converged:
        for iteration in range(1, max_iter + 1):
            final_details = evaluate(pressure_for_density)
            rho_l_eff_avg = float(final_details["rho_l_eff_avg"])
            if previous_rho_l_eff_avg is None:
                density_rel_error = np.inf
            else:
                density_rel_error = abs(rho_l_eff_avg - previous_rho_l_eff_avg) / max(
                    abs(rho_l_eff_avg),
                    eps,
                )

            next_pressure_for_density = min(
                max(0.0, float(final_details["pressure_formula"])),
                pressure_limit,
            )
            pressure_rel_error = abs(
                next_pressure_for_density - pressure_for_density
            ) / max(abs(next_pressure_for_density), 1.0)
            history.append({
                "iteration": int(iteration),
                "pressure_for_density": float(pressure_for_density),
                "pressure_formula": float(final_details["pressure_formula"]),
                "rho_l_eff_avg": float(rho_l_eff_avg),
                "delta_rho_l_avg": float(final_details["delta_rho_l_avg"]),
                "density_rel_error": (
                    None if not np.isfinite(density_rel_error)
                    else float(density_rel_error)
                ),
                "pressure_rel_error": float(pressure_rel_error),
            })

            if previous_rho_l_eff_avg is not None and density_rel_error <= density_rel_tol:
                converged = True
                break

            previous_rho_l_eff_avg = rho_l_eff_avg
            pressure_for_density = next_pressure_for_density

    final_details.update({
        "density_pressure_solved": bool(liquid_compressibility > 0.0),
        "density_converged": bool(converged),
        "density_iterations": int(len(history)),
        "density_rel_error": (
            0.0 if not history and liquid_compressibility <= 0.0
            else float(density_rel_error)
        ),
        "density_rel_tol": float(density_rel_tol),
        "density_iteration_history": history,
        "pressure_for_density": float(final_details["pressure_for_density"]),
        "pressure_density_iteration_mode": "one_way_fixed_point",
        "pressure_limited_by_compressibility": bool(
            np.isfinite(pressure_limit)
            and max(0.0, float(final_details["pressure_formula"])) > pressure_limit
        ),
    })
    return final_details


def compute_fixed_phase_pressure_volume_details(
        T_phys,
        mask_core_phy_t,
        phys_coords_t,
        dr,
        core_volume_ref,
        shell_deltaT,
        core_reference_temperature,
        phase_volume_expansion_fraction,
        alpha_core_s,
        alpha_core_l,
        Ts,
        Tl,
        E_shell,
        nu_shell,
        outer_radius,
        inner_radius,
        alpha_shell,
        rho_s=None,
        rho_l=None,
        liquid_compressibility=0.0,
        density_rel_tol=1.0e-8,
        density_iter_max=80,
):
    phase_volume_expansion = float(phase_volume_expansion_fraction)
    rho_s_value = None if rho_s is None else float(rho_s)
    rho_l_value = None if rho_l is None else float(rho_l)
    liquid_volume_equiv = 0.0
    density_change_total = 0.0
    phase_density_change = 0.0
    thermal_density_change = 0.0
    solid_thermal_density_change = 0.0
    liquid_thermal_density_change = 0.0
    liquid_compression_density_change = 0.0
    delta_rho_s_avg = 0.0
    delta_rho_l_avg = 0.0
    rho_s_eff_avg = rho_s_value
    rho_l_eff_avg = rho_l_value
    solid_density_weight = 0.0
    liquid_density_weight = 0.0
    phase_density_ratio = phase_volume_expansion
    core_volume_grid = 0.0
    pressure_for_density = 0.0
    density_converged = True
    density_iterations = 0
    density_rel_error = 0.0
    density_iteration_history = []
    pressure_density_iteration_mode = "not_used"
    pressure_limited_by_compressibility = False
    phase_width = max(float(Tl) - float(Ts), 1.0e-30)
    liquid_fraction_ref = float(
        np.clip((float(core_reference_temperature) - float(Ts)) / phase_width, 0.0, 1.0)
    )

    core_mask = np.asarray(mask_core_phy_t, dtype=bool)
    if np.any(core_mask):
        dA = float(dr) * float(dr)
        r_phys = np.asarray(phys_coords_t)[:, 0]
        cell_volume_node = 2.0 * np.pi * r_phys * dA
        T_core_raw = np.asarray(T_phys, dtype=np.float64)[core_mask]
        cv_core = cell_volume_node[core_mask]
        core_volume_grid += float(np.sum(cv_core))
        if rho_s_value is None or rho_l_value is None:
            liquid_fraction_node = np.clip((T_core_raw - float(Ts)) / phase_width, 0.0, 1.0)
            phase_density_change_node = (
                (liquid_fraction_node - liquid_fraction_ref) * phase_density_ratio
            )
            thermal_density_change_node = (
                liquid_fraction_node
                * (1.0 + phase_volume_expansion)
                * 3.0
                * float(alpha_core_l)
                * (T_core_raw - float(core_reference_temperature))
            )
            terms = {
                "liquid_fraction": liquid_fraction_node,
                "phase_density_ratio": float(phase_density_ratio),
                "phase_density_change": phase_density_change_node,
                "solid_thermal_density_change": np.zeros_like(T_core_raw, dtype=np.float64),
                "liquid_thermal_density_change": (
                    liquid_fraction_node
                    * (1.0 + phase_volume_expansion)
                    * 3.0
                    * float(alpha_core_l)
                    * (T_core_raw - float(core_reference_temperature))
                ),
                "liquid_compression_density_change": np.zeros_like(T_core_raw, dtype=np.float64),
                "thermal_density_change": thermal_density_change_node,
                "total_density_change": phase_density_change_node + thermal_density_change_node,
                "rho_s_eff": None,
                "rho_l_eff": None,
                "pressure_for_density": 0.0,
                "liquid_compressibility": 0.0,
            }
        else:
            density_details = iterate_liquid_density_for_shell_pressure(
                T_core_raw,
                cv_core,
                core_volume_ref,
                shell_deltaT,
                core_reference_temperature,
                rho_s_value,
                rho_l_value,
                alpha_core_l,
                Ts,
                Tl,
                E_shell,
                nu_shell,
                outer_radius,
                inner_radius,
                alpha_shell,
                liquid_compressibility,
                density_rel_tol=density_rel_tol,
                max_iter=density_iter_max,
            )
            liquid_volume_equiv = float(density_details["liquid_volume_equiv"])
            density_change_total = float(density_details["density_change_total"])
            phase_density_change = float(density_details["phase_density_change"])
            thermal_density_change = float(density_details["thermal_density_change"])
            solid_thermal_density_change = float(density_details["solid_thermal_density_change"])
            liquid_thermal_density_change = float(density_details["liquid_thermal_density_change"])
            liquid_compression_density_change = float(
                density_details["liquid_compression_density_change"]
            )
            solid_density_weight = float(density_details["solid_density_weight"])
            liquid_density_weight = float(density_details["liquid_density_weight"])
            phase_density_ratio = float(density_details["phase_density_ratio"])
            pressure_for_density = float(density_details["pressure_for_density"])
            delta_rho_s_avg = float(density_details["delta_rho_s_avg"])
            delta_rho_l_avg = float(density_details["delta_rho_l_avg"])
            rho_s_eff_avg = float(density_details["rho_s_eff_avg"])
            rho_l_eff_avg = float(density_details["rho_l_eff_avg"])
            density_converged = bool(density_details["density_converged"])
            density_iterations = int(density_details["density_iterations"])
            density_rel_error = float(density_details["density_rel_error"])
            density_iteration_history = density_details["density_iteration_history"]
            pressure_density_iteration_mode = density_details["pressure_density_iteration_mode"]
            pressure_limited_by_compressibility = bool(
                density_details["pressure_limited_by_compressibility"]
            )

            terms = None

        if rho_s_value is None or rho_l_value is None:
            liquid_fraction_node = terms["liquid_fraction"]
            liquid_volume_equiv = float(np.sum(liquid_fraction_node * cv_core))
            core_volume_ref = max(float(core_volume_ref), 1.0e-30)
            density_change_total = float(np.sum(terms["total_density_change"] * cv_core) / core_volume_ref)
            phase_density_change = float(np.sum(terms["phase_density_change"] * cv_core) / core_volume_ref)
            thermal_density_change = float(
                np.sum(terms["thermal_density_change"] * cv_core) / core_volume_ref
            )
            solid_thermal_density_change = float(
                np.sum(terms["solid_thermal_density_change"] * cv_core) / core_volume_ref
            )
            liquid_thermal_density_change = float(
                np.sum(terms["liquid_thermal_density_change"] * cv_core) / core_volume_ref
            )
            liquid_compression_density_change = float(
                np.sum(terms["liquid_compression_density_change"] * cv_core) / core_volume_ref
            )
            solid_density_weight = float(np.sum((1.0 - liquid_fraction_node) * cv_core))
            liquid_density_weight = float(np.sum(liquid_fraction_node * cv_core))

    core_volume_ref = max(float(core_volume_ref), 1.0e-30)
    deltaV_phase = phase_density_change * core_volume_ref
    deltaV_thermal = thermal_density_change * core_volume_ref
    deltaV_compression = liquid_compression_density_change * core_volume_ref
    deltaV_total = density_change_total * core_volume_ref
    total_relative_deltaV = density_change_total
    phase_relative_deltaV = phase_density_change
    thermal_relative_deltaV = thermal_density_change
    effective_deltaV = deltaV_total
    effective_relative_deltaV = density_change_total
    pcm_growth = np.cbrt(max(0.0, 1.0 + effective_relative_deltaV))
    pressure_formula = shell_pressure_from_relative_volume_change(
        E_shell,
        nu_shell,
        outer_radius,
        inner_radius,
        density_change_total,
        alpha=alpha_shell,
        deltaT=shell_deltaT,
    )
    pressure = float(pressure_formula)

    return {
        "deltaV_total": float(deltaV_total),
        "deltaV_phase": float(deltaV_phase),
        "deltaV_thermal": float(deltaV_thermal),
        "deltaV_mushy_thermal": 0.0,
        "deltaV_liquid_postphase_thermal": 0.0,
        "deltaV_liquid_thermal": 0.0,
        "deltaV_compression": float(deltaV_compression),
        "melt_volume_equiv": float(liquid_volume_equiv),
        "relative_deltaV": float(total_relative_deltaV),
        "phase_relative_deltaV": float(phase_relative_deltaV),
        "thermal_relative_deltaV": float(thermal_relative_deltaV),
        "mushy_thermal_relative_deltaV": 0.0,
        "liquid_thermal_relative_deltaV": float(liquid_thermal_density_change),
        "total_relative_deltaV": float(total_relative_deltaV),
        "melt_fraction": float(liquid_volume_equiv / core_volume_ref),
        "reference_temperature": float(core_reference_temperature),
        "reference_liquid_fraction": float(liquid_fraction_ref),
        "core_volume": float(core_volume_ref),
        "core_volume_grid": float(core_volume_grid),
        "effective_deltaV": float(effective_deltaV),
        "effective_relative_deltaV": float(effective_relative_deltaV),
        "pressure_deltaV": float(effective_deltaV),
        "pressure_relative_deltaV": float(effective_relative_deltaV),
        "pressure_formula": float(pressure_formula),
        "pressure": float(pressure),
        "pcm_growth": float(pcm_growth),
        "phase_volume_expansion_fraction": float(phase_volume_expansion),
        "density_change_total": float(density_change_total),
        "density_relative_change": float(density_change_total),
        "phase_density_change": float(phase_density_change),
        "thermal_density_change": float(thermal_density_change),
        "solid_thermal_density_change": float(solid_thermal_density_change),
        "liquid_thermal_density_change": float(liquid_thermal_density_change),
        "liquid_compression_density_change": float(liquid_compression_density_change),
        "phase_density_ratio": float(phase_density_ratio),
        "shell_growth": float(1.0 + float(alpha_shell) * float(shell_deltaT)),
        "shell_deltaT": float(shell_deltaT),
        "rho_s": None if rho_s_value is None else float(rho_s_value),
        "rho_l": None if rho_l_value is None else float(rho_l_value),
        "delta_rho_s_avg": float(delta_rho_s_avg),
        "delta_rho_l_avg": float(delta_rho_l_avg),
        "rho_s_eff_avg": None if rho_s_eff_avg is None else float(rho_s_eff_avg),
        "rho_l_eff_avg": None if rho_l_eff_avg is None else float(rho_l_eff_avg),
        "solid_density_weight": float(solid_density_weight),
        "liquid_density_weight": float(liquid_density_weight),
        "pressure_for_density": float(pressure_for_density),
        "liquid_compressibility": float(liquid_compressibility),
        "density_pressure_solved": bool(float(liquid_compressibility) > 0.0),
        "density_converged": bool(density_converged),
        "density_iterations": int(density_iterations),
        "density_rel_error": float(density_rel_error),
        "density_rel_tol": float(density_rel_tol),
        "density_iteration_history": density_iteration_history,
        "pressure_density_iteration_mode": pressure_density_iteration_mode,
        "pressure_limited_by_compressibility": bool(pressure_limited_by_compressibility),
        "thermal_volume_factor": 3.0,
        "solid_thermal_expansion": float(alpha_core_s),
        "liquid_thermal_expansion": float(alpha_core_l),
        "mode": (
            "phase_and_core_thermal_density_change_with_liquid_compressibility"
            if float(liquid_compressibility) > 0.0
            else "phase_and_core_thermal_density_change"
        ),
    }


def build_inner_pressure_body_force(
        coords_all_m,
        surface_info,
        arc_correction,
        pressure_value,
        dr,
        nu_shell,
):
    N = coords_all_m.shape[0]
    br = np.zeros(N, dtype=np.float64)
    bz = np.zeros(N, dtype=np.float64)
    idx_local = surface_info["indices"]
    if idx_local.size == 0 or pressure_value == 0.0:
        return br, bz

    p_body = pressure_value / dr
    r_node_surface = coords_all_m[idx_local, 0]
    radial_cell_factor = r_node_surface / (r_node_surface + 0.5 * dr)
    poisson_factor = (1.0 - nu_shell) / (1.0 + nu_shell)
    br[idx_local] = (
        p_body * arc_correction["factor_r"] * poisson_factor * radial_cell_factor
    )
    bz[idx_local] = p_body * arc_correction["factor_z"] * poisson_factor
    return br, bz

@njit(parallel=True, fastmath=True)
def _compute_core_shell_bond_force_to_shell_axisym_csr_numba(
        indptr, indices, area_edge,
        shape_edge,
        eij_edge,
        n_r_edge, n_z_edge,
        crack,
        lamda_edge, miu_edge,
        r_node,
        z_node,
        center_r,
        center_z,
        Ur,
        dilation,
        T_m, Tpre_avg,
        kprime_edge,
        csr,
        mask_core,
        shell_source_node_mask,
        delta,
        eps=1e-30,
):
    N = Ur.size
    br_transfer = np.zeros(N, dtype=np.float64)
    bz_transfer = np.zeros(N, dtype=np.float64)

    coeff1 = 3.0 / (np.pi * (delta ** 3))
    coeff2 = 12.0 / (np.pi * (delta ** 3))

    for i in prange(N):
        if mask_core[i] or not shell_source_node_mask[i]:
            continue

        ri = r_node[i]
        inv_ri = 1.0 / (ri + eps)
        theta_i = dilation[i]
        Uri = Ur[i]
        dTi = T_m[i] - Tpre_avg

        Fi_r = 0.0
        Fi_z = 0.0

        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j == i:
                continue
            if not mask_core[j]:
                continue
            if crack[p] == 0 or area_edge[p] <= eps:
                continue

            theta_j = dilation[j]
            dTj = T_m[j] - Tpre_avg

            lam_e = lamda_edge[p]
            mu_e = miu_edge[p]
            kp_e = kprime_edge[p]

            stretch = eij_edge[p]
            ti = lam_e * theta_i + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTi
            tj = lam_e * theta_j + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTj

            base_i = coeff1 * ti + coeff2 * mu_e * stretch
            base_j = coeff1 * tj + coeff2 * mu_e * stretch

            s_i = shape_edge[p]
            dforce_i = area_edge[p] * base_i * s_i * csr[p]
            dforce_j = area_edge[p] * base_j * s_i * csr[p]
            dforce = dforce_i + dforce_j

            Fi_r += n_r_edge[p] * dforce * 0.5
            Fi_z += n_z_edge[p] * dforce * 0.5

        br_transfer[i] = Fi_r
        bz_transfer[i] = Fi_z

    return br_transfer, bz_transfer


def compute_core_shell_bond_force_to_shell_axisym_csr(
        indptr, indices, area_edge,
        shape_edge,
        eij_edge,
        n_r_edge, n_z_edge,
        crack,
        lamda_edge, miu_edge,
        r_node,
        z_node,
        center_r,
        center_z,
        Ur,
        dilation,
        T_m, Tpre_avg,
        kprime_edge,
        csr,
        mask_core,
        delta,
        eps=1e-30,
        shell_source_node_mask=None,
):
    mask_core_arr = np.asarray(mask_core, dtype=np.bool_)
    if shell_source_node_mask is None:
        shell_source_node_mask_arr = ~mask_core_arr
    else:
        shell_source_node_mask_arr = np.asarray(shell_source_node_mask, dtype=np.bool_)
        if shell_source_node_mask_arr.shape[0] != mask_core_arr.shape[0]:
            raise ValueError("shell_source_node_mask must have the same length as mask_core")
        shell_source_node_mask_arr = shell_source_node_mask_arr & (~mask_core_arr)

    return _compute_core_shell_bond_force_to_shell_axisym_csr_numba(
        indptr, indices, area_edge,
        shape_edge,
        eij_edge,
        n_r_edge, n_z_edge,
        crack,
        lamda_edge, miu_edge,
        r_node,
        z_node,
        center_r,
        center_z,
        Ur,
        dilation,
        T_m, Tpre_avg,
        kprime_edge,
        csr,
        mask_core_arr,
        shell_source_node_mask_arr,
        delta,
        eps,
    )


def compute_core_shell_interface_force_flux_pressure(
        indptr,
        indices,
        area_edge,
        shape_edge,
        eij_edge,
        n_r_edge,
        n_z_edge,
        crack,
        lamda_edge,
        miu_edge,
        r_node,
        Ur,
        dilation,
        T_m,
        Tpre_avg,
        kprime_edge,
        csr,
        mask_core,
        surface_node_mask,
        surface_normal,
        surface_arc_length,
        dr,
        delta,
        eps=1e-30,
):
    """
    Axisymmetric discrete force-flux pressure across the core-shell interface.

    This is the shell-core specialization of the paper's Eq. (53)/(60):
    sum pairwise forces carried by bonds crossing the inner boundary and divide
    their normal resultant by the represented axisymmetric surface area.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    area_edge = np.asarray(area_edge, dtype=np.float64)
    shape_edge = np.asarray(shape_edge, dtype=np.float64)
    eij_edge = np.asarray(eij_edge, dtype=np.float64)
    n_r_edge = np.asarray(n_r_edge, dtype=np.float64)
    n_z_edge = np.asarray(n_z_edge, dtype=np.float64)
    crack = np.asarray(crack, dtype=np.int8)
    lamda_edge = np.asarray(lamda_edge, dtype=np.float64)
    miu_edge = np.asarray(miu_edge, dtype=np.float64)
    r_node = np.asarray(r_node, dtype=np.float64)
    Ur = np.asarray(Ur, dtype=np.float64)
    dilation = np.asarray(dilation, dtype=np.float64)
    T_m = np.asarray(T_m, dtype=np.float64)
    kprime_edge = np.asarray(kprime_edge, dtype=np.float64)
    csr = np.asarray(csr, dtype=np.float64)
    mask_core = np.asarray(mask_core, dtype=bool)
    surface_node_mask = np.asarray(surface_node_mask, dtype=bool)
    surface_normal = np.asarray(surface_normal, dtype=np.float64)
    surface_arc_length = np.asarray(surface_arc_length, dtype=np.float64)

    N = indptr.size - 1
    if r_node.shape[0] != N or Ur.shape[0] != N or dilation.shape[0] != N or T_m.shape[0] != N:
        raise ValueError("node arrays must have CSR node length")
    if mask_core.shape[0] != N or surface_node_mask.shape[0] != N:
        raise ValueError("node masks must have CSR node length")
    if surface_normal.shape != (N, 2):
        raise ValueError("surface_normal must have shape (N, 2)")
    if surface_arc_length.shape[0] != N:
        raise ValueError("surface_arc_length must have CSR node length")
    if not (
            indices.shape == area_edge.shape == shape_edge.shape == eij_edge.shape
            == n_r_edge.shape == n_z_edge.shape == crack.shape == lamda_edge.shape
            == miu_edge.shape == kprime_edge.shape == csr.shape
    ):
        raise ValueError("edge arrays must have the same CSR nnz length")

    dr = float(dr)
    delta = float(delta)
    if dr <= eps or delta <= eps:
        raise ValueError("dr and delta must be positive")

    coeff1 = 3.0 / (np.pi * (delta ** 3))
    coeff2 = 12.0 / (np.pi * (delta ** 3))

    total_force_r = 0.0
    total_force_z = 0.0
    normal_force = 0.0
    represented_area = 0.0
    surface_nodes = 0
    contributing_bonds = 0
    skipped_wrong_side = 0

    for i in np.where(surface_node_mask & (~mask_core))[0]:
        arc_i = float(surface_arc_length[i])
        if not np.isfinite(arc_i) or arc_i <= eps:
            continue

        normal_i = surface_normal[i]
        normal_len = float(np.sqrt(normal_i[0] * normal_i[0] + normal_i[1] * normal_i[1]))
        if not np.isfinite(normal_len) or normal_len <= eps:
            continue
        nr_i = float(normal_i[0] / normal_len)
        nz_i = float(normal_i[1] / normal_len)

        # The shell node is centered roughly half a cell outside the inner face.
        # Shift back along the pressure normal to estimate the revolved boundary area.
        r_face = max(float(r_node[i]) - 0.5 * dr * nr_i, 0.0)
        dA_i = 2.0 * np.pi * r_face * arc_i
        if not np.isfinite(dA_i) or dA_i <= eps:
            continue

        cell_volume_i = 2.0 * np.pi * max(float(r_node[i]), 0.0) * dr * dr
        if not np.isfinite(cell_volume_i) or cell_volume_i <= eps:
            continue

        surface_nodes += 1
        represented_area += dA_i

        ri = float(r_node[i])
        inv_ri = 1.0 / (ri + eps)
        theta_i = float(dilation[i])
        Uri = float(Ur[i])
        dTi = float(T_m[i] - Tpre_avg)

        for p in range(indptr[i], indptr[i + 1]):
            j = int(indices[p])
            if j == i or not mask_core[j]:
                continue
            if crack[p] == 0 or area_edge[p] <= eps:
                continue

            # Eq. (53) counts only the orientation satisfying m_jk . n > 0.
            # CSR rows here are shell -> core, so the crossing orientation is -n_edge.
            m_dot_n = -(float(n_r_edge[p]) * nr_i + float(n_z_edge[p]) * nz_i)
            if m_dot_n <= eps:
                skipped_wrong_side += 1
                continue

            theta_j = float(dilation[j])
            dTj = float(T_m[j] - Tpre_avg)
            lam_e = float(lamda_edge[p])
            mu_e = float(miu_edge[p])
            kp_e = float(kprime_edge[p])
            stretch = float(eij_edge[p])

            ti = lam_e * theta_i + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTi
            tj = lam_e * theta_j + (lam_e + mu_e) * (Uri * inv_ri) - kp_e * dTj

            base_i = coeff1 * ti + coeff2 * mu_e * stretch
            base_j = coeff1 * tj + coeff2 * mu_e * stretch
            # The m_dot_n test above already restricts the traction integral
            # to one bond orientation. The 1/2 in the continuum force-flux
            # formula is only needed when both +m and -m are integrated.
            dforce_density = (
                float(area_edge[p])
                * (base_i + base_j)
                * float(shape_edge[p])
                * float(csr[p])
            )

            force_r = float(n_r_edge[p]) * dforce_density * cell_volume_i
            force_z = float(n_z_edge[p]) * dforce_density * cell_volume_i
            total_force_r += force_r
            total_force_z += force_z
            normal_force += force_r * nr_i + force_z * nz_i
            contributing_bonds += 1

    pressure = 0.0 if represented_area <= eps else normal_force / represented_area
    flux_r = 0.0 if represented_area <= eps else total_force_r / represented_area
    flux_z = 0.0 if represented_area <= eps else total_force_z / represented_area

    return float(pressure), {
        "pressure": float(pressure),
        "flux_r": float(flux_r),
        "flux_z": float(flux_z),
        "normal_force": float(normal_force),
        "resultant_force_r": float(total_force_r),
        "resultant_force_z": float(total_force_z),
        "represented_area": float(represented_area),
        "surface_nodes": int(surface_nodes),
        "contributing_bonds": int(contributing_bonds),
        "skipped_wrong_side_bonds": int(skipped_wrong_side),
        "pressure_sign": "positive_along_inner_surface_normal",
        "area_model": "axisymmetric_surface_area_from_inner_arc_lengths",
        "direction_domain": "single_orientation_m_dot_n_positive",
        "direction_domain_half_factor_applied": False,
        "bond_force_model": "single_orientation_force_state_difference",
    }

def build_displacement_target_equivalent_body_force(
        target_Ur,
        target_Uz,
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
        T_shell_only_m,
        Tpre_avg,
        kprime_edge_shell_only,
        alpha_edge_shell_only,
        csr_shell_only,
        axis_mask_shell_only=None,
        damage_bond_mask_shell_only=None,
        damage_threshold=0.5,
        enable_cracking=False,
        use_crack_in_dilation=True,
        n_phys=None,
        eps=1e-30,
):
    target_Ur = np.asarray(target_Ur, dtype=np.float64).copy()
    target_Uz = np.asarray(target_Uz, dtype=np.float64).copy()
    N = csr_indptr_shell_only_m.size - 1
    if target_Ur.shape[0] != N or target_Uz.shape[0] != N:
        raise ValueError("target displacement arrays must have shell-only node length")

    if axis_mask_shell_only is not None:
        axis_mask = np.asarray(axis_mask_shell_only, dtype=bool)
        if axis_mask.shape[0] != N:
            raise ValueError("axis_mask_shell_only must have shell-only node length")
        target_Ur[axis_mask] = 0.0
    else:
        axis_mask = None

    crack_work = np.asarray(crack_shell_only, dtype=np.int8).copy()
    if use_crack_in_dilation:
        dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr_rows_crack_numba(
            csr_indptr_shell_only_m,
            csr_indices_shell_only_m,
            r_flat_shell_only,
            z_flat_shell_only,
            target_Ur,
            target_Uz,
            csr_dist_shell_only_m,
            csr_area_shell_only_m,
            shape_edge_shell_only_m,
            coeff,
            csr_shell_only,
            crack_work,
        )
    else:
        dilation, eij_edge, n_r_edge, n_z_edge = pfc.compute_dilation_axisym_csr_rows_numba(
            csr_indptr_shell_only_m,
            csr_indices_shell_only_m,
            r_flat_shell_only,
            z_flat_shell_only,
            target_Ur,
            target_Uz,
            csr_dist_shell_only_m,
            csr_area_shell_only_m,
            shape_edge_shell_only_m,
            coeff,
            csr_shell_only,
        )

    if damage_bond_mask_shell_only is None:
        damage_node = np.zeros(N, dtype=np.float64)
    else:
        damage_node = mt.compute_damage_variable_axisym_csr(
            csr_indptr_shell_only_m,
            csr_indices_shell_only_m,
            csr_area_shell_only_m,
            shape_edge_shell_only_m,
            csr_shell_only,
            crack_work,
            np.asarray(damage_bond_mask_shell_only, dtype=bool),
        )

    zero_br = np.zeros(N, dtype=np.float64)
    zero_bz = np.zeros(N, dtype=np.float64)
    internal_Fr, internal_Fz = mt.compute_accel_osbpd_axisym_csr_inplace_mu(
        csr_indptr_shell_only_m,
        csr_indices_shell_only_m,
        csr_area_shell_only_m,
        shape_edge_shell_only_m,
        eij_edge,
        n_r_edge,
        n_z_edge,
        crack_work,
        s0_edge_shell_only,
        lamda_edge_shell_only,
        miu_edge_shell_only,
        r_flat_shell_only,
        target_Ur,
        dilation,
        zero_br,
        zero_bz,
        delta_i,
        T_shell_only_m,
        Tpre_avg,
        kprime_edge_shell_only,
        alpha_edge_shell_only,
        csr_shell_only,
        damage_node,
        0.0,
        damage_threshold,
        False,
        enable_cracking,
        eps,
    )

    br_equiv = -internal_Fr
    bz_equiv = -internal_Fz
    if axis_mask is not None:
        br_equiv[axis_mask] = 0.0

    n = N if n_phys is None else min(int(n_phys), N)
    force_norm = float(np.sqrt(np.sum(internal_Fr[:n] ** 2 + internal_Fz[:n] ** 2)))
    body_force_norm = float(np.sqrt(np.sum(br_equiv[:n] ** 2 + bz_equiv[:n] ** 2)))
    target_norm = float(np.sqrt(np.sum(target_Ur[:n] ** 2 + target_Uz[:n] ** 2)))
    support_mask = (
        np.isfinite(br_equiv[:n])
        & np.isfinite(bz_equiv[:n])
        & ((np.abs(br_equiv[:n]) + np.abs(bz_equiv[:n])) > eps)
    )

    return br_equiv, bz_equiv, {
        "mode": "displacement_target_residual_body_force",
        "target_nodes": int(n),
        "support_nodes": int(np.count_nonzero(support_mask)),
        "internal_force_norm": force_norm,
        "equivalent_body_force_norm": body_force_norm,
        "target_displacement_norm": target_norm,
        "max_abs_equivalent_body_force": (
            float(np.max(np.sqrt(br_equiv[:n] ** 2 + bz_equiv[:n] ** 2)))
            if n > 0
            else 0.0
        ),
        "crack_changed": bool(np.any(crack_work != np.asarray(crack_shell_only, dtype=np.int8))),
    }, crack_work


def integrate_axisym_body_force_resultant(
        body_force_br,
        body_force_bz,
        coords_all_m,
        dr,
        n_phys=None,
        mask=None,
        eps=1e-30,
):
    br = np.asarray(body_force_br, dtype=np.float64)
    bz = np.asarray(body_force_bz, dtype=np.float64)
    coords = np.asarray(coords_all_m, dtype=np.float64)
    if br.shape != bz.shape:
        raise ValueError("body force arrays must have the same shape")
    if coords.ndim != 2 or coords.shape[0] < br.shape[0] or coords.shape[1] < 2:
        raise ValueError("coords_all_m must contain coordinates for every body force node")

    n = br.size if n_phys is None else min(int(n_phys), br.size)
    selected = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)[:n]
    if selected.shape[0] != n:
        raise ValueError("mask must have at least n_phys entries")

    cell_volume = 2.0 * np.pi * np.maximum(coords[:n, 0], 0.0) * float(dr) * float(dr)
    finite = (
        selected
        & np.isfinite(br[:n])
        & np.isfinite(bz[:n])
        & np.isfinite(cell_volume)
        & (cell_volume > eps)
    )
    support = finite & ((np.abs(br[:n]) + np.abs(bz[:n])) > eps)
    force_r = br[:n] * cell_volume
    force_z = bz[:n] * cell_volume
    resultant_r = float(np.sum(force_r[finite]))
    resultant_z = float(np.sum(force_z[finite]))
    resultant_mag = float(np.sqrt(resultant_r * resultant_r + resultant_z * resultant_z))

    return {
        "resultant_r": resultant_r,
        "resultant_z": resultant_z,
        "resultant_magnitude": resultant_mag,
        "selected_nodes": int(np.count_nonzero(finite)),
        "support_nodes": int(np.count_nonzero(support)),
        "cell_volume_sum": float(np.sum(cell_volume[finite])),
    }


def project_transfer_force_to_uniform_pressure(
        raw_shell_transfer_br,
        raw_shell_transfer_bz,
        unit_pressure_br,
        unit_pressure_bz,
        coords_all_m=None,
        dr=None,
        center_r=0.0,
        center_z=0.0,
        inner_axis_x=None,
        inner_axis_z=None,
        n_phys=None,
        eps=1e-30,
):
    raw_br = np.asarray(raw_shell_transfer_br, dtype=np.float64)
    raw_bz = np.asarray(raw_shell_transfer_bz, dtype=np.float64)
    unit_br = np.asarray(unit_pressure_br, dtype=np.float64)
    unit_bz = np.asarray(unit_pressure_bz, dtype=np.float64)
    if raw_br.shape != raw_bz.shape or raw_br.shape != unit_br.shape or raw_br.shape != unit_bz.shape:
        raise ValueError("raw and unit pressure body-force arrays must have the same shape")

    n = raw_br.size if n_phys is None else min(int(n_phys), raw_br.size)
    if coords_all_m is not None and dr is not None:
        coords = np.asarray(coords_all_m, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < n or coords.shape[1] < 2:
            raise ValueError("coords_all_m must have at least n rows and two coordinate columns")

        coords_n = coords[:n, :2]
        cell_volume = 2.0 * np.pi * np.maximum(coords_n[:, 0], 0.0) * float(dr) * float(dr)
        finite = (
            np.isfinite(raw_br[:n])
            & np.isfinite(raw_bz[:n])
            & np.isfinite(unit_br[:n])
            & np.isfinite(unit_bz[:n])
            & np.isfinite(cell_volume)
            & (cell_volume > eps)
        )
        raw_mask = (
            finite
            & ((np.abs(raw_br[:n]) + np.abs(raw_bz[:n])) > eps)
        )
        unit_mask = (
            finite
            & ((np.abs(unit_br[:n]) + np.abs(unit_bz[:n])) > eps)
        )
        raw_resultant_r = float(np.sum(raw_br[:n][finite] * cell_volume[finite]))
        raw_resultant_z = float(np.sum(raw_bz[:n][finite] * cell_volume[finite]))
        unit_resultant_r = float(np.sum(unit_br[:n][finite] * cell_volume[finite]))
        unit_resultant_z = float(np.sum(unit_bz[:n][finite] * cell_volume[finite]))
        numerator = raw_resultant_r * unit_resultant_r + raw_resultant_z * unit_resultant_z
        denominator = unit_resultant_r * unit_resultant_r + unit_resultant_z * unit_resultant_z
        if abs(denominator) <= eps:
            pressure = 0.0
        else:
            pressure = numerator / denominator

        residual_br = raw_br[:n][finite] - pressure * unit_br[:n][finite]
        residual_bz = raw_bz[:n][finite] - pressure * unit_bz[:n][finite]
        raw_force_r = raw_br[:n][finite] * cell_volume[finite]
        raw_force_z = raw_bz[:n][finite] * cell_volume[finite]
        residual_force_r = residual_br * cell_volume[finite]
        residual_force_z = residual_bz * cell_volume[finite]
        raw_norm = float(np.sqrt(np.sum(raw_force_r * raw_force_r + raw_force_z * raw_force_z)))
        residual_norm = float(np.sqrt(np.sum(
            residual_force_r * residual_force_r
            + residual_force_z * residual_force_z
        )))
        relative_residual = np.nan if raw_norm <= eps else residual_norm / raw_norm
        support_nodes = int(np.count_nonzero(raw_mask))
        unit_support_nodes = int(np.count_nonzero(unit_mask))
    else:
        load_norm2 = unit_br[:n] ** 2 + unit_bz[:n] ** 2
        mask = (
            np.isfinite(raw_br[:n])
            & np.isfinite(raw_bz[:n])
            & np.isfinite(unit_br[:n])
            & np.isfinite(unit_bz[:n])
            & (load_norm2 > eps)
        )
        if not np.any(mask):
            return 0.0, {
                "pressure": 0.0,
                "support_nodes": 0,
                "unit_support_nodes": 0,
                "numerator": 0.0,
                "denominator": 0.0,
                "raw_norm": 0.0,
                "residual_norm": 0.0,
                "relative_residual": np.nan,
            }

        rb = raw_br[:n][mask]
        rz = raw_bz[:n][mask]
        ub = unit_br[:n][mask]
        uz = unit_bz[:n][mask]
        numerator = float(np.sum(rb * ub + rz * uz))
        denominator = float(np.sum(ub * ub + uz * uz))
        pressure = numerator / max(denominator, eps)
        residual_br = rb - pressure * ub
        residual_bz = rz - pressure * uz
        raw_norm = float(np.sqrt(np.sum(rb * rb + rz * rz)))
        residual_norm = float(np.sqrt(np.sum(residual_br * residual_br + residual_bz * residual_bz)))
        relative_residual = np.nan if raw_norm <= eps else residual_norm / raw_norm
        support_nodes = int(np.count_nonzero(mask))
        unit_support_nodes = support_nodes

    return float(pressure), {
        "pressure": float(pressure),
        "support_nodes": support_nodes,
        "unit_support_nodes": unit_support_nodes,
        "numerator": numerator,
        "denominator": denominator,
        "raw_norm": raw_norm,
        "residual_norm": float(residual_norm),
        "relative_residual": float(relative_residual),
        "raw_resultant_r": float(raw_resultant_r) if coords_all_m is not None and dr is not None else np.nan,
        "raw_resultant_z": float(raw_resultant_z) if coords_all_m is not None and dr is not None else np.nan,
        "unit_resultant_r": float(unit_resultant_r) if coords_all_m is not None and dr is not None else np.nan,
        "unit_resultant_z": float(unit_resultant_z) if coords_all_m is not None and dr is not None else np.nan,
        "signed_pressure_projection": True,
    }


def solve_shell_body_force_equilibrium(
        Ur_start,
        Uz_start,
        crack_start,
        br_load,
        bz_load,
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
        T_shell_only_m,
        Tpre_avg,
        kprime_edge_shell_only,
        alpha_edge_shell_only,
        csr_shell_only,
        axis_mask_shell_only,
        damage_bond_mask_shell_only,
        nsteps_m,
        shell_only_rms,
        enable_cracking,
        damage_threshold,
        crack_pressure=0.0,
        enable_crack_pressure_force=False,
        log_prefix="[Switch match mech]",
        log_label=None,
        min_steps=0,
):
    min_steps = max(0, int(min_steps))
    nsteps_m = max(int(nsteps_m), min_steps)
    trial_Ur = Ur_start.copy()
    trial_Uz = Uz_start.copy()
    trial_Fr_0 = np.zeros_like(trial_Ur)
    trial_Fz_0 = np.zeros_like(trial_Uz)
    trial_Vr_half = np.zeros_like(trial_Ur)
    trial_Vz_half = np.zeros_like(trial_Uz)
    trial_crack = crack_start.copy()
    trial_damage_phi = np.zeros_like(trial_Ur)
    trial_converged_step = nsteps_m - 1
    trial_rms = np.nan
    label_text = "" if log_label is None else f" {log_label}"

    for match_step in range(nsteps_m):
        trial_Ur_prev = trial_Ur.copy()
        trial_Uz_prev = trial_Uz.copy()
        trial_damage_phi = mt.compute_damage_variable_axisym_csr(
            csr_indptr_shell_only_m,
            csr_indices_shell_only_m,
            csr_area_shell_only_m,
            shape_edge_shell_only_m,
            csr_shell_only,
            trial_crack,
            damage_bond_mask_shell_only,
        )
        trial_Ur, trial_Uz, trial_Fr_0, trial_Fz_0, \
            trial_Vr_half, trial_Vz_half = mt.compute_mechanical_step_csr(
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
                trial_crack,
                s0_edge_shell_only,
                trial_Ur,
                trial_Uz,
                br_load,
                bz_load,
                trial_Fr_0,
                trial_Fz_0,
                trial_Vr_half,
                trial_Vz_half,
                lambda_diag_shell_only,
                dt_ADR,
                delta_i,
                T_shell_only_m,
                Tpre_avg,
                kprime_edge_shell_only,
                alpha_edge_shell_only,
                csr_shell_only,
                enable_cracking=enable_cracking,
                damage_node=trial_damage_phi,
                crack_pressure=crack_pressure,
                damage_threshold=damage_threshold,
                enable_crack_pressure_force=enable_crack_pressure_force,
                use_crack_in_dilation=True,
            )
        trial_Fr_0 = trial_Fr_0.copy()
        trial_Fz_0 = trial_Fz_0.copy()
        trial_Ur[axis_mask_shell_only] = 0.0
        trial_rms = np.sqrt(
            np.mean(
                (trial_Ur - trial_Ur_prev) ** 2
                + (trial_Uz - trial_Uz_prev) ** 2
            )
        )
        trial_converged_step = match_step
        steps_done = match_step + 1
        if trial_rms < shell_only_rms and steps_done >= min_steps:
            print(
                f"{log_prefix}{label_text}: converged at step {match_step} "
                f"with RMS {trial_rms:.3e}",
                flush=True,
            )
            break
        if match_step > 0 and match_step % 10 == 0:
            print(
                f"{log_prefix}{label_text}: step {match_step}, RMS {trial_rms:.3e}",
                flush=True,
            )

    trial_damage_phi = mt.compute_damage_variable_axisym_csr(
        csr_indptr_shell_only_m,
        csr_indices_shell_only_m,
        csr_area_shell_only_m,
        shape_edge_shell_only_m,
        csr_shell_only,
        trial_crack,
        damage_bond_mask_shell_only,
    )

    return {
        "Ur": trial_Ur,
        "Uz": trial_Uz,
        "Fr_0": trial_Fr_0,
        "Fz_0": trial_Fz_0,
        "Vr_half": trial_Vr_half,
        "Vz_half": trial_Vz_half,
        "crack": trial_crack,
        "damage_phi": trial_damage_phi,
        "converged_step": int(trial_converged_step),
        "rms": float(trial_rms),
    }


def match_shell_transfer_body_force(
        raw_shell_transfer_br,
        raw_shell_transfer_bz,
        Ur_shell_start,
        Uz_shell_start,
        crack_shell_only,
        shell_switch_ur_reference,
        shell_switch_uz_reference,
        core_Ur,
        core_Uz,
        phys_coords_shell_only_m,
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
        T_shell_only_m,
        Tpre_avg,
        kprime_edge_shell_only,
        alpha_edge_shell_only,
        csr_shell_only,
        axis_mask_shell_only,
        damage_bond_mask_shell_only,
        state_displacement_kwargs,
        nsteps_m,
        shell_transfer_match_iterations,
        shell_transfer_match_relaxation,
        shell_transfer_match_rel_tol,
        shell_only_rms,
        enable_cracking,
        damage_threshold,
):
    if (
            shell_switch_ur_reference is None
            or shell_switch_ur_reference.get("local_idx") is None
    ):
        raise RuntimeError("cannot switch shell model: (rshell, r) point is unavailable")
    if (
            shell_switch_uz_reference is None
            or shell_switch_uz_reference.get("local_idx") is None
    ):
        raise RuntimeError("cannot switch shell model: (0, 2r-dshell) point is unavailable")

    target_ur, _ = mt.tracked_displacement_components(
        shell_switch_ur_reference,
        [core_Ur],
        [core_Uz],
    )
    _, target_uz = mt.tracked_displacement_components(
        shell_switch_uz_reference,
        [core_Ur],
        [core_Uz],
    )
    ur_reference_coord = np.asarray(
        shell_switch_ur_reference["coord"],
        dtype=np.float64,
    )
    uz_reference_coord = np.asarray(
        shell_switch_uz_reference["coord"],
        dtype=np.float64,
    )

    shell_match_state = {"phys_coords_list_m": [phys_coords_shell_only_m]}

    transfer_scale = 1.0
    accepted_scale_r = transfer_scale
    accepted_scale_z = transfer_scale
    accepted_trial_state = None
    initial_ur = np.nan
    initial_uz = np.nan
    final_ur = np.nan
    final_uz = np.nan
    final_error_ur = np.nan
    final_error_uz = np.nan
    accepted_ur = np.nan
    accepted_uz = np.nan
    accepted_error_ur = np.nan
    accepted_error_uz = np.nan
    accepted_error_norm = np.inf
    scale_history = []
    scale_r_history = []
    scale_z_history = []
    next_scale_r_history = []
    next_scale_z_history = []
    ur_history = []
    uz_history = []
    error_ur_history = []
    error_uz_history = []
    error_norm_history = []
    iterations_done = 0
    max_relative_error_history = []
    ur_scale = max(abs(float(target_ur)), 1.0e-30)
    uz_scale = max(abs(float(target_uz)), 1.0e-30)
    accepted_max_relative_error = np.inf

    for iteration in range(shell_transfer_match_iterations):
        tested_scale = float(transfer_scale)
        tested_scale_r = tested_scale
        tested_scale_z = tested_scale
        trial_br = tested_scale * raw_shell_transfer_br
        trial_bz = tested_scale * raw_shell_transfer_bz
        trial_state = solve_shell_body_force_equilibrium(
            Ur_shell_start,
            Uz_shell_start,
            crack_shell_only,
            trial_br,
            trial_bz,
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
            T_shell_only_m,
            Tpre_avg,
            kprime_edge_shell_only,
            alpha_edge_shell_only,
            csr_shell_only,
            axis_mask_shell_only,
            damage_bond_mask_shell_only,
            nsteps_m,
            shell_only_rms,
            enable_cracking,
            damage_threshold,
            crack_pressure=0.0,
            enable_crack_pressure_force=False,
            log_label=f"iter {iteration + 1}",
        )
        trial_Ur = trial_state["Ur"]
        trial_Uz = trial_state["Uz"]

        trial_ur, _, _ = mt.state_displacement_components_at_point(
            shell_match_state,
            [trial_Ur],
            [trial_Uz],
            shell_switch_ur_reference,
            **state_displacement_kwargs,
        )
        _, trial_uz, _ = mt.state_displacement_components_at_point(
            shell_match_state,
            [trial_Ur],
            [trial_Uz],
            shell_switch_uz_reference,
            **state_displacement_kwargs,
        )
        if iteration == 0:
            initial_ur = trial_ur
            initial_uz = trial_uz
        trial_error_ur = target_ur - trial_ur
        trial_error_uz = target_uz - trial_uz
        relative_error_ur = abs(trial_error_ur) / ur_scale
        relative_error_uz = abs(trial_error_uz) / uz_scale
        max_relative_error = max(relative_error_ur, relative_error_uz)
        error_norm = float(np.sqrt((trial_error_ur / ur_scale) ** 2 + (trial_error_uz / uz_scale) ** 2))
        if np.isfinite(max_relative_error) and max_relative_error <= accepted_max_relative_error:
            accepted_max_relative_error = max_relative_error
            accepted_error_norm = error_norm
            accepted_scale_r = tested_scale_r
            accepted_scale_z = tested_scale_z
            accepted_trial_state = trial_state
            accepted_ur = trial_ur
            accepted_uz = trial_uz
            accepted_error_ur = trial_error_ur
            accepted_error_uz = trial_error_uz
        iterations_done = iteration + 1
        scale_history.append(float(0.5 * (tested_scale_r + tested_scale_z)))
        scale_r_history.append(float(tested_scale_r))
        scale_z_history.append(float(tested_scale_z))
        ur_history.append(float(trial_ur))
        uz_history.append(float(trial_uz))
        error_ur_history.append(float(trial_error_ur))
        error_uz_history.append(float(trial_error_uz))
        error_norm_history.append(error_norm)
        max_relative_error_history.append(float(max_relative_error))

        next_scale = tested_scale
        normalized_target_ur = target_ur / ur_scale
        normalized_target_uz = target_uz / uz_scale
        normalized_trial_ur = trial_ur / ur_scale
        normalized_trial_uz = trial_uz / uz_scale
        ratio_denominator = (
            normalized_trial_ur * normalized_trial_ur
            + normalized_trial_uz * normalized_trial_uz
        )
        if (
                ratio_denominator > 1.0e-30
                and np.isfinite(ratio_denominator)
                and np.isfinite(normalized_trial_ur)
                and np.isfinite(normalized_trial_uz)
        ):
            ratio = (
                normalized_target_ur * normalized_trial_ur
                + normalized_target_uz * normalized_trial_uz
            ) / ratio_denominator
            if np.isfinite(ratio):
                next_scale = tested_scale * (
                    1.0 + shell_transfer_match_relaxation * (ratio - 1.0)
                )
                next_scale = max(0.0, next_scale)

        next_scale_r = next_scale
        next_scale_z = next_scale
        next_scale_r_history.append(float(next_scale) if np.isfinite(next_scale) else np.nan)
        next_scale_z_history.append(float(next_scale) if np.isfinite(next_scale) else np.nan)
        print(
            f"[Switch match] iter {iterations_done}: "
            f"mode=two_point_component_uniform_scale, "
            f"scale={tested_scale:.6e}, "
            f"scale_r={tested_scale_r:.6e}, "
            f"scale_z={tested_scale_z:.6e}, "
            f"pressure_load={0.0:.6e}, "
            f"next_scale={next_scale:.6e}, "
            f"next_scale_r={next_scale_r:.6e}, "
            f"next_scale_z={next_scale_z:.6e}, "
            f"mech_step={trial_state['converged_step']}, "
            f"mech_rms={trial_state['rms']:.3e}, "
            f"match_error_norm={error_norm:.3e}, "
            f"max_relative_error={max_relative_error:.3e}, "
            f"Ur_after={trial_ur:.6e}, "
            f"Ur_before={target_ur:.6e}, "
            f"Ur_before_minus_after={trial_error_ur:.6e}, "
            f"Uz_after={trial_uz:.6e}, "
            f"Uz_before={target_uz:.6e}, "
            f"Uz_before_minus_after={trial_error_uz:.6e}",
            flush=True,
        )
        if not (np.isfinite(trial_ur) and np.isfinite(trial_uz)):
            break
        reached_min_iterations = iterations_done >= shell_transfer_match_iterations
        if reached_min_iterations and max_relative_error <= shell_transfer_match_rel_tol:
            break

        if not np.isfinite(next_scale):
            break
        if reached_min_iterations and any(
                abs(next_scale - previous_scale)
                <= 1.0e-12 * max(1.0, abs(next_scale), abs(previous_scale))
                for previous_scale in scale_history
        ):
            break
        transfer_scale = float(next_scale)

    final_ur = float(accepted_ur)
    final_uz = float(accepted_uz)
    final_error_ur = float(accepted_error_ur)
    final_error_uz = float(accepted_error_uz)
    final_scale_r = float(accepted_scale_r)
    final_scale_z = float(accepted_scale_z)
    shell_transfer_br_shell_only = final_scale_r * raw_shell_transfer_br
    shell_transfer_bz_shell_only = final_scale_z * raw_shell_transfer_bz

    match_info = {
        "iterations": int(iterations_done),
        "actual_correction_iterations": int(iterations_done),
        "match_mode": "two_reference_components_uniform_scale",
        "ur_reference_label": str(shell_switch_ur_reference["label"]),
        "uz_reference_label": str(shell_switch_uz_reference["label"]),
        "ur_reference_target_r": float(shell_switch_ur_reference["target_r"]),
        "ur_reference_target_z": float(shell_switch_ur_reference["target_z"]),
        "uz_reference_target_r": float(shell_switch_uz_reference["target_r"]),
        "uz_reference_target_z": float(shell_switch_uz_reference["target_z"]),
        "ur_reference_coord_r": float(ur_reference_coord[0]),
        "ur_reference_coord_z": float(ur_reference_coord[1]),
        "uz_reference_coord_r": float(uz_reference_coord[0]),
        "uz_reference_coord_z": float(uz_reference_coord[1]),
        "target_ur": float(target_ur),
        "target_uz": float(target_uz),
        "initial_ur": float(initial_ur),
        "initial_uz": float(initial_uz),
        "final_ur": float(final_ur),
        "final_uz": float(final_uz),
        "final_error_ur": float(final_error_ur),
        "final_error_uz": float(final_error_uz),
        "best_match_error_norm": float(accepted_error_norm),
        "best_max_relative_error": float(accepted_max_relative_error),
        "match_relative_tolerance": float(shell_transfer_match_rel_tol),
        "effective_force_factor": float(0.5 * (accepted_scale_r + accepted_scale_z)),
        "effective_force_factor_r": float(accepted_scale_r),
        "effective_force_factor_z": float(accepted_scale_z),
        "shell_only_br_force_factor": float(final_scale_r),
        "shell_only_bz_force_factor": float(final_scale_z),
        "shell_only_force_factor_ratio": (
            np.nan
            if abs(final_scale_z) <= 1.0e-30
            else float(final_scale_r / final_scale_z)
        ),
        "pressure_load": 0.0,
        "scale_history": scale_history,
        "scale_r_history": scale_r_history,
        "scale_z_history": scale_z_history,
        "next_scale_r_history": next_scale_r_history,
        "next_scale_z_history": next_scale_z_history,
        "ur_history": ur_history,
        "uz_history": uz_history,
        "error_ur_history": error_ur_history,
        "error_uz_history": error_uz_history,
        "error_norm_history": error_norm_history,
        "max_relative_error_history": max_relative_error_history,
    }

    return {
        "accepted_scale_r": final_scale_r,
        "accepted_scale_z": final_scale_z,
        "shell_transfer_br": shell_transfer_br_shell_only,
        "shell_transfer_bz": shell_transfer_bz_shell_only,
        "state": accepted_trial_state,
        "info": match_info,
    }


def plot_shell_only_inner_surface_points(
        phys_coords_shell_only_m,
        surface_info_shell_only,
        outer_surface_node_mask_shell_only,
        r_start,
        r,
        outer_axis_x,
        outer_axis_z,
        inner_axis_x,
        inner_axis_z,
        show_plot=False,
        save_plot=False,
        plot_path=None,
):
    if not (show_plot or save_plot):
        return None

    if phys_coords_shell_only_m.size == 0:
        return None

    shell_coords = phys_coords_shell_only_m[:, :2]
    idx_inner = surface_info_shell_only["indices"]
    inner_coords = shell_coords[idx_inner] if idx_inner.size else np.empty((0, 2), dtype=np.float64)
    outer_mask = outer_surface_node_mask_shell_only[:shell_coords.shape[0]]
    outer_coords = shell_coords[outer_mask] if np.any(outer_mask) else np.empty((0, 2), dtype=np.float64)

    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 360)
    outer_r = r_start + outer_axis_x * np.cos(theta)
    outer_z = r + outer_axis_z * np.sin(theta)
    inner_r = r_start + inner_axis_x * np.cos(theta)
    inner_z = r + inner_axis_z * np.sin(theta)

    fig, ax = plt.subplots(figsize=(7.0, 7.0), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(
        shell_coords[:, 0] * 1.0e6,
        shell_coords[:, 1] * 1.0e6,
        s=5,
        c="0.82",
        linewidths=0,
        label="Shell physical points",
    )
    if outer_coords.size:
        ax.scatter(
            outer_coords[:, 0] * 1.0e6,
            outer_coords[:, 1] * 1.0e6,
            s=12,
            c="#1f77b4",
            linewidths=0,
            label="Outer surface mask",
        )
    if inner_coords.size:
        ax.scatter(
            inner_coords[:, 0] * 1.0e6,
            inner_coords[:, 1] * 1.0e6,
            s=16,
            c="#d62728",
            linewidths=0,
            label="Selected inner surface",
        )
    ax.plot(outer_r * 1.0e6, outer_z * 1.0e6, color="black", linewidth=0.8)
    ax.plot(inner_r * 1.0e6, inner_z * 1.0e6, color="black", linewidth=0.8, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("r (um)")
    ax.set_ylabel("z (um)")
    ax.set_title("Shell-only inner surface selection")
    ax.set_xlim(r_start * 1.0e6, (r_start + outer_axis_x) * 1.0e6)
    ax.set_ylim((r - outer_axis_z) * 1.0e6, (r + outer_axis_z) * 1.0e6)
    ax.legend(loc="best")
    fig.tight_layout()

    saved_path = None
    if save_plot:
        if not plot_path:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            plot_path = os.path.join(desktop_path, "THM3_shell_only_inner_surface_points.png")
        output_dir = os.path.dirname(plot_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(plot_path, dpi=300)
        saved_path = plot_path
        print(f"[Shell-only inner surface plot] saved to {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
