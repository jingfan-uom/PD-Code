import os

import numpy as np


def normalize_thermal_stepper(value):
    stepper = value.strip().lower()
    if stepper in ("sts", "rkg", "rkg2"):
        return "rkg2"
    if stepper == "euler":
        return "euler"
    raise ValueError("THM_THERMAL_STEPPER must be 'euler' or 'rkg2'")


def rkg2_w1(stages):
    stages = int(stages)
    if stages < 2:
        raise ValueError("RKG2 requires at least 2 stages")
    return 6.0 / ((stages + 4.0) * (stages - 1.0))


def rkg2_b(stage):
    stage = int(stage)
    if stage == 0:
        return 1.0
    if stage == 1:
        return 1.0 / 3.0
    return (
        4.0 * (stage - 1.0) * (stage + 4.0)
        / (3.0 * stage * (stage + 1.0) * (stage + 2.0) * (stage + 3.0))
    )


def rkg2_a(stage):
    stage = int(stage)
    if stage == 0:
        return 1.0
    return 1.0 - 0.5 * (stage + 1.0) * (stage + 2.0) * rkg2_b(stage)


def select_rkg2_stage_count(dt_explicit, target_dt, min_stages, max_stages):
    if not np.isfinite(dt_explicit) or dt_explicit <= 0.0:
        raise ValueError(f"Invalid explicit thermal step for RKG2: {dt_explicit}")
    if not np.isfinite(target_dt) or target_dt <= 0.0:
        raise ValueError(f"Invalid target thermal step for RKG2: {target_dt}")

    min_stages = max(2, int(min_stages))
    max_stages = max(min_stages, int(max_stages))
    for stages in range(min_stages, max_stages + 1):
        capacity_dt = dt_explicit / rkg2_w1(stages)
        if capacity_dt >= target_dt:
            return stages, capacity_dt, False
    stages = max_stages
    return stages, dt_explicit / rkg2_w1(stages), True


def configure_thermal_step(
        thermal_stepper,
        dt_explicit,
        dt_euler,
        default_rkg2_target_dt,
        total_time,
):
    info = {
        "stages": 0,
        "target_dt": 0.0,
        "capacity_dt": 0.0,
        "capacity_capped": False,
        "safety": 1.0,
        "mech_load_dt_limit": 0.0,
    }

    if thermal_stepper == "euler":
        return dt_euler, info

    rkg2_target_dt_text = os.environ.get("THM_RKG2_TARGET_DT")
    if rkg2_target_dt_text:
        target_dt = float(rkg2_target_dt_text)
    else:
        target_dt = default_rkg2_target_dt
    if np.isfinite(total_time) and total_time > 0.0:
        target_dt = min(target_dt, total_time)
    target_dt = max(dt_explicit, target_dt)

    mech_load_dt_text = os.environ.get("THM_MECH_LOAD_DT")
    if mech_load_dt_text:
        info["mech_load_dt_limit"] = float(mech_load_dt_text)
    if info["mech_load_dt_limit"] > 0.0:
        target_dt = min(target_dt, max(dt_explicit, info["mech_load_dt_limit"]))

    min_stages = int(os.environ.get("THM_RKG2_MIN_STAGES", "2"))
    max_stages = int(os.environ.get("THM_RKG2_MAX_STAGES", "200"))
    stages, capacity_dt, capacity_capped = select_rkg2_stage_count(
        dt_explicit,
        target_dt,
        min_stages,
        max_stages,
    )
    safety = float(os.environ.get("THM_RKG2_SAFETY", "1.0"))
    safety = float(np.clip(safety, 0.1, 1.0))
    dt_outer = min(target_dt, capacity_dt * safety)

    info.update({
        "stages": int(stages),
        "target_dt": float(target_dt),
        "capacity_dt": float(capacity_dt),
        "capacity_capped": bool(capacity_capped),
        "safety": float(safety),
    })
    return dt_outer, info


def add_state_dicts(terms, keys=None):
    if keys is None:
        keys = terms[0][1].keys()
    out = {}
    for key in keys:
        template = terms[0][1][key]
        acc = np.zeros_like(template, dtype=np.float64)
        for coeff, state in terms:
            if coeff != 0.0:
                acc = acc + coeff * state[key]
        out[key] = acc
    return out


def advance_rkg2_state(
        state,
        super_dt,
        stages,
        rhs_func,
        boundary_value=None,
        apply_boundary_func=None,
        check_state_func=None,
):
    keys = tuple(state.keys())
    if apply_boundary_func is None:
        y0 = {key: np.array(state[key], copy=True) for key in keys}
    else:
        y0 = apply_boundary_func(state, boundary_value)

    my0 = rhs_func(y0, boundary_value)
    w1 = rkg2_w1(stages)
    y_prev2 = y0
    y_prev1 = add_state_dicts([
        (1.0, y0),
        (w1 * super_dt, my0),
    ], keys=keys)
    if apply_boundary_func is not None:
        y_prev1 = apply_boundary_func(y_prev1, boundary_value)

    for stage in range(2, int(stages) + 1):
        b_j = rkg2_b(stage)
        b_jm1 = rkg2_b(stage - 1)
        b_jm2 = rkg2_b(stage - 2)
        mu_j = ((2.0 * stage + 1.0) / stage) * (b_j / b_jm1)
        nu_j = -((stage + 1.0) / stage) * (b_j / b_jm2)
        mu_tilde_j = mu_j * w1
        gamma_tilde_j = -mu_tilde_j * rkg2_a(stage - 1)
        my_prev1 = rhs_func(y_prev1, boundary_value)
        y_j = add_state_dicts([
            (mu_j, y_prev1),
            (nu_j, y_prev2),
            (1.0 - mu_j - nu_j, y0),
            (mu_tilde_j * super_dt, my_prev1),
            (gamma_tilde_j * super_dt, my0),
        ], keys=keys)
        if apply_boundary_func is not None:
            y_j = apply_boundary_func(y_j, boundary_value)
        if check_state_func is not None:
            check_state_func(y_j, f"RKG2 stage {stage}")
        y_prev2 = y_prev1
        y_prev1 = y_j

    return y_prev1
