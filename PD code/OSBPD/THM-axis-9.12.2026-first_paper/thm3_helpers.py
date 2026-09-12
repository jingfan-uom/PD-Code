import numpy as np

import core_funcs as cf
import Physical_Field_Calculation as pfc
import geometry_utils as geom


class TemperatureEvolutionContext:
    def __init__(
            self,
            total_time,
            Tinit,
            Tsurr,
            segment_slices_t,
            boundary_neighbors_t,
            T_phys,
            T_left,
            T_right,
            T_top,
            T_bot,
            mask_core_regions_t,
            factor_data_t,
            csr_area_t,
            shape_factor_t,
            csr_dist_t,
            csr_indptr_t,
            csr_indices_t,
            zones,
            rho_s,
            rho_l,
            rho_shell,
            cs,
            cl,
            c_shell,
            L,
            Ts,
            Tl,
            ks,
            kl,
            k_shell,
            rho_void,
            Cp_void,
            k_void,
            rho_void_old,
            Cp_void_old,
            rho_void_new,
            Cp_void_new,
    ):
        self.total_time = total_time
        self.Tinit = Tinit
        self.Tsurr = Tsurr
        self.segment_slices_t = segment_slices_t
        self.boundary_neighbors_t = boundary_neighbors_t
        self.T_phys = T_phys
        self.T_left = T_left
        self.T_right = T_right
        self.T_top = T_top
        self.T_bot = T_bot
        self.mask_core_regions_t = mask_core_regions_t
        self.factor_data_t = factor_data_t
        self.csr_area_t = csr_area_t
        self.shape_factor_t = shape_factor_t
        self.csr_dist_t = csr_dist_t
        self.csr_indptr_t = csr_indptr_t
        self.csr_indices_t = csr_indices_t
        self.zones = zones
        self.n_slices = len(zones)
        self.rho_s = rho_s
        self.rho_l = rho_l
        self.rho_shell = rho_shell
        self.cs = cs
        self.cl = cl
        self.c_shell = c_shell
        self.L = L
        self.Ts = Ts
        self.Tl = Tl
        self.ks = ks
        self.kl = kl
        self.k_shell = k_shell
        self.set_void_state(
            rho_void,
            Cp_void,
            k_void,
            rho_void_old,
            Cp_void_old,
            rho_void_new,
            Cp_void_new,
        )

    def set_void_state(
            self,
            rho_void,
            Cp_void,
            k_void,
            rho_void_old,
            Cp_void_old,
            rho_void_new,
            Cp_void_new,
    ):
        self.rho_void = rho_void
        self.Cp_void = Cp_void
        self.k_void = k_void
        self.rho_void_old = rho_void_old
        self.Cp_void_old = Cp_void_old
        self.rho_void_new = rho_void_new
        self.Cp_void_new = Cp_void_new

    def boundary_temperature_at_time(self, t_now):
        if self.total_time <= 0.0:
            return self.Tsurr
        frac = float(np.clip(2.0 * t_now / self.total_time, 0.0, 1.0))
        return self.Tinit + (self.Tsurr - self.Tinit) * frac

    def split_temperature_state_from(self, T_state):
        for i in range(self.n_slices):
            sl = self.segment_slices_t[i]
            self.T_phys[i] = T_state[i][sl["phys"]]
            self.T_left[i] = T_state[i][sl["left"]]
            self.T_right[i] = T_state[i][sl["right"]]
            self.T_top[i] = T_state[i][sl["top"]]
            self.T_bot[i] = T_state[i][sl["bot"]]

    def apply_dirichlet_boundary_to_temperature_state(self, T_state, boundary_value):
        bounded = {i: np.array(T_state[i], copy=True) for i in range(self.n_slices)}
        for (region_id, direction), neighbor_data in self.boundary_neighbors_t.items():
            if direction == 'right':
                sl = self.segment_slices_t[region_id]
                T_region = bounded[region_id]
                T1 = np.concatenate([
                    T_region[sl["phys"]],
                    T_region[sl["left"]],
                    T_region[sl["top"]],
                    T_region[sl["bot"]],
                ])
                ghost_indices = neighbor_data["ghost_indices"]
                phys_indices = neighbor_data["phys_indices"]
                T_region[sl["right"]][ghost_indices] = 2.0 * boundary_value - T1[phys_indices]
        return bounded

    def enthalpy_from_temperature_state(self, T_state):
        H_state = {}
        for i in range(self.n_slices):
            H_state[i] = cf.get_enthalpy(
                T_state[i],
                self.mask_core_regions_t[i],
                self.rho_s,
                self.rho_l,
                self.rho_shell,
                self.cs,
                self.cl,
                self.c_shell,
                self.L,
                self.Ts,
                self.Tl,
                self.rho_void,
                self.Cp_void,
            )
        return H_state

    def heat_capacity_from_temperature(self, T_region, mask_core):
        heat_capacity = np.where(mask_core, self.rho_s * self.cs, self.rho_shell * self.c_shell).astype(np.float64)
        core = np.asarray(mask_core, dtype=bool)
        if np.any(core):
            T_core = T_region[core]
            heat_capacity_core = np.empty_like(T_core, dtype=np.float64)
            heat_capacity_core[T_core < self.Ts] = self.rho_s * self.cs
            heat_capacity_core[T_core > self.Tl] = self.rho_l * self.cl
            phase = (T_core >= self.Ts) & (T_core <= self.Tl)
            if np.any(phase):
                dT = self.Tl - self.Ts
                alpha = (T_core[phase] - self.Ts) / dT
                drho = self.rho_l - self.rho_s
                dcp = self.cl - self.cs
                a0 = self.rho_s * (self.cs + self.L / dT)
                a1 = self.rho_s * dcp + drho * (self.cs + self.L / dT)
                a2 = drho * dcp
                heat_capacity_core[phase] = a0 + a1 * alpha + a2 * alpha * alpha
            heat_capacity[core] = heat_capacity_core
        return np.maximum(heat_capacity, 1.0e-30)

    def check_temperature_state_finite(self, T_state, label):
        for i in range(self.n_slices):
            if not np.all(np.isfinite(T_state[i])):
                raise FloatingPointError(f"[Thermal] non-finite temperature in {label}, region {i}")

    def thermal_rhs_temperature(self, T_state, boundary_value):
        T_work = self.apply_dirichlet_boundary_to_temperature_state(T_state, boundary_value)
        rhs = {}
        for i in range(self.n_slices):
            K_data, diag = cf.build_Kdata_and_rowsum_csr_numba(
                T_work[i],
                self.mask_core_regions_t[i],
                self.factor_data_t[i],
                self.csr_area_t[i],
                self.shape_factor_t[i],
                self.csr_dist_t[i],
                self.csr_indptr_t[i],
                self.csr_indices_t[i],
                self.ks,
                self.kl,
                self.Ts,
                self.Tl,
                self.k_shell,
                self.zones[i]["delta"],
                self.k_void,
                1.0,
            )
            dH_dt = cf.apply_K_with_diag_csr_numba(
                self.csr_indptr_t[i],
                self.csr_indices_t[i],
                K_data,
                diag,
                T_work[i],
                self.rho_void_old,
                self.Cp_void_old,
                self.rho_void_new,
                self.Cp_void_new,
            )
            rhs[i] = dH_dt / self.heat_capacity_from_temperature(T_work[i], self.mask_core_regions_t[i])
        return rhs

    def advance_temperature_euler(self, T_state, dt_stage, boundary_value):
        T_work = self.apply_dirichlet_boundary_to_temperature_state(T_state, boundary_value)
        H_start = self.enthalpy_from_temperature_state(T_work)
        T_new = {}
        for i in range(self.n_slices):
            K_data, diag = cf.build_Kdata_and_rowsum_csr_numba(
                T_work[i],
                self.mask_core_regions_t[i],
                self.factor_data_t[i],
                self.csr_area_t[i],
                self.shape_factor_t[i],
                self.csr_dist_t[i],
                self.csr_indptr_t[i],
                self.csr_indices_t[i],
                self.ks,
                self.kl,
                self.Ts,
                self.Tl,
                self.k_shell,
                self.zones[i]["delta"],
                self.k_void,
                dt_stage,
            )
            dH = cf.apply_K_with_diag_csr_numba(
                self.csr_indptr_t[i],
                self.csr_indices_t[i],
                K_data,
                diag,
                T_work[i],
                self.rho_void_old,
                self.Cp_void_old,
                self.rho_void_new,
                self.Cp_void_new,
            )
            H_i = H_start[i] + dH
            T_new[i] = cf.temperature_from_enthalpy_numba(
                H_i,
                self.mask_core_regions_t[i],
                self.rho_s,
                self.rho_l,
                self.cs,
                self.cl,
                self.L,
                self.Ts,
                self.Tl,
                self.rho_shell,
                self.c_shell,
                self.rho_void,
                self.Cp_void,
            )
        T_new = self.apply_dirichlet_boundary_to_temperature_state(T_new, boundary_value)
        self.check_temperature_state_finite(T_new, "Euler thermal step")
        return T_new, self.enthalpy_from_temperature_state(T_new)


def find_inner_surface_band(
        coords_phys_m,
        outer_radius,
        inner_radius,
        dr,
        center_r=0.0,
        width_factor=1.0,
        shell_mask=None,
        modify_coordinates=False,
        coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):
    return pfc.find_inner_surface_band(
        coords_phys_m,
        outer_radius,
        inner_radius,
        dr,
        center_r,
        width_factor,
        shell_mask,
        modify_coordinates,
        coordinate_x_scale,
    )


def compute_temperature_volume_expansion_over_slices(
        T_phys,
        mask_core_regions_phy_t,
        phys_coords_list_t,
        zones,
        core_volume_ref,
        rho_s,
        rho_l,
        Ts,
        Tl,
        beta_s,
        beta_l,
        T_ref,
        pressure=0.0,
        compressibility_s=0.0,
        compressibility_l=0.0,
):
    totals = {
        "deltaV_total": 0.0,
        "deltaV_phase": 0.0,
        "deltaV_thermal": 0.0,
        "deltaV_compression": 0.0,
        "melt_volume_equiv": 0.0,
    }
    density_s_num = 0.0
    density_l_num = 0.0
    density_s_weight = 0.0
    density_l_weight = 0.0
    for i in range(len(zones)):
        dr_i = zones[i]["dr"]
        dA = dr_i * dr_i
        r_phys = phys_coords_list_t[i][:, 0]
        cell_volume_node = 2.0 * np.pi * r_phys * dA
        info = pfc.compute_pcm_volume_change_consistent(
            T_phys[i],
            mask_core_regions_phy_t[i],
            rho_s,
            rho_l,
            Ts,
            Tl,
            cell_volume_node,
            beta_s=beta_s,
            beta_l=beta_l,
            T_ref=T_ref,
            pressure=pressure,
            compressibility_s=compressibility_s,
            compressibility_l=compressibility_l,
            core_volume_ref=core_volume_ref,
            return_details=True,
        )
        for key in totals:
            totals[key] += float(info[key])
        ws = float(info.get("solid_density_weight", 0.0))
        wl = float(info.get("liquid_density_weight", 0.0))
        density_s_num += float(info.get("delta_rho_s_avg", 0.0)) * ws
        density_l_num += float(info.get("delta_rho_l_avg", 0.0)) * wl
        density_s_weight += ws
        density_l_weight += wl

    if core_volume_ref > 0.0:
        totals["relative_deltaV"] = float(totals["deltaV_total"] / core_volume_ref)
        totals["melt_fraction"] = float(np.clip(totals["melt_volume_equiv"] / core_volume_ref, 0.0, 1.0))
    else:
        totals["relative_deltaV"] = 0.0
        totals["melt_fraction"] = 0.0
    delta_rho_s_avg = density_s_num / density_s_weight if density_s_weight > 0.0 else 0.0
    delta_rho_l_avg = density_l_num / density_l_weight if density_l_weight > 0.0 else 0.0
    totals["delta_rho_s_avg"] = float(delta_rho_s_avg)
    totals["delta_rho_l_avg"] = float(delta_rho_l_avg)
    totals["rho_s_eff_avg"] = float(rho_s + delta_rho_s_avg)
    totals["rho_l_eff_avg"] = float(rho_l + delta_rho_l_avg)
    totals["solid_density_weight"] = float(density_s_weight)
    totals["liquid_density_weight"] = float(density_l_weight)
    totals["core_volume"] = float(core_volume_ref)
    totals["pressure"] = float(pressure)
    return totals

def compute_pressure_from_temperature_volume_expansion(
        T_phys,
        mask_core_regions_phy_t,
        phys_coords_list_t,
        zones,
        core_volume_ref,
        shell_deltaT,
        rho_s,
        rho_l,
        Ts,
        Tl,
        beta_s,
        beta_l,
        T_ref,
        compressibility_s,
        compressibility_l,
        E,
        nu,
        R,
        r_m0,
        alpha_shell,
        pressure_for_density=0.0,
        density_iter_max=80,
        density_rel_tol=1.0e-8,
        density_relaxation=1.0,
        eps=1.0e-30,
):
    raise NotImplementedError(
        "The previous thick-shell pressure formula has been removed. "
        "Implement the paper-based pressure solver before calling this function."
    )
