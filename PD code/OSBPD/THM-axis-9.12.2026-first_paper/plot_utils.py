import os
import argparse
import csv
import math
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import geometry_utils as geom
from scipy.interpolate import griddata

DEFAULT_COLOR_COUNT = 10


def plot_pressure_history(
    time_history,
    pressure_history,
    pressure_detail_history=None,
    mechanical_mode_history=None,
    title="Pressure history",
    line_label=r"$p$",
    save_path=None,
    show=True,
    include_components=False,
    comparison_history=None,
    comparison_label="Reference",
    error_history=None,
    rel_error_history=None,
):
    time_arr = np.asarray(time_history, dtype=np.float64)
    pressure_arr = np.asarray(pressure_history, dtype=np.float64)
    if time_arr.size == 0 or pressure_arr.size == 0:
        return None
    if time_arr.size != pressure_arr.size:
        raise ValueError("time_history and pressure_history must have the same length")

    comparison_arr = None
    if comparison_history is not None:
        comparison_arr = np.asarray(comparison_history, dtype=np.float64)
        if comparison_arr.size != time_arr.size:
            raise ValueError("comparison_history must have the same length as time_history")

    error_arr = None
    if error_history is not None:
        error_arr = np.asarray(error_history, dtype=np.float64)
        if error_arr.size != time_arr.size:
            raise ValueError("error_history must have the same length as time_history")
    elif comparison_arr is not None:
        error_arr = pressure_arr - comparison_arr

    rel_error_arr = None
    if rel_error_history is not None:
        rel_error_arr = np.asarray(rel_error_history, dtype=np.float64)
        if rel_error_arr.size != time_arr.size:
            raise ValueError("rel_error_history must have the same length as time_history")

    show_error_panel = error_arr is not None
    if show_error_panel:
        fig, (ax, ax_err) = plt.subplots(
            2,
            1,
            figsize=(8.0, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
            facecolor="white",
        )
        ax_err.set_facecolor("white")
    else:
        fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
        ax_err = None

    ax.set_facecolor("white")
    ax.plot(
        time_arr,
        pressure_arr,
        color="#0b1736",
        linewidth=2.1,
        label=line_label,
    )
    if comparison_arr is not None and np.any(np.isfinite(comparison_arr)):
        ax.plot(
            time_arr,
            comparison_arr,
            color="#d1495b",
            linestyle="--",
            linewidth=1.9,
            label=comparison_label,
        )

    if include_components and pressure_detail_history is not None:
        details = list(pressure_detail_history)
        if len(details) != time_arr.size:
            raise ValueError("pressure_detail_history must have the same length as time_history")

        def component(keys, default=0.0):
            values = np.full(time_arr.size, default, dtype=np.float64)
            for i, item in enumerate(details):
                if not isinstance(item, dict):
                    continue
                for key in keys:
                    if key in item:
                        try:
                            value = float(item[key])
                        except (TypeError, ValueError):
                            value = default
                        values[i] = value if np.isfinite(value) else default
                        break
            return values

        p_phase = component(("pressure_phase",))
        p_calculated = component(("pressure_calculated", "pressure_phase_target", "pressure_formula"))
        p_transfer = component(("pressure_transfer_applied", "pressure_transfer"))

        component_specs = [
            (p_phase, "#2e79a7", "-", r"$p_{\mathrm{phase}}$", False),
            (p_calculated, "#2e79a7", ":", r"$p_{\mathrm{calc}}$", False),
            (p_transfer, "#f08a24", "--", r"$p_{\mathrm{transfer}}$", False),
        ]
        for values, color, linestyle, label, show_if_finite in component_specs:
            finite_values = values[np.isfinite(values)]
            should_plot = (
                finite_values.size > 0
                and (show_if_finite or np.any(np.abs(finite_values) > 0.0))
            )
            if should_plot:
                ax.plot(
                    time_arr,
                    values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=label,
                )

    if show_error_panel:
        finite_error = np.isfinite(error_arr)
        if np.any(finite_error):
            ax_err.axhline(0.0, color="#777777", linewidth=0.9, alpha=0.7)
            ax_err.plot(
                time_arr,
                error_arr,
                color="#2f6f6d",
                linewidth=1.7,
                label=r"$p_{\mathrm{PD}} - p_{\mathrm{Lame}}$",
            )
            ax_err.set_ylabel("Error (Pa)", fontsize=13)
            ax_err.tick_params(axis="both", labelsize=11)
            ax_err.grid(True, linestyle="--", alpha=0.30)

            if rel_error_arr is not None and np.any(np.isfinite(rel_error_arr)):
                ax_rel = ax_err.twinx()
                ax_rel.plot(
                    time_arr,
                    rel_error_arr,
                    color="#8a5a00",
                    linestyle=":",
                    linewidth=1.4,
                    label="relative error",
                )
                ax_rel.set_ylabel("Relative error", fontsize=13)
                ax_rel.tick_params(axis="y", labelsize=11)
                lines, labels = ax_err.get_legend_handles_labels()
                rel_lines, rel_labels = ax_rel.get_legend_handles_labels()
                ax_err.legend(lines + rel_lines, labels + rel_labels, frameon=False, fontsize=10, loc="best")
            else:
                ax_err.legend(frameon=False, fontsize=10, loc="best")

            final_idx = int(np.where(finite_error)[0][-1])
            summary_lines = [
                f"final PD = {pressure_arr[final_idx]:.6e} Pa",
            ]
            if comparison_arr is not None and np.isfinite(comparison_arr[final_idx]):
                summary_lines.append(f"final Lame = {comparison_arr[final_idx]:.6e} Pa")
            summary_lines.append(f"final error = {error_arr[final_idx]:.6e} Pa")
            if rel_error_arr is not None and np.isfinite(rel_error_arr[final_idx]):
                summary_lines.append(f"final rel. error = {rel_error_arr[final_idx]:.6e}")
            ax.text(
                0.02,
                0.98,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9.5,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.88,
                },
            )

    if mechanical_mode_history is not None:
        modes = list(mechanical_mode_history)
        if len(modes) != time_arr.size:
            raise ValueError("mechanical_mode_history must have the same length as time_history")
        switch_idx = next(
            (i for i, mode in enumerate(modes) if str(mode).startswith("shell_only")),
            None,
        )
        if switch_idx is not None:
            ax.axvline(
                time_arr[switch_idx],
                color="#df2935",
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
                label="shell-only switch",
            )

    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Pressure (Pa)", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=11, loc="best")
    if ax_err is not None:
        ax.set_xlabel("")
        ax_err.set_xlabel("Time (s)", fontsize=14)
    fig.tight_layout()

    saved_path = save_path
    if saved_path is None:
        saved_path = _save_figure_from_env(fig, "pressure_history.png")
    elif saved_path:
        output_dir = os.path.dirname(saved_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(saved_path, dpi=int(os.environ.get("THM_PLOT_DPI", "450")), bbox_inches="tight")
        print(f"[Pressure plot] saved to {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def _save_figure_from_env(fig, default_name):
    output_dir = os.environ.get("THM_CONTOUR_PLOT_DIR")
    if not output_dir:
        return None

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.environ.get("THM_PLOT_PREFIX", "").strip()
    filename = f"{prefix}_{default_name}" if prefix else default_name
    path = os.path.join(output_dir, filename)

    try:
        dpi = int(os.environ.get("THM_PLOT_DPI", "450"))
    except ValueError:
        dpi = 450

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[Contour plot] saved to {path}")
    return path


def _contour_levels(vmin, vmax, color_count=DEFAULT_COLOR_COUNT):
    color_count = max(int(color_count), 1)
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, color_count + 1)


def _discrete_cmap(cmap, color_count=DEFAULT_COLOR_COUNT):
    color_count = max(int(color_count), 1)
    if isinstance(cmap, str):
        return plt.get_cmap(cmap, color_count)
    if hasattr(cmap, "resampled"):
        return cmap.resampled(color_count)
    return cmap


def _griddata_linear_with_nearest_fallback(points, values, point):
    value = griddata(points, values, point, method='linear')
    if np.all(np.isfinite(value)):
        return value
    return griddata(points, values, point, method='nearest')


def build_tracked_points(
    phys_coords_m,
    target_points,
    shell_phys_mask=None,
    shell_labels=None,
    center_z=0.0,
    tolerance=1.0e-14,
    inner_label="(rshell, r)",
    outer_label="(r, r)",
    verbose=True,
):
    import Multiprocess_task_function as mt

    if isinstance(phys_coords_m, list):
        phys_coords_list_m = phys_coords_m
    else:
        phys_coords_list_m = [phys_coords_m]

    if shell_phys_mask is None:
        shell_phys_masks_by_slice = None
    elif isinstance(shell_phys_mask, list):
        shell_phys_masks_by_slice = shell_phys_mask
    else:
        shell_phys_masks_by_slice = [shell_phys_mask]
    shell_label_set = set() if shell_labels is None else set(shell_labels)

    tracked_points = []
    for label, target_r, target_z in target_points:
        candidate_masks = shell_phys_masks_by_slice if (
            label == inner_label or label in shell_label_set
        ) else None
        best_slice, best_local_idx, best_coord = mt.find_closest_point_in_phys(
            phys_coords_list_m,
            target_r,
            target_z,
            candidate_masks_by_slice=candidate_masks,
        )
        mirror_slice = None
        mirror_local_idx = None
        mirror_coord = None
        tracking_mode = "nearest_particle"
        track_coord = best_coord

        if best_coord is not None and abs(target_z - center_z) <= tolerance:
            mirror_target = np.array([best_coord[0], 2.0 * target_z - best_coord[1]], dtype=np.float64)
            mirror_slice, mirror_local_idx, mirror_coord = mt.find_closest_point_to_coord(
                phys_coords_list_m,
                mirror_target,
                candidate_masks_by_slice=candidate_masks,
            )
            if mirror_coord is not None and (
                    mirror_slice != best_slice or mirror_local_idx != best_local_idx
            ):
                tracking_mode = "midline_symmetric_average"
                track_coord = 0.5 * (best_coord + mirror_coord)

        surface_type = None
        if label == inner_label:
            surface_type = "inner"
        elif label == outer_label:
            surface_type = "outer"

        tracked_points.append({
            "label": label,
            "target_r": target_r,
            "target_z": target_z,
            "surface_type": surface_type,
            "slice": best_slice,
            "local_idx": best_local_idx,
            "coord": track_coord,
            "primary_coord": best_coord,
            "mirror_slice": mirror_slice,
            "mirror_local_idx": mirror_local_idx,
            "mirror_coord": mirror_coord,
            "tracking_mode": tracking_mode,
        })

        if verbose:
            print(label)
            print("  best_slice =", best_slice)
            print("  best_local_idx =", best_local_idx)
            print("  best_coord =", best_coord)
            if tracking_mode == "midline_symmetric_average":
                print("  tracking_mode = midline_symmetric_average")
                print("  mirror_slice =", mirror_slice)
                print("  mirror_local_idx =", mirror_local_idx)
                print("  mirror_coord =", mirror_coord)
                print("  averaged_coord =", track_coord)

    return tracked_points

def plot_temperature_contour_in_circle(
    phys_coords_list,
    dr,
    T_phys,
    radius,
    cmap,
    title=r"$T$ (K)",
    levels=DEFAULT_COLOR_COUNT,
    r_start=0.0,
    shell_thickness=None,
    shell_only=False,
    modify_coordinates=False,
    coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):
    """
    Plot contour lines of the temperature field only within the shifted semicircular region (r, z).

    Geometry:
    - circle center: (r_start, radius)
    - circle radius: radius
    """

    # 1. Merge the physical point coordinates of all regions
    all_coords = np.vstack([arr[:, :2] for arr in phys_coords_list])  # shape: (N, 2)

    # 2. Merge all temperatures
    if isinstance(T_phys, dict):
        all_temps = np.concatenate([T_phys[i] for i in range(len(phys_coords_list))])
    elif isinstance(T_phys, list):
        all_temps = np.concatenate(T_phys)
    else:
        raise TypeError("T_phys 必须是 list 或 dict 类型")

    # 3. Create a regular grid
    r_vals = all_coords[:, 0]
    z_vals = all_coords[:, 1]

    outer_axis_x, outer_axis_z = geom.capsule_axes(
        radius,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
    )
    r_lin = np.linspace(r_start, r_start + outer_axis_x, 300)
    z_lin = np.linspace(0.0, 2.0 * radius, 300)
    r_grid, z_grid = np.meshgrid(r_lin, z_lin)

    # 4. Interpolation
    T_grid = griddata(all_coords, all_temps, (r_grid, z_grid), method='linear')

    # characteristic points in the shifted semicircle
    T_rmin = _griddata_linear_with_nearest_fallback(
        all_coords, all_temps, (r_start + dr / 2, radius)
    )
    T_zmin = _griddata_linear_with_nearest_fallback(
        all_coords, all_temps, (r_start + dr / 2, dr / 2)
    )
    T_zmax = _griddata_linear_with_nearest_fallback(
        all_coords, all_temps, (r_start + dr / 2, 2.0 * radius - dr / 2)
    )
    T_zmax1 = _griddata_linear_with_nearest_fallback(
        all_coords, all_temps, (r_start + radius - dr / 2, radius)
    )

    print(f"T(r={r_start + dr / 2:.3e}, z={radius:.3e}) = {T_rmin:.2f} K")
    print(f"T(r={r_start + dr / 2:.3e}, z={dr / 2:.3e}) = {T_zmin:.2f} K")
    print(f"T(r={r_start + dr / 2:.3e}, z={2 * radius - dr / 2:.3e}) = {T_zmax:.2f} K")
    print(f"T(r={r_start + radius - dr / 2:.3e}, z={radius:.3e}) = {T_zmax1:.2f} K")

    # 5. Mask processing: shifted capsule/semicircle.
    outer_level = geom.ellipse_level(
        r_grid, z_grid,
        r_start, radius,
        outer_axis_x, outer_axis_z,
    )
    mask_domain = (
        (outer_level <= 1.0) &
        (r_grid >= r_start) &
        (z_grid >= 0.0) &
        (z_grid <= 2.0 * radius)
    )
    has_shell = shell_thickness is not None and shell_thickness > 0.0
    if has_shell:
        inner_axis_x, inner_axis_z = geom.capsule_axes(
            radius,
            dshell=shell_thickness,
            modify_coordinates=modify_coordinates,
            coordinate_x_scale=coordinate_x_scale,
            inner=True,
        )
        inner_level = geom.ellipse_level(
            r_grid, z_grid,
            r_start, radius,
            inner_axis_x, inner_axis_z,
        )
        if shell_only:
            mask_domain &= inner_level >= 1.0

    T_grid[~mask_domain] = np.nan

    # 6. Set contour levels
    vmin = np.nanmin(T_grid)
    vmax = np.nanmax(T_grid)
    level_values = _contour_levels(vmin, vmax, levels)
    cmap = _discrete_cmap(cmap, levels)

    # 7. Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(r_grid, z_grid, T_grid, levels=level_values, cmap=cmap)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(
        f"Temperature (K)\nMin: {vmin:.2f} K\nMax: {vmax:.2f} K",
        rotation=270, labelpad=15, va='bottom', fontsize=16
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.get_offset_text().set_fontsize(14)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r"$r$ (m)", fontsize=16)
    ax.set_ylabel(r"$z$ (m)", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    ax.xaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontsize(14)

    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 360)
    outer_r = r_start + outer_axis_x * np.cos(theta)
    outer_z = radius + outer_axis_z * np.sin(theta)
    ax.plot(outer_r, outer_z, color="black", linewidth=0.8)
    if has_shell:
        inner_r = r_start + inner_axis_x * np.cos(theta)
        inner_z = radius + inner_axis_z * np.sin(theta)
        ax.plot(inner_r, inner_z, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlim([r_start, r_start + outer_axis_x])
    ax.set_ylim([0, 2 * radius])

    plt.tight_layout()
    _save_figure_from_env(fig, "temperature_contour.png")
    plt.show()

def plot_displacement_contours_in_circle(
    phys_coords_list_m,
    U_phys,
    radius,
    dr,
    r_start=0.0,
    titles=(r"$U_r$ (m)", r"$U_z$ (m)", r"$U_{\mathrm{mag}}$ (m)"),
    cmaps=('jet', 'jet', 'jet'),
    levels=DEFAULT_COLOR_COUNT,
    grid_n=300,
    shell_thickness=None,
    shell_only=False,
    modify_coordinates=False,
    coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):

    # 1) Merge physical coordinates of all regions
    all_coords = np.vstack([arr[:, :2] for arr in phys_coords_list_m])  # (N, 2)

    # 2) Merge Ur, Uz, Umag from U_phys
    if isinstance(U_phys, dict):
        Ur_all = np.concatenate([U_phys[i]["Ur"] for i in range(len(phys_coords_list_m))])
        Uz_all = np.concatenate([U_phys[i]["Uz"] for i in range(len(phys_coords_list_m))])
        if all(("Umag" in U_phys[i]) for i in range(len(phys_coords_list_m))):
            Umag_all = np.concatenate([U_phys[i]["Umag"] for i in range(len(phys_coords_list_m))])
        else:
            Umag_all = np.sqrt(Ur_all ** 2 + Uz_all ** 2)

    elif isinstance(U_phys, list):
        Ur_all = np.concatenate([item[0] for item in U_phys])
        Uz_all = np.concatenate([item[1] for item in U_phys])
        if len(U_phys[0]) >= 3:
            Umag_all = np.concatenate([item[2] for item in U_phys])
        else:
            Umag_all = np.sqrt(Ur_all ** 2 + Uz_all ** 2)
    else:
        raise TypeError("U_phys must be a dict or list")

    # 3) Build regular grid using the same capsule/ellipse geometry as coordinates.
    outer_axis_x, outer_axis_z = geom.capsule_axes(
        radius,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
    )
    z_min = radius - outer_axis_z
    z_max = radius + outer_axis_z
    r_lin = np.linspace(r_start, r_start + outer_axis_x, grid_n)
    z_lin = np.linspace(z_min, z_max, grid_n)
    r_grid, z_grid = np.meshgrid(r_lin, z_lin)

    # 4) Interpolate Ur, Uz, Umag onto the grid
    Ur_grid = griddata(all_coords, Ur_all, (r_grid, z_grid), method='linear')
    Uz_grid = griddata(all_coords, Uz_all, (r_grid, z_grid), method='linear')
    Umag_grid = griddata(all_coords, Umag_all, (r_grid, z_grid), method='linear')
    Ur_nearest = griddata(all_coords, Ur_all, (r_grid, z_grid), method='nearest')
    Uz_nearest = griddata(all_coords, Uz_all, (r_grid, z_grid), method='nearest')
    Umag_nearest = griddata(all_coords, Umag_all, (r_grid, z_grid), method='nearest')

    # 5) Mask: keep only the shifted semi-ellipse, and optionally only its shell.
    outer_level = geom.ellipse_level(
        r_grid, z_grid,
        r_start, radius,
        outer_axis_x, outer_axis_z,
    )
    mask_domain = (
        (outer_level <= 1.0) &
        (r_grid >= r_start) &
        (z_grid >= z_min) &
        (z_grid <= z_max)
    )

    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 360)
    outer_r = r_start + outer_axis_x * np.cos(theta)
    outer_z = radius + outer_axis_z * np.sin(theta)
    has_shell = shell_thickness is not None and shell_thickness > 0.0
    if has_shell:
        inner_axis_x, inner_axis_z = geom.capsule_axes(
            radius,
            dshell=shell_thickness,
            modify_coordinates=modify_coordinates,
            coordinate_x_scale=coordinate_x_scale,
            inner=True,
        )
        inner_r = r_start + inner_axis_x * np.cos(theta)
        inner_z = radius + inner_axis_z * np.sin(theta)
        if shell_only:
            inner_level = geom.ellipse_level(
                r_grid, z_grid,
                r_start, radius,
                inner_axis_x, inner_axis_z,
            )
            mask_domain &= inner_level >= 1.0

    for G, G_nearest in (
        (Ur_grid, Ur_nearest),
        (Uz_grid, Uz_nearest),
        (Umag_grid, Umag_nearest),
    ):
        fill = mask_domain & ~np.isfinite(G)
        G[fill] = G_nearest[fill]
        G[~mask_domain] = np.nan

    # 6) Levels for each field
    def level_array(G):
        finite = G[np.isfinite(G)]
        if finite.size == 0:
            raise ValueError("no finite displacement values remain inside the plot domain")
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        return _contour_levels(vmin, vmax, levels), vmin, vmax

    levels_Ur, vmin_Ur, vmax_Ur = level_array(Ur_grid)
    levels_Uz, vmin_Uz, vmax_Uz = level_array(Uz_grid)
    levels_Um, vmin_Um, vmax_Um = level_array(Umag_grid)

    # 7) Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    grids = [Ur_grid, Uz_grid, Umag_grid]
    levels_list = [levels_Ur, levels_Uz, levels_Um]
    vmins = [vmin_Ur, vmin_Uz, vmin_Um]
    vmaxs = [vmax_Ur, vmax_Uz, vmax_Um]

    for ax, G, lv, ttl, cmap, vmin, vmax in zip(
        axes, grids, levels_list, titles, cmaps, vmins, vmaxs
    ):
        cmap = _discrete_cmap(cmap, levels)
        cf = ax.contourf(r_grid, z_grid, G, levels=lv, cmap=cmap)
        ax.plot(outer_r, outer_z, color="black", linewidth=0.8)
        if has_shell:
            ax.plot(inner_r, inner_z, color="black", linewidth=0.8, linestyle="--")
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(
            f"Min: {vmin:.3e}\nMax: {vmax:.3e}",
            rotation=270, labelpad=6, va='bottom', fontsize=16
        )
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.yaxis.get_offset_text().set_fontsize(14)

        ax.set_title(ttl, fontsize=20)
        ax.set_xlabel(r"$r$ (m)", fontsize=16)
        ax.set_ylabel(r"$z$ (m)", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)

        ax.xaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_fontsize(14)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([r_start, r_start + outer_axis_x])
        ax.set_ylim([z_min, z_max])

    _save_figure_from_env(fig, "displacement_contours.png")
    plt.show()

# ---- Merged legacy plot scripts ----
# From plot_current_conversion_diagram.py
_current_conversion_OUT_DIR = Path(__file__).resolve().parent / 'figures'
_current_conversion_PNG_PATH = _current_conversion_OUT_DIR / 'current_conversion_diagram.png'
_current_conversion_SVG_PATH = _current_conversion_OUT_DIR / 'current_conversion_diagram.svg'
_current_conversion_NAVY = '#0b1736'
_current_conversion_BLUE = '#cfeeff'
_current_conversion_BLUE_EDGE = '#2e79a7'
_current_conversion_CORE = '#f7dfb5'
_current_conversion_GREEN = '#15935f'
_current_conversion_ORANGE = '#f49a27'
_current_conversion_RED = '#e0282f'
_current_conversion_GRAY = '#7a8795'
_current_conversion_PANEL = '#f7f9fb'
_current_conversion_TEXT = '#111827'

def _current_conversion_semicircle_wedge(center, outer_r, inner_r=0.0, theta1=-90, theta2=90, **kwargs):
    """Right half annulus used for axisymmetric cross-section sketches."""
    cx, cy = center
    n = 100
    ts_outer = [math.radians(theta1 + (theta2 - theta1) * i / (n - 1)) for i in range(n)]
    outer = [(cx + outer_r * math.cos(t), cy + outer_r * math.sin(t)) for t in ts_outer]
    if inner_r <= 0:
        verts = [(cx, cy + outer_r * math.sin(math.radians(theta1)))] + outer
        verts += [(cx, cy + outer_r * math.sin(math.radians(theta2))), (cx, cy + outer_r * math.sin(math.radians(theta1)))]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * len(outer) + [MplPath.LINETO, MplPath.CLOSEPOLY]
    else:
        ts_inner = [math.radians(theta2 - (theta2 - theta1) * i / (n - 1)) for i in range(n)]
        inner = [(cx + inner_r * math.cos(t), cy + inner_r * math.sin(t)) for t in ts_inner]
        verts = outer + inner + [outer[0]]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer) - 1)
        codes += [MplPath.LINETO] * len(inner) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kwargs)

def _current_conversion_add_arrow(ax, start, end, color=_current_conversion_GREEN, lw=1.8, mutation_scale=14, alpha=1.0):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=mutation_scale, linewidth=lw, color=color, alpha=alpha, shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)
    return arrow

def _current_conversion_add_box(ax, xy, wh, text, edge, face='#ffffff', lw=2.0, fontsize=12):
    x, y = xy
    w, h = wh
    box = Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color=_current_conversion_TEXT, fontweight='bold', linespacing=1.15)
    return box

def _current_conversion_draw_geometry(ax):
    ax.text(0.02, 0.91, 'A. State transfer at phase switch', fontsize=16, color=_current_conversion_NAVY, fontweight='bold', ha='left')
    c1 = (0.17, 0.52)
    outer = 0.27
    inner = 0.19
    ax.add_patch(_current_conversion_semicircle_wedge(c1, outer, 0, facecolor=_current_conversion_CORE, edgecolor=_current_conversion_NAVY, linewidth=1.4))
    ax.add_patch(_current_conversion_semicircle_wedge(c1, outer, inner, facecolor=_current_conversion_BLUE, edgecolor=_current_conversion_BLUE_EDGE, linewidth=1.8))
    ax.plot([c1[0], c1[0]], [c1[1] - outer, c1[1] + outer], color=_current_conversion_NAVY, lw=1.2)
    ax.text(c1[0] - 0.035, c1[1] + 0.15, 'Axis r = 0', rotation=90, fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(c1[0] + 0.075, c1[1], 'Core', fontsize=14, fontweight='bold', color='#6b3f0b')
    ax.text(c1[0] + 0.2, c1[1] - 0.22, 'Shell', fontsize=13, fontweight='bold', color=_current_conversion_NAVY)
    ax.text(c1[0] + 0.04, c1[1] + outer + 0.06, 'Core-shell model', fontsize=13, fontweight='bold', ha='center')
    ax.text(c1[0] + 0.04, c1[1] + outer + 0.025, 'current U = (Ur, Uz)', fontsize=10.5, color=_current_conversion_GRAY, ha='center')
    for deg in [-62, -34, -10, 18, 47]:
        t = math.radians(deg)
        x = c1[0] + inner * math.cos(t)
        y = c1[1] + inner * math.sin(t)
        ax.add_patch(Circle((x, y), 0.009, facecolor=_current_conversion_RED, edgecolor='white', linewidth=0.7))
        ax.add_patch(Circle((x + 0.035 * math.cos(t), y + 0.018 * math.sin(t)), 0.006, facecolor=_current_conversion_ORANGE, edgecolor='white', linewidth=0.5))
        _current_conversion_add_arrow(ax, (x, y), (x + 0.06 * math.cos(t), y + 0.06 * math.sin(t)), color=_current_conversion_GREEN, lw=1.4, mutation_scale=12)
    _current_conversion_add_arrow(ax, (0.44, 0.52), (0.54, 0.52), color=_current_conversion_NAVY, lw=2.2, mutation_scale=18)
    ax.text(0.49, 0.58, 'copy shell-node\nU target', fontsize=10.5, color=_current_conversion_NAVY, ha='center', fontweight='bold')
    c2 = (0.67, 0.52)
    ax.add_patch(_current_conversion_semicircle_wedge(c2, outer, 0, facecolor='#fff8ea', edgecolor=_current_conversion_NAVY, linewidth=1.0, alpha=0.5))
    ax.add_patch(_current_conversion_semicircle_wedge(c2, outer, inner, facecolor=_current_conversion_BLUE, edgecolor=_current_conversion_BLUE_EDGE, linewidth=1.8))
    ax.add_patch(Arc(c2, 2 * inner, 2 * inner, theta1=-90, theta2=90, color=_current_conversion_NAVY, lw=1.1, ls='--'))
    ax.text(c2[0] + 0.05, c2[1], 'Core\nremoved', fontsize=11, color=_current_conversion_GRAY, ha='center', va='center')
    ax.text(c2[0] + 0.2, c2[1] - 0.22, 'Shell-only', fontsize=13, fontweight='bold', color=_current_conversion_NAVY)
    ax.text(c2[0] + 0.04, c2[1] + outer + 0.06, 'Shell-only model', fontsize=13, fontweight='bold', ha='center')
    ax.text(c2[0] + 0.04, c2[1] + outer + 0.025, 'load = p_transfer * b_unit + b_res', fontsize=10.5, color=_current_conversion_GRAY, ha='center')
    for deg in [-58, -28, 3, 33, 61]:
        t = math.radians(deg)
        x = c2[0] + inner * math.cos(t)
        y = c2[1] + inner * math.sin(t)
        ax.add_patch(Circle((x, y), 0.008, facecolor=_current_conversion_RED, edgecolor='white', linewidth=0.7))
        _current_conversion_add_arrow(ax, (x - 0.012 * math.cos(t), y - 0.012 * math.sin(t)), (x + 0.075 * math.cos(t), y + 0.075 * math.sin(t)), color=_current_conversion_ORANGE, lw=1.5, mutation_scale=12)
    for deg in [-44, -4, 44]:
        t = math.radians(deg)
        x = c2[0] + 0.235 * math.cos(t)
        y = c2[1] + 0.235 * math.sin(t)
        _current_conversion_add_arrow(ax, (x - 0.03, y - 0.015), (x + 0.035, y + 0.02), color=_current_conversion_GREEN, lw=1.3, mutation_scale=11)
    ax.text(0.05, 0.16, 'Red nodes: matched shell nodes carried into shell-only grid', fontsize=9.5, color=_current_conversion_TEXT)
    ax.add_patch(Rectangle((0.05, 0.127), 0.018, 0.018, facecolor=_current_conversion_ORANGE, edgecolor=_current_conversion_ORANGE))
    ax.text(0.077, 0.136, 'uniform retained pressure component', fontsize=9.5, va='center', color=_current_conversion_TEXT)
    ax.add_patch(Rectangle((0.05, 0.093), 0.018, 0.018, facecolor=_current_conversion_GREEN, edgecolor=_current_conversion_GREEN))
    ax.text(0.077, 0.102, 'residual equivalent body-force component', fontsize=9.5, va='center', color=_current_conversion_TEXT)
    ax.add_patch(Rectangle((0.05, 0.059), 0.018, 0.018, facecolor=_current_conversion_RED, edgecolor=_current_conversion_RED))
    ax.text(0.077, 0.068, 'crack state is not advanced by the conversion load', fontsize=9.5, va='center', color=_current_conversion_TEXT)

def _current_conversion_draw_flow(ax):
    ax.text(0.02, 0.91, 'B. Current code path', fontsize=16, color=_current_conversion_NAVY, fontweight='bold', ha='left')
    ax.text(0.02, 0.865, 'main_validation.py: switch block 1198-1506; pressure block 1587-1638', fontsize=10.5, color='#b85c00', fontweight='bold', ha='left')
    boxes = [((0.04, 0.74), (0.36, 0.085), '1. Detect phase switch\ncore_temp_max >= Ts', _current_conversion_ORANGE), ((0.54, 0.74), (0.36, 0.085), '2. Map U from core-shell\nto shell-only nodes', '#5794f2'), ((0.18, 0.62), (0.58, 0.075), '3. Build target displacement state U_shell_start', _current_conversion_GRAY), ((0.04, 0.49), (0.36, 0.095), '4. Compute internal force\nat U_shell_start', _current_conversion_RED), ((0.54, 0.49), (0.36, 0.095), '5. Equivalent load\nb_eq = -F_internal', '#8b5cf6'), ((0.18, 0.35), (0.58, 0.095), '6. Decompose conversion load\nb_eq = p_transfer * b_unit + b_res', _current_conversion_GREEN), ((0.18, 0.21), (0.58, 0.095), '7. Solve shell-only equilibrium\nwith transfer load only', '#f0b429'), ((0.18, 0.075), (0.58, 0.09), '8. Later thermal steps add phase pressure\np = p_transfer + p_phase', '#38bdf8')]
    for xy, wh, text, edge in boxes:
        _current_conversion_add_box(ax, xy, wh, text, edge=edge, face='#ffffff', fontsize=11)
    _current_conversion_add_arrow(ax, (0.4, 0.782), (0.54, 0.782), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    _current_conversion_add_arrow(ax, (0.72, 0.74), (0.48, 0.695), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    _current_conversion_add_arrow(ax, (0.47, 0.62), (0.27, 0.585), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    _current_conversion_add_arrow(ax, (0.4, 0.537), (0.54, 0.537), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    _current_conversion_add_arrow(ax, (0.72, 0.49), (0.48, 0.445), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    _current_conversion_add_arrow(ax, (0.47, 0.35), (0.47, 0.305), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    _current_conversion_add_arrow(ax, (0.47, 0.21), (0.47, 0.165), color=_current_conversion_NAVY, lw=1.7, mutation_scale=13)
    callout = Rectangle((0.79, 0.205), 0.17, 0.24, linewidth=1.5, edgecolor=_current_conversion_RED, facecolor='#fff3f3')
    ax.add_patch(callout)
    ax.text(0.875, 0.39, 'Conversion\nassumptions', ha='center', va='center', fontsize=10.5, color=_current_conversion_RED, fontweight='bold')
    ax.text(0.875, 0.315, 'crack_pressure = 0\ncrack force off\ninitial crack\nintact', ha='center', va='center', fontsize=9.5, color=_current_conversion_TEXT, linespacing=1.25)
    ax.text(0.49, 0.032, 'pressure_ramp_steps belongs to post-conversion phase-pressure loading;\nit is not part of the displacement-target conversion.', ha='center', va='center', fontsize=9.2, color=_current_conversion_GRAY)

def _current_conversion_main():
    _current_conversion_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.0, 'savefig.facecolor': '#fbfaf7'})
    fig = plt.figure(figsize=(15.2, 8.6), facecolor='#fbfaf7')
    fig.text(0.03, 0.965, 'Current Conversion: Core-Shell Displacement -> Shell-Only Equivalent Load', fontsize=20, color=_current_conversion_NAVY, fontweight='bold', ha='left', va='top')
    fig.text(0.03, 0.925, 'The conversion preserves the pre-switch shell displacement by a one-time equivalent body force. Fracture and phase-pressure growth are handled after the conversion.', fontsize=10.5, color=_current_conversion_TEXT, fontweight='bold', ha='left', va='top')
    ax_left = fig.add_axes([0.025, 0.06, 0.44, 0.82])
    ax_right = fig.add_axes([0.49, 0.06, 0.485, 0.82])
    for ax in (ax_left, ax_right):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=_current_conversion_PANEL, edgecolor='#e6e0d6', linewidth=1.0, zorder=-10))
    _current_conversion_draw_geometry(ax_left)
    _current_conversion_draw_flow(ax_right)
    fig.savefig(_current_conversion_PNG_PATH, dpi=220, bbox_inches='tight')
    fig.savefig(_current_conversion_SVG_PATH, bbox_inches='tight')
    print(_current_conversion_PNG_PATH)
    print(_current_conversion_SVG_PATH)


# From plot_main_validation_flowchart_zh.py
_main_validation_flowchart_zh_OUT_DIR = Path(__file__).resolve().parent / 'figures'
_main_validation_flowchart_zh_PNG_PATH = _main_validation_flowchart_zh_OUT_DIR / 'main_validation_flowchart_zh.png'
_main_validation_flowchart_zh_SVG_PATH = _main_validation_flowchart_zh_OUT_DIR / 'main_validation_flowchart_zh.svg'
_main_validation_flowchart_zh_NAVY = '#0b1736'
_main_validation_flowchart_zh_TEXT = '#111827'
_main_validation_flowchart_zh_MUTED = '#667085'
_main_validation_flowchart_zh_PANEL = '#fbfaf7'
_main_validation_flowchart_zh_CORE = '#f6dfb5'
_main_validation_flowchart_zh_SHELL = '#d5f0ff'
_main_validation_flowchart_zh_SHELL_EDGE = '#2b7aaa'
_main_validation_flowchart_zh_GREEN = '#16875a'
_main_validation_flowchart_zh_ORANGE = '#f08a24'
_main_validation_flowchart_zh_RED = '#df2935'
_main_validation_flowchart_zh_BLUE = '#3b82f6'
_main_validation_flowchart_zh_PURPLE = '#7c3aed'
_main_validation_flowchart_zh_YELLOW = '#e6a700'
_main_validation_flowchart_zh_TEAL = '#0f9f9a'
_main_validation_flowchart_zh_GRAY = '#7a8795'

def _main_validation_flowchart_zh_choose_font():
    candidates = ['Heiti SC', 'Hiragino Sans GB', 'PingFang SC', 'Arial Unicode MS', 'Songti SC', 'Noto Sans CJK SC', 'SimHei']
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return 'DejaVu Sans'

def _main_validation_flowchart_zh_semicircle_wedge(center, outer_r, inner_r=0.0, theta1=-90, theta2=90, **kwargs):
    cx, cy = center
    n = 120
    ts_outer = [math.radians(theta1 + (theta2 - theta1) * i / (n - 1)) for i in range(n)]
    outer = [(cx + outer_r * math.cos(t), cy + outer_r * math.sin(t)) for t in ts_outer]
    if inner_r <= 0:
        verts = [(cx, cy + outer_r * math.sin(math.radians(theta1)))] + outer
        verts += [(cx, cy + outer_r * math.sin(math.radians(theta2))), (cx, cy + outer_r * math.sin(math.radians(theta1)))]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * len(outer)
        codes += [MplPath.LINETO, MplPath.CLOSEPOLY]
    else:
        ts_inner = [math.radians(theta2 - (theta2 - theta1) * i / (n - 1)) for i in range(n)]
        inner = [(cx + inner_r * math.cos(t), cy + inner_r * math.sin(t)) for t in ts_inner]
        verts = outer + inner + [outer[0]]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer) - 1)
        codes += [MplPath.LINETO] * len(inner) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kwargs)

def _main_validation_flowchart_zh_add_arrow(ax, start, end, color=_main_validation_flowchart_zh_NAVY, lw=1.6, scale=13, alpha=1.0, rad=0.0):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, linewidth=lw, color=color, alpha=alpha, shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(arrow)
    return arrow

def _main_validation_flowchart_zh_add_box(ax, x, y, w, h, text, edge, face='#ffffff', fontsize=10.4, lw=1.8):
    box = Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color=_main_validation_flowchart_zh_TEXT, fontweight='bold', linespacing=1.18)
    return box

def _main_validation_flowchart_zh_add_diamond(ax, cx, cy, w, h, text, edge, face='#ffffff', fontsize=10.4):
    points = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    diamond = Polygon(points, closed=True, linewidth=1.8, edgecolor=edge, facecolor=face)
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, color=_main_validation_flowchart_zh_TEXT, fontweight='bold', linespacing=1.15)
    return diamond

def _main_validation_flowchart_zh_draw_capsule_schematic(ax):
    ax.text(0.03, 0.95, 'A. 物理模型与转场载荷', fontsize=16, color=_main_validation_flowchart_zh_NAVY, fontweight='bold', ha='left', va='top')
    ax.text(0.03, 0.9, 'core-shell 模型在相变触发后转换为 shell-only 模型', fontsize=10, color=_main_validation_flowchart_zh_MUTED, ha='left', va='top')
    center = (0.31, 0.51)
    outer = 0.31
    inner = 0.22
    ax.add_patch(_main_validation_flowchart_zh_semicircle_wedge(center, outer, 0.0, facecolor=_main_validation_flowchart_zh_CORE, edgecolor=_main_validation_flowchart_zh_NAVY, linewidth=1.2))
    ax.add_patch(_main_validation_flowchart_zh_semicircle_wedge(center, outer, inner, facecolor=_main_validation_flowchart_zh_SHELL, edgecolor=_main_validation_flowchart_zh_SHELL_EDGE, linewidth=1.8))
    ax.plot([center[0], center[0]], [center[1] - outer, center[1] + outer], color=_main_validation_flowchart_zh_NAVY, lw=1.3)
    ax.text(center[0] - 0.045, center[1] + 0.14, 'Axis r = 0', rotation=90, fontsize=10, color=_main_validation_flowchart_zh_NAVY, fontweight='bold', ha='center', va='center')
    ax.text(center[0] + 0.09, center[1] + 0.03, 'Core', fontsize=15, color='#70470d', fontweight='bold')
    ax.text(center[0] + 0.24, center[1] - 0.23, 'Shell', fontsize=14, color=_main_validation_flowchart_zh_NAVY, fontweight='bold')
    bond_angles = [-62, -38, -12, 19, 47, 70]
    for k, deg in enumerate(bond_angles):
        t = math.radians(deg)
        x_shell = center[0] + inner * math.cos(t)
        y_shell = center[1] + inner * math.sin(t)
        x_core = center[0] + (inner - 0.06 - 0.015 * (k % 2)) * math.cos(t)
        y_core = center[1] + (inner - 0.06 - 0.015 * (k % 2)) * math.sin(t)
        ax.plot([x_core, x_shell], [y_core, y_shell], color=_main_validation_flowchart_zh_RED, lw=2.0)
        ax.add_patch(Circle((x_shell, y_shell), 0.01, facecolor=_main_validation_flowchart_zh_RED, edgecolor='white', linewidth=0.7))
        ax.add_patch(Circle((x_core, y_core), 0.007, facecolor=_main_validation_flowchart_zh_ORANGE, edgecolor='white', linewidth=0.5))
        _main_validation_flowchart_zh_add_arrow(ax, (x_shell + 0.006 * math.cos(t), y_shell + 0.006 * math.sin(t)), (x_shell + 0.07 * math.cos(t), y_shell + 0.07 * math.sin(t)), color=_main_validation_flowchart_zh_GREEN, lw=1.5, scale=12)
    ax.add_patch(Rectangle((0.58, 0.64), 0.34, 0.14, edgecolor=_main_validation_flowchart_zh_GRAY, facecolor='#ffffff', linewidth=1.2))
    ax.text(0.75, 0.72, '只集成内表面 shell 节点\n连接 core 节点的键力', ha='center', va='center', fontsize=10.2, color=_main_validation_flowchart_zh_TEXT, fontweight='bold', linespacing=1.18)
    _main_validation_flowchart_zh_add_arrow(ax, (0.58, 0.68), (center[0] + inner * 0.82, center[1] + inner * 0.25), color=_main_validation_flowchart_zh_GRAY, lw=1.3, scale=11)
    ax.add_patch(Rectangle((0.06, 0.17), 0.025, 0.025, facecolor=_main_validation_flowchart_zh_RED, edgecolor=_main_validation_flowchart_zh_RED))
    ax.text(0.095, 0.183, '红线：转场时参与积分的 shell-core 键', fontsize=9.5, color=_main_validation_flowchart_zh_TEXT, va='center')
    ax.add_patch(Rectangle((0.06, 0.125), 0.025, 0.025, facecolor=_main_validation_flowchart_zh_GRAY, edgecolor=_main_validation_flowchart_zh_GRAY))
    ax.text(0.095, 0.138, '灰线：非内表面 shell-core 键不参与', fontsize=9.5, color=_main_validation_flowchart_zh_TEXT, va='center')
    ax.add_patch(Rectangle((0.06, 0.08), 0.025, 0.025, facecolor=_main_validation_flowchart_zh_GREEN, edgecolor=_main_validation_flowchart_zh_GREEN))
    ax.text(0.095, 0.093, '绿箭头：等效内压方向 / 残余体力方向', fontsize=9.5, color=_main_validation_flowchart_zh_TEXT, va='center')
    ax.text(0.05, 0.035, '转场载荷 = p_transfer * b_unit + b_res；后续再叠加 p_phase * b_unit', fontsize=10.2, color=_main_validation_flowchart_zh_NAVY, fontweight='bold', ha='left')

def _main_validation_flowchart_zh_draw_flowchart(ax):
    ax.text(0.03, 0.95, 'B. main_validation.py 主流程', fontsize=16, color=_main_validation_flowchart_zh_NAVY, fontweight='bold', ha='left', va='top')
    ax.text(0.03, 0.9, '热场显式步进；力学场用 ADR 收敛；相变后切换到 shell-only', fontsize=10, color=_main_validation_flowchart_zh_MUTED, ha='left', va='top')
    _main_validation_flowchart_zh_add_box(ax, 0.08, 0.8, 0.3, 0.075, '1. 参数初始化\n材料 / 几何 / 相变区间', _main_validation_flowchart_zh_ORANGE, fontsize=9.8)
    _main_validation_flowchart_zh_add_box(ax, 0.55, 0.8, 0.3, 0.075, '2. 生成四套网格\n温度、力学、shell-only', _main_validation_flowchart_zh_BLUE, fontsize=9.8)
    _main_validation_flowchart_zh_add_box(ax, 0.28, 0.69, 0.38, 0.075, '3. 构建 CSR 邻域与边属性\npartial area / shape / harmonic edge', _main_validation_flowchart_zh_GRAY, fontsize=9.5)
    _main_validation_flowchart_zh_add_box(ax, 0.28, 0.58, 0.38, 0.075, '4. 热传导步进\nH -> T，右边界镜像温度', _main_validation_flowchart_zh_TEAL, fontsize=9.8)
    _main_validation_flowchart_zh_add_diamond(ax, 0.47, 0.465, 0.34, 0.11, '5. core_temp_max\n>= Ts ?', _main_validation_flowchart_zh_YELLOW, fontsize=10)
    _main_validation_flowchart_zh_add_box(ax, 0.05, 0.34, 0.34, 0.09, '否：core-shell 力学平衡\nbr=bz=0，OSBPD + ADR', _main_validation_flowchart_zh_BLUE, fontsize=9.5)
    _main_validation_flowchart_zh_add_box(ax, 0.55, 0.34, 0.35, 0.09, '是：进入 shell-only 转场\n提取当前界面键力', _main_validation_flowchart_zh_RED, face='#fff7f7', fontsize=9.5)
    _main_validation_flowchart_zh_add_box(ax, 0.55, 0.22, 0.35, 0.09, '6. 转场力分解\np_transfer * b_unit + b_res', _main_validation_flowchart_zh_GREEN, face='#f4fff8', fontsize=9.5)
    _main_validation_flowchart_zh_add_box(ax, 0.55, 0.105, 0.35, 0.085, '7. shell-only 后续计算\np = p_transfer + p_phase', _main_validation_flowchart_zh_PURPLE, face='#faf7ff', fontsize=9.5)
    _main_validation_flowchart_zh_add_box(ax, 0.16, 0.105, 0.28, 0.085, '8. 损伤 / 裂纹判断\n输出历史、图和 JSON', _main_validation_flowchart_zh_YELLOW, face='#fffaf0', fontsize=9.5)
    _main_validation_flowchart_zh_add_arrow(ax, (0.38, 0.837), (0.55, 0.837), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.7, 0.8), (0.52, 0.765), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.47, 0.69), (0.47, 0.655), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.47, 0.58), (0.47, 0.52), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.35, 0.445), (0.3, 0.43), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.59, 0.445), (0.68, 0.43), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.72, 0.34), (0.72, 0.31), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.72, 0.22), (0.72, 0.19), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.55, 0.147), (0.44, 0.147), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.22, 0.34), (0.21, 0.62), color=_main_validation_flowchart_zh_GRAY, lw=1.2, scale=11, alpha=0.85, rad=-0.25)
    ax.text(0.13, 0.49, '下一热步', fontsize=9.4, color=_main_validation_flowchart_zh_GRAY, fontweight='bold', rotation=82)
    ax.text(0.395, 0.432, 'No', fontsize=9.2, color=_main_validation_flowchart_zh_BLUE, fontweight='bold')
    ax.text(0.575, 0.432, 'Yes', fontsize=9.2, color=_main_validation_flowchart_zh_RED, fontweight='bold')

def _main_validation_flowchart_zh_draw_transfer_detail(ax):
    ax.text(0.03, 0.95, 'C. 转场力计算子流程', fontsize=16, color=_main_validation_flowchart_zh_NAVY, fontweight='bold', ha='left', va='top')
    ax.text(0.03, 0.9, '对应代码：main_validation.py 1207-1730；shell_only_functions.py 198-607', fontsize=10, color='#b55c00', fontweight='bold', ha='left', va='top')
    _main_validation_flowchart_zh_add_box(ax, 0.07, 0.78, 0.35, 0.08, '1. 定位 shell-only\n内表面节点', _main_validation_flowchart_zh_ORANGE, fontsize=9.7)
    _main_validation_flowchart_zh_add_box(ax, 0.57, 0.78, 0.35, 0.08, '2. 映射到 core-shell\n力学网格', _main_validation_flowchart_zh_BLUE, fontsize=9.7)
    _main_validation_flowchart_zh_add_box(ax, 0.22, 0.665, 0.55, 0.075, '3. 保存转场目标位移\nU_start = shell part of U_core-shell', _main_validation_flowchart_zh_GRAY, fontsize=9.3)
    _main_validation_flowchart_zh_add_box(ax, 0.07, 0.535, 0.35, 0.085, '4. 计算 shell-only 内力\nF_shell(U_start)', _main_validation_flowchart_zh_RED, face='#fff6f6', fontsize=9.5)
    _main_validation_flowchart_zh_add_box(ax, 0.57, 0.535, 0.35, 0.085, '5. 构造位移目标等效体力\nb_eq = -F_shell(U_start)', _main_validation_flowchart_zh_PURPLE, face='#faf7ff', fontsize=9.4)
    _main_validation_flowchart_zh_add_box(ax, 0.22, 0.395, 0.55, 0.085, '6. 仅用于记录的压力分解\nb_eq = p_transfer * b_unit + b_res', _main_validation_flowchart_zh_GREEN, face='#f4fff8', fontsize=9.7)
    _main_validation_flowchart_zh_add_box(ax, 0.22, 0.26, 0.55, 0.085, '7. 初始化真实 shell 位移\nU_shell = U_start', _main_validation_flowchart_zh_YELLOW, face='#fffaf0', fontsize=9.7)
    _main_validation_flowchart_zh_add_box(ax, 0.22, 0.125, 0.55, 0.085, '8. ADR 迭代校验位移一致\n后续直接用 U_shell 计算', _main_validation_flowchart_zh_TEAL, face='#f4fffe', fontsize=9.3)
    _main_validation_flowchart_zh_add_arrow(ax, (0.42, 0.82), (0.57, 0.82), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.74, 0.78), (0.56, 0.74), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.42, 0.665), (0.29, 0.62), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.42, 0.577), (0.57, 0.577), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.74, 0.535), (0.58, 0.48), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.49, 0.395), (0.49, 0.345), color=_main_validation_flowchart_zh_NAVY)
    _main_validation_flowchart_zh_add_arrow(ax, (0.49, 0.26), (0.49, 0.21), color=_main_validation_flowchart_zh_NAVY)
    formula = '实际外载：b = (p_phase + p_transfer) * b_unit + b_res'
    ax.add_patch(Rectangle((0.05, 0.025), 0.88, 0.06, edgecolor=_main_validation_flowchart_zh_GREEN, facecolor='#ffffff', linewidth=1.3))
    ax.text(0.49, 0.055, formula, ha='center', va='center', fontsize=8.8, color=_main_validation_flowchart_zh_TEXT, linespacing=1.1)

def _main_validation_flowchart_zh_main():
    _main_validation_flowchart_zh_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': _main_validation_flowchart_zh_choose_font(), 'axes.unicode_minus': False, 'figure.facecolor': _main_validation_flowchart_zh_PANEL, 'savefig.facecolor': _main_validation_flowchart_zh_PANEL, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(16.5, 9.2), facecolor=_main_validation_flowchart_zh_PANEL)
    fig.text(0.025, 0.965, 'main_validation.py 计算流程图', fontsize=23, color=_main_validation_flowchart_zh_NAVY, fontweight='bold', ha='left', va='top')
    fig.text(0.025, 0.925, '核心路径：温度场更新 -> core-shell 力学平衡 -> 相变触发 -> shell-only 转场 -> 相变压力与损伤判断', fontsize=11.5, color=_main_validation_flowchart_zh_TEXT, fontweight='bold', ha='left', va='top')
    axes = [fig.add_axes([0.02, 0.06, 0.31, 0.82]), fig.add_axes([0.345, 0.06, 0.31, 0.82]), fig.add_axes([0.67, 0.06, 0.31, 0.82])]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='#f7f9fb', edgecolor='#e6e0d6', linewidth=1.0, zorder=-10))
    _main_validation_flowchart_zh_draw_capsule_schematic(axes[0])
    _main_validation_flowchart_zh_draw_flowchart(axes[1])
    _main_validation_flowchart_zh_draw_transfer_detail(axes[2])
    fig.savefig(_main_validation_flowchart_zh_PNG_PATH, dpi=220, bbox_inches='tight')
    fig.savefig(_main_validation_flowchart_zh_SVG_PATH, bbox_inches='tight')
    print(_main_validation_flowchart_zh_PNG_PATH)
    print(_main_validation_flowchart_zh_SVG_PATH)


# From plot_code_structure_refactor_diagram_zh.py
_code_structure_refactor_zh_OUT_DIR = Path(__file__).resolve().parent / 'figures'
_code_structure_refactor_zh_PNG_PATH = _code_structure_refactor_zh_OUT_DIR / 'code_structure_refactor_diagram_zh.png'
_code_structure_refactor_zh_SVG_PATH = _code_structure_refactor_zh_OUT_DIR / 'code_structure_refactor_diagram_zh.svg'
_code_structure_refactor_zh_NAVY = '#0b1736'
_code_structure_refactor_zh_TEXT = '#111827'
_code_structure_refactor_zh_MUTED = '#667085'
_code_structure_refactor_zh_PANEL = '#fbfaf7'
_code_structure_refactor_zh_PANEL_INNER = '#f7f9fb'
_code_structure_refactor_zh_BLUE = '#3284d8'
_code_structure_refactor_zh_LIGHT_BLUE = '#d8f1ff'
_code_structure_refactor_zh_GREEN = '#14945f'
_code_structure_refactor_zh_LIGHT_GREEN = '#dff5e9'
_code_structure_refactor_zh_ORANGE = '#f08a24'
_code_structure_refactor_zh_YELLOW = '#f6bd16'
_code_structure_refactor_zh_RED = '#df2935'
_code_structure_refactor_zh_PURPLE = '#7c3aed'
_code_structure_refactor_zh_CYAN = '#14a9d4'
_code_structure_refactor_zh_TEAL = '#0f9f9a'
_code_structure_refactor_zh_GRAY = '#7a8795'

def _code_structure_refactor_zh_choose_font():
    candidates = ['Heiti SC', 'Hiragino Sans GB', 'PingFang SC', 'Arial Unicode MS', 'Songti SC', 'Noto Sans CJK SC', 'SimHei']
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return 'DejaVu Sans'

def _code_structure_refactor_zh_semidisk_path(center, radius, theta1=-90, theta2=90):
    cx, cy = center
    n = 150
    angles = [math.radians(theta1 + (theta2 - theta1) * i / (n - 1)) for i in range(n)]
    outer = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
    verts = [(cx, cy + radius * math.sin(math.radians(theta1)))] + outer
    verts += [(cx, cy + radius * math.sin(math.radians(theta2))), (cx, cy + radius * math.sin(math.radians(theta1)))]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * len(outer)
    codes += [MplPath.LINETO, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)

def _code_structure_refactor_zh_semiannulus_path(center, outer_r, inner_r, theta1=-90, theta2=90):
    cx, cy = center
    n = 150
    outer_angles = [math.radians(theta1 + (theta2 - theta1) * i / (n - 1)) for i in range(n)]
    inner_angles = [math.radians(theta2 - (theta2 - theta1) * i / (n - 1)) for i in range(n)]
    outer = [(cx + outer_r * math.cos(a), cy + outer_r * math.sin(a)) for a in outer_angles]
    inner = [(cx + inner_r * math.cos(a), cy + inner_r * math.sin(a)) for a in inner_angles]
    verts = outer + inner + [outer[0]]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer) - 1)
    codes += [MplPath.LINETO] * len(inner) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)

def _code_structure_refactor_zh_add_arrow(ax, start, end, color=_code_structure_refactor_zh_NAVY, lw=1.6, scale=13, rad=0.0, alpha=1.0):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, linewidth=lw, color=color, shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}', alpha=alpha)
    ax.add_patch(arrow)
    return arrow

def _code_structure_refactor_zh_add_box(ax, xy, wh, text, edge, face='#ffffff', fontsize=9.8, lw=1.7):
    x, y = xy
    w, h = wh
    box = Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color=_code_structure_refactor_zh_TEXT, fontweight='bold', linespacing=1.16)
    return box

def _code_structure_refactor_zh_add_problem(ax, y, text):
    ax.add_patch(Rectangle((0.56, y), 0.39, 0.082, facecolor='#fff5f5', edgecolor=_code_structure_refactor_zh_RED, linewidth=1.25))
    ax.text(0.575, y + 0.041, text, ha='left', va='center', fontsize=8.9, color=_code_structure_refactor_zh_TEXT, linespacing=1.15)
    ax.add_patch(Circle((0.545, y + 0.041), 0.012, facecolor=_code_structure_refactor_zh_RED, edgecolor=_code_structure_refactor_zh_RED))

def _code_structure_refactor_zh_draw_old_region_panel(ax):
    ax.text(0.03, 0.95, '1. 旧结构：列表式多区域', ha='left', va='top', fontsize=15.5, color=_code_structure_refactor_zh_NAVY, fontweight='bold')
    ax.text(0.03, 0.902, '目标是把微胶囊切成多个 region，再用 list 管理', ha='left', va='top', fontsize=9.6, color=_code_structure_refactor_zh_MUTED)
    center = (0.21, 0.53)
    radius = 0.29
    inner = 0.22
    clip_patch = PathPatch(_code_structure_refactor_zh_semidisk_path(center, radius), transform=ax.transData)
    bands = [(center[1] + 0.21, center[1] + radius, _code_structure_refactor_zh_RED), (center[1] + 0.12, center[1] + 0.21, _code_structure_refactor_zh_YELLOW), (center[1] - 0.16, center[1] + 0.12, '#92cf50'), (center[1] - radius, center[1] - 0.16, _code_structure_refactor_zh_CYAN)]
    for y0, y1, color in bands:
        rect = Rectangle((center[0], y0), radius + 0.02, y1 - y0, facecolor=color, edgecolor='none', alpha=0.96)
        rect.set_clip_path(clip_patch)
        ax.add_patch(rect)
    ax.add_patch(PathPatch(_code_structure_refactor_zh_semidisk_path(center, radius), facecolor='none', edgecolor=_code_structure_refactor_zh_NAVY, linewidth=1.4))
    ax.add_patch(Arc(center, 2 * inner, 2 * inner, theta1=-90, theta2=90, edgecolor=_code_structure_refactor_zh_NAVY, linewidth=1.2))
    ax.plot([center[0], center[0]], [center[1] - radius - 0.035, center[1] + radius + 0.035], color=_code_structure_refactor_zh_NAVY, lw=1.3, ls=':')
    for yy in [center[1] + 0.21, center[1] + 0.12, center[1] - 0.16]:
        ax.plot([0.03, 0.47], [yy, yy], color='#4c72d9', lw=1.5, ls=(0, (5, 5)))
    labels = [('Region 0', (0.32, 0.78), (0.245, 0.75)), ('Region 1', (0.4, 0.7), (0.3, 0.66)), ('Region 2 ~ n-2', (0.36, 0.54), (0.34, 0.53)), ('Region n-1', (0.33, 0.31), (0.29, 0.36))]
    for text, label_xy, target_xy in labels:
        ax.text(*label_xy, text, fontsize=10.4, color=_code_structure_refactor_zh_TEXT, fontweight='bold', ha='left', va='center')
        _code_structure_refactor_zh_add_arrow(ax, label_xy, target_xy, color=_code_structure_refactor_zh_TEXT, lw=1.2, scale=11)
    _code_structure_refactor_zh_add_box(ax, (0.06, 0.12), (0.34, 0.105), 'region_list = [R0, R1, ..., Rn-1]\n温度/力学状态也按 list 拆开', _code_structure_refactor_zh_GRAY, face='#ffffff', fontsize=8.6)
    _code_structure_refactor_zh_add_problem(ax, 0.61, '密集颗粒区域与粗糙区域\n结果差异明显')
    _code_structure_refactor_zh_add_problem(ax, 0.5, '跨区邻域、映射和边属性\n增加大量代码分支')
    _code_structure_refactor_zh_add_problem(ax, 0.39, '未来含空腔薄壳结构中\n粗糙区域会产生大误差')

def _code_structure_refactor_zh_draw_current_structure_panel(ax):
    ax.text(0.03, 0.95, '2. 当前结构：删除 list 分区', ha='left', va='top', fontsize=15.5, color=_code_structure_refactor_zh_NAVY, fontweight='bold')
    ax.text(0.03, 0.902, 'main_validation.py 使用单区域网格和显式 shell-only 转场', ha='left', va='top', fontsize=9.6, color=_code_structure_refactor_zh_MUTED)
    boxes = [((0.08, 0.78), (0.34, 0.075), '单一 core-shell 网格\ncoords_t / coords_m', _code_structure_refactor_zh_BLUE), ((0.58, 0.78), (0.32, 0.075), '单一 shell-only 网格\ncoords_shell_only_*', _code_structure_refactor_zh_BLUE), ((0.25, 0.655), (0.48, 0.075), '统一 CSR 邻域与边属性\npartial area / shape / edge property', _code_structure_refactor_zh_GRAY), ((0.25, 0.535), (0.48, 0.075), '热场更新\nH -> T -> T_m / T_shell_only_m', _code_structure_refactor_zh_TEAL), ((0.25, 0.415), (0.48, 0.075), 'core-shell 力学平衡\nOSBPD + ADR', _code_structure_refactor_zh_GREEN), ((0.25, 0.285), (0.48, 0.083), '相变触发后转换\n提取内表面 shell-core 键力', _code_structure_refactor_zh_RED), ((0.25, 0.155), (0.48, 0.083), 'shell-only 后续计算\nb = (p_phase + p_transfer)*b_unit + b_res', _code_structure_refactor_zh_PURPLE)]
    for xy, wh, text, color in boxes:
        face = '#ffffff'
        if color == _code_structure_refactor_zh_RED:
            face = '#fff7f7'
        elif color == _code_structure_refactor_zh_GREEN:
            face = _code_structure_refactor_zh_LIGHT_GREEN
        elif color == _code_structure_refactor_zh_PURPLE:
            face = '#faf7ff'
        _code_structure_refactor_zh_add_box(ax, xy, wh, text, color, face=face, fontsize=8.9)
    _code_structure_refactor_zh_add_arrow(ax, (0.42, 0.817), (0.58, 0.817), color=_code_structure_refactor_zh_NAVY)
    _code_structure_refactor_zh_add_arrow(ax, (0.72, 0.78), (0.55, 0.73), color=_code_structure_refactor_zh_NAVY)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.655), (0.49, 0.61), color=_code_structure_refactor_zh_NAVY)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.535), (0.49, 0.49), color=_code_structure_refactor_zh_NAVY)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.415), (0.49, 0.368), color=_code_structure_refactor_zh_NAVY)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.285), (0.49, 0.238), color=_code_structure_refactor_zh_NAVY)
    _code_structure_refactor_zh_add_arrow(ax, (0.18, 0.445), (0.16, 0.57), color=_code_structure_refactor_zh_GRAY, lw=1.2, scale=11, rad=-0.3, alpha=0.85)
    ax.text(0.105, 0.515, '下一热步', rotation=82, fontsize=8.8, color=_code_structure_refactor_zh_GRAY, fontweight='bold')
    ax.add_patch(Rectangle((0.07, 0.035), 0.84, 0.078, facecolor='#ffffff', edgecolor=_code_structure_refactor_zh_GREEN, linewidth=1.35))
    ax.text(0.49, 0.075, '收益：状态数组更少、流程更直、转场接口清晰，避免 list 区域之间反复匹配', ha='center', va='center', fontsize=9.0, color=_code_structure_refactor_zh_TEXT, fontweight='bold')

def _code_structure_refactor_zh_draw_refined_shell(ax, center, outer_r, inner_r):
    ax.add_patch(PathPatch(_code_structure_refactor_zh_semidisk_path(center, outer_r), facecolor='#ffffff', edgecolor=_code_structure_refactor_zh_NAVY, linewidth=1.1))
    ax.add_patch(PathPatch(_code_structure_refactor_zh_semiannulus_path(center, outer_r, inner_r), facecolor=_code_structure_refactor_zh_LIGHT_BLUE, edgecolor=_code_structure_refactor_zh_BLUE, linewidth=1.6))
    ax.add_patch(Arc(center, 2 * inner_r, 2 * inner_r, theta1=-90, theta2=90, edgecolor=_code_structure_refactor_zh_RED, linewidth=1.4, linestyle='--'))
    ax.plot([center[0], center[0]], [center[1] - outer_r, center[1] + outer_r], color=_code_structure_refactor_zh_NAVY, lw=1.1)
    ax.text(center[0] + 0.075, center[1] + 0.02, 'Void / Core', ha='center', va='center', fontsize=9.5, color=_code_structure_refactor_zh_GRAY, fontweight='bold')
    ax.text(center[0] + outer_r * 0.72, center[1] - outer_r * 0.7, 'Refined\nshell', ha='center', va='center', fontsize=9.8, color=_code_structure_refactor_zh_NAVY, fontweight='bold')
    xs = np.arange(center[0] + 0.015, center[0] + outer_r + 0.001, 0.024)
    ys = np.arange(center[1] - outer_r + 0.015, center[1] + outer_r, 0.024)
    for x in xs:
        for y in ys:
            rr = math.hypot(x - center[0], y - center[1])
            if inner_r <= rr <= outer_r and x >= center[0]:
                ax.add_patch(Circle((x, y), 0.0037, facecolor=_code_structure_refactor_zh_NAVY, edgecolor='none', alpha=0.82))
    xs_core = np.arange(center[0] + 0.035, center[0] + inner_r - 0.005, 0.058)
    ys_core = np.arange(center[1] - inner_r + 0.035, center[1] + inner_r - 0.005, 0.058)
    for x in xs_core:
        for y in ys_core:
            rr = math.hypot(x - center[0], y - center[1])
            if rr < inner_r and x >= center[0]:
                ax.add_patch(Circle((x, y), 0.0035, facecolor=_code_structure_refactor_zh_GRAY, edgecolor='none', alpha=0.35))

def _code_structure_refactor_zh_draw_future_panel(ax):
    ax.text(0.03, 0.95, '3. 后续路线：只加密壳层', ha='left', va='top', fontsize=15.5, color=_code_structure_refactor_zh_NAVY, fontweight='bold')
    ax.text(0.03, 0.902, '利用当前转场结构，把加密目标集中在 shell 上', ha='left', va='top', fontsize=9.6, color=_code_structure_refactor_zh_MUTED)
    _code_structure_refactor_zh_draw_refined_shell(ax, (0.24, 0.55), 0.29, 0.205)
    for deg in [-52, -25, 0, 28, 55]:
        t = math.radians(deg)
        x0 = 0.24 + 0.205 * math.cos(t)
        y0 = 0.55 + 0.205 * math.sin(t)
        x1 = 0.24 + 0.27 * math.cos(t)
        y1 = 0.55 + 0.27 * math.sin(t)
        _code_structure_refactor_zh_add_arrow(ax, (x0, y0), (x1, y1), color=_code_structure_refactor_zh_GREEN, lw=1.3, scale=11)
    steps = [((0.55, 0.7), (0.37, 0.075), '壳层局部加密\n细颗粒描述应力和裂纹', _code_structure_refactor_zh_BLUE), ((0.55, 0.58), (0.37, 0.075), '芯区或空腔保持简单\n不再依赖多 region list', _code_structure_refactor_zh_GRAY), ((0.55, 0.46), (0.37, 0.075), '沿用转场载荷接口\np_transfer + b_res', _code_structure_refactor_zh_GREEN), ((0.55, 0.34), (0.37, 0.075), '适合薄壳 / 含空腔结构\n降低粗糙区域误差', _code_structure_refactor_zh_RED)]
    for xy, wh, text, color in steps:
        face = '#ffffff'
        if color == _code_structure_refactor_zh_RED:
            face = '#fff7f7'
        elif color == _code_structure_refactor_zh_GREEN:
            face = '#f4fff8'
        _code_structure_refactor_zh_add_box(ax, xy, wh, text, color, face=face, fontsize=9.0)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.66), (0.55, 0.735), color=_code_structure_refactor_zh_NAVY, lw=1.4, scale=12)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.56), (0.55, 0.615), color=_code_structure_refactor_zh_NAVY, lw=1.4, scale=12)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.48), (0.55, 0.495), color=_code_structure_refactor_zh_NAVY, lw=1.4, scale=12)
    _code_structure_refactor_zh_add_arrow(ax, (0.49, 0.4), (0.55, 0.375), color=_code_structure_refactor_zh_NAVY, lw=1.4, scale=12)
    ax.add_patch(Rectangle((0.06, 0.125), 0.86, 0.125, facecolor='#ffffff', edgecolor=_code_structure_refactor_zh_YELLOW, linewidth=1.45))
    ax.text(0.49, 0.188, '核心判断：我们不需要把整个微胶囊切成很多区域，\n只需要让壳层获得足够分辨率，并用清晰的转换载荷连接历史状态。', ha='center', va='center', fontsize=9.6, color=_code_structure_refactor_zh_TEXT, fontweight='bold', linespacing=1.18)
    ax.text(0.49, 0.055, '代码层面：从 region list 控制流 -> 单区域数组 + shell-only 转场模块', ha='center', va='center', fontsize=9.2, color=_code_structure_refactor_zh_NAVY, fontweight='bold')

def _code_structure_refactor_zh_main():
    _code_structure_refactor_zh_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': _code_structure_refactor_zh_choose_font(), 'axes.unicode_minus': False, 'figure.facecolor': _code_structure_refactor_zh_PANEL, 'savefig.facecolor': _code_structure_refactor_zh_PANEL, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(16.5, 9.0), facecolor=_code_structure_refactor_zh_PANEL)
    fig.text(0.025, 0.965, '代码结构重写与壳层加密路线', ha='left', va='top', fontsize=23, color=_code_structure_refactor_zh_NAVY, fontweight='bold')
    fig.text(0.025, 0.925, '从列表式区域切片，重构为单区域主流程，并为未来 shell-only 壳层局部加密保留清晰接口', ha='left', va='top', fontsize=11.3, color=_code_structure_refactor_zh_TEXT, fontweight='bold')
    axes = [fig.add_axes([0.02, 0.06, 0.31, 0.82]), fig.add_axes([0.345, 0.06, 0.31, 0.82]), fig.add_axes([0.67, 0.06, 0.31, 0.82])]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=_code_structure_refactor_zh_PANEL_INNER, edgecolor='#e6e0d6', linewidth=1.0, zorder=-10))
    _code_structure_refactor_zh_draw_old_region_panel(axes[0])
    _code_structure_refactor_zh_draw_current_structure_panel(axes[1])
    _code_structure_refactor_zh_draw_future_panel(axes[2])
    fig.savefig(_code_structure_refactor_zh_PNG_PATH, dpi=220, bbox_inches='tight')
    fig.savefig(_code_structure_refactor_zh_SVG_PATH, bbox_inches='tight')
    print(_code_structure_refactor_zh_PNG_PATH)
    print(_code_structure_refactor_zh_SVG_PATH)


# From plot_lame_fem_pd_comparison.py
_lame_fem_pd_comparison_OUTER_RADIUS = 2.4e-05
_lame_fem_pd_comparison_SHELL_THICKNESS = 6e-06
_lame_fem_pd_comparison_INNER_RADIUS = _lame_fem_pd_comparison_OUTER_RADIUS - _lame_fem_pd_comparison_SHELL_THICKNESS
_lame_fem_pd_comparison_PRESSURE = 10000000000.0
_lame_fem_pd_comparison_E_SHELL = 222720000000.0
_lame_fem_pd_comparison_NU_SHELL = 0.284
_lame_fem_pd_comparison_POINTS = [{'label': '(rshell, r)', 'radius': _lame_fem_pd_comparison_INNER_RADIUS, 'fem_col': 'Point 1 displacement (m)'}, {'label': '(r, r)', 'radius': _lame_fem_pd_comparison_OUTER_RADIUS, 'fem_col': 'Point 2 displacement (m)'}, {'label': '(r-rshell/2, r)', 'radius': _lame_fem_pd_comparison_OUTER_RADIUS - 0.5 * _lame_fem_pd_comparison_SHELL_THICKNESS, 'fem_col': 'Point 3 displacement (m)'}]
_lame_fem_pd_comparison_NS_MAIN = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
_lame_fem_pd_comparison_NS_REL = {'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'}
_lame_fem_pd_comparison_REL_ID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'

def _lame_fem_pd_comparison_safe_label(label):
    return label.replace('(', '').replace(')', '').replace(',', '_').replace('/', '_').replace('\\', '_').replace(' ', '')

def _lame_fem_pd_comparison_col_to_index(cell_ref):
    match = re.match('([A-Z]+)', cell_ref)
    if not match:
        return 0
    value = 0
    for ch in match.group(1):
        value = value * 26 + (ord(ch) - ord('A') + 1)
    return value - 1

def _lame_fem_pd_comparison_xlsx_cell_value(cell, shared_strings):
    value = cell.find('main:v', _lame_fem_pd_comparison_NS_MAIN)
    if value is None:
        inline = cell.find('main:is', _lame_fem_pd_comparison_NS_MAIN)
        if inline is None:
            return ''
        return ''.join((text.text or '' for text in inline.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')))
    raw = value.text or ''
    if cell.attrib.get('t') == 's':
        return shared_strings[int(raw)]
    return raw

def _lame_fem_pd_comparison_read_xlsx_sheet(path, sheet_name):
    path = Path(path)
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        rel_map = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels}
        sheet_target = None
        for sheet in workbook.find('main:sheets', _lame_fem_pd_comparison_NS_MAIN):
            if sheet.attrib.get('name') == sheet_name:
                target = rel_map[sheet.attrib[_lame_fem_pd_comparison_REL_ID]].lstrip('/')
                sheet_target = target if target.startswith('xl/') else f'xl/{target}'
                break
        if sheet_target is None:
            raise ValueError(f'Sheet not found: {sheet_name}')
        shared_strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared_xml = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for item in shared_xml.findall('main:si', _lame_fem_pd_comparison_NS_MAIN):
                shared_strings.append(''.join((text.text or '' for text in item.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'))))
        sheet_xml = ET.fromstring(archive.read(sheet_target))
        table = []
        for row in sheet_xml.find('main:sheetData', _lame_fem_pd_comparison_NS_MAIN).findall('main:row', _lame_fem_pd_comparison_NS_MAIN):
            values = []
            for cell in row.findall('main:c', _lame_fem_pd_comparison_NS_MAIN):
                idx = _lame_fem_pd_comparison_col_to_index(cell.attrib.get('r', 'A'))
                while len(values) <= idx:
                    values.append('')
                values[idx] = _lame_fem_pd_comparison_xlsx_cell_value(cell, shared_strings)
            table.append(values)
    return table

def _lame_fem_pd_comparison_table_to_dict_rows(table):
    headers = [str(item).strip() for item in table[0]]
    rows = []
    for raw in table[1:]:
        row = {}
        for i, header in enumerate(headers):
            row[header] = raw[i] if i < len(raw) else ''
        rows.append(row)
    return rows

def _lame_fem_pd_comparison_to_float(value):
    if value is None or value == '':
        return math.nan
    return float(value)

def _lame_fem_pd_comparison_read_pd_csv(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No rows in PD csv: {path}')
    time = np.array([_lame_fem_pd_comparison_to_float(row['Time (s)']) for row in rows], dtype=float)
    data = {}
    for point in _lame_fem_pd_comparison_POINTS:
        key = _lame_fem_pd_comparison_safe_label(point['label'])
        ur_col = f'{key}_Ur (m)'
        uz_col = f'{key}_Uz (m)'
        coord_r_col = f'{key}_coord_r'
        coord_z_col = f'{key}_coord_z'
        if ur_col not in rows[0]:
            raise ValueError(f'Missing PD column: {ur_col}')
        data[point['label']] = {'ur': np.array([_lame_fem_pd_comparison_to_float(row[ur_col]) for row in rows], dtype=float), 'uz': np.array([_lame_fem_pd_comparison_to_float(row.get(uz_col, math.nan)) for row in rows], dtype=float), 'coord_r': _lame_fem_pd_comparison_to_float(rows[-1].get(coord_r_col, math.nan)), 'coord_z': _lame_fem_pd_comparison_to_float(rows[-1].get(coord_z_col, math.nan))}
    return (time, data)

def _lame_fem_pd_comparison_read_fem_xlsx(path):
    rows = _lame_fem_pd_comparison_table_to_dict_rows(_lame_fem_pd_comparison_read_xlsx_sheet(path, 'Combined'))
    time = np.array([_lame_fem_pd_comparison_to_float(row['Time (s)']) for row in rows], dtype=float)
    data = {}
    for point in _lame_fem_pd_comparison_POINTS:
        col = point['fem_col']
        if col not in rows[0]:
            raise ValueError(f'Missing FEM column: {col}')
        data[point['label']] = np.array([_lame_fem_pd_comparison_to_float(row[col]) for row in rows], dtype=float)
    return (time, data)

def _lame_fem_pd_comparison_lame_spherical_shell_ur(radius):
    a = _lame_fem_pd_comparison_INNER_RADIUS
    b = _lame_fem_pd_comparison_OUTER_RADIUS
    denom = b ** 3 - a ** 3
    A = _lame_fem_pd_comparison_PRESSURE * a ** 3 / denom
    B = _lame_fem_pd_comparison_PRESSURE * a ** 3 * b ** 3 / denom
    return ((1.0 - 2.0 * _lame_fem_pd_comparison_NU_SHELL) * A * radius + (1.0 + _lame_fem_pd_comparison_NU_SHELL) * B / (2.0 * radius ** 2)) / _lame_fem_pd_comparison_E_SHELL

def _lame_fem_pd_comparison_write_summary(path, pd_time, pd_data, fem_data, lame_data):
    fieldnames = ['point', 'target_r_m', 'pd_coord_r_m', 'pd_coord_z_m', 'pd_time_final_s', 'pd_ur_final_m', 'pd_uz_final_m', 'fem_ur_final_m', 'lame_ur_m', 'lame_uz_m']
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in _lame_fem_pd_comparison_POINTS:
            label = point['label']
            writer.writerow({'point': label, 'target_r_m': point['radius'], 'pd_coord_r_m': pd_data[label]['coord_r'], 'pd_coord_z_m': pd_data[label]['coord_z'], 'pd_time_final_s': pd_time[-1], 'pd_ur_final_m': pd_data[label]['ur'][-1], 'pd_uz_final_m': pd_data[label]['uz'][-1], 'fem_ur_final_m': fem_data[label][-1], 'lame_ur_m': lame_data[label]['ur'], 'lame_uz_m': lame_data[label]['uz']})

def _lame_fem_pd_comparison_plot_comparison(path, pd_time, pd_data, fem_time, fem_data, lame_data):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex=False)
    scale = 1000000.0
    for col, point in enumerate(_lame_fem_pd_comparison_POINTS):
        label = point['label']
        ur_ax = axes[0, col]
        uz_ax = axes[1, col]
        ur_ax.plot(pd_time, pd_data[label]['ur'] * scale, 'o-', markersize=3.5, linewidth=1.6, label='PD')
        ur_ax.plot(fem_time, fem_data[label] * scale, '-', linewidth=1.4, label='FEM')
        ur_ax.axhline(lame_data[label]['ur'] * scale, linestyle='--', linewidth=1.6, label='Lame')
        ur_ax.set_title(f'{label}  Ur')
        ur_ax.set_xlabel('Time (s)')
        ur_ax.grid(True, alpha=0.28)
        uz_ax.plot(pd_time, pd_data[label]['uz'] * scale, 'o-', markersize=3.5, linewidth=1.6, label='PD')
        uz_ax.axhline(lame_data[label]['uz'] * scale, linestyle='--', linewidth=1.6, label='Lame')
        uz_ax.text(0.03, 0.92, 'FEM Uz not in xlsx', transform=uz_ax.transAxes, va='top', fontsize=8.5, color='0.35')
        uz_ax.set_title(f'{label}  Uz')
        uz_ax.set_xlabel('Time (s)')
        uz_ax.grid(True, alpha=0.28)
    axes[0, 0].set_ylabel('Displacement (um)')
    axes[1, 0].set_ylabel('Displacement (um)')
    axes[0, 2].legend(frameon=False, loc='best')
    axes[1, 2].legend(frameon=False, loc='best')
    fig.suptitle('PD vs FEM vs Lame displacement comparison, pressure = 1.0e10 Pa', y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=300)
    plt.close(fig)

def _lame_fem_pd_comparison_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd-csv', default='comparison_outputs/pd_tracked_points.csv')
    parser.add_argument('--fem-xlsx', default='point_results_separated.xlsx')
    parser.add_argument('--out-dir', default='comparison_outputs')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd_time, pd_data = _lame_fem_pd_comparison_read_pd_csv(args.pd_csv)
    fem_time, fem_data = _lame_fem_pd_comparison_read_fem_xlsx(args.fem_xlsx)
    lame_data = {point['label']: {'ur': _lame_fem_pd_comparison_lame_spherical_shell_ur(point['radius']), 'uz': 0.0} for point in _lame_fem_pd_comparison_POINTS}
    figure_path = out_dir / 'pd_fem_lame_ur_uz_comparison.png'
    summary_path = out_dir / 'pd_fem_lame_summary.csv'
    _lame_fem_pd_comparison_plot_comparison(figure_path, pd_time, pd_data, fem_time, fem_data, lame_data)
    _lame_fem_pd_comparison_write_summary(summary_path, pd_time, pd_data, fem_data, lame_data)
    print(f'figure: {figure_path}')
    print(f'summary: {summary_path}')
    for point in _lame_fem_pd_comparison_POINTS:
        label = point['label']
        print(f"{label}: PD Ur={pd_data[label]['ur'][-1]:.12e}, PD Uz={pd_data[label]['uz'][-1]:.12e}, FEM Ur={fem_data[label][-1]:.12e}, Lame Ur={lame_data[label]['ur']:.12e}")

def plot_current_conversion_diagram():
    """Generate figures/current_conversion_diagram.{png,svg}."""
    return _current_conversion_main()


def plot_main_validation_flowchart_zh():
    """Generate figures/main_validation_flowchart_zh.{png,svg}."""
    return _main_validation_flowchart_zh_main()


def plot_code_structure_refactor_diagram_zh():
    """Generate figures/code_structure_refactor_diagram_zh.{png,svg}."""
    return _code_structure_refactor_zh_main()


def plot_lame_fem_pd_comparison(
    pd_csv='comparison_outputs/pd_tracked_points.csv',
    fem_xlsx='point_results_separated.xlsx',
    out_dir='comparison_outputs',
):
    """Generate the PD/FEM/Lame displacement comparison figure and summary CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd_time, pd_data = _lame_fem_pd_comparison_read_pd_csv(pd_csv)
    fem_time, fem_data = _lame_fem_pd_comparison_read_fem_xlsx(fem_xlsx)
    lame_data = {
        point['label']: {
            'ur': _lame_fem_pd_comparison_lame_spherical_shell_ur(point['radius']),
            'uz': 0.0,
        }
        for point in _lame_fem_pd_comparison_POINTS
    }
    figure_path = out_dir / 'pd_fem_lame_ur_uz_comparison.png'
    summary_path = out_dir / 'pd_fem_lame_summary.csv'
    _lame_fem_pd_comparison_plot_comparison(
        figure_path,
        pd_time,
        pd_data,
        fem_time,
        fem_data,
        lame_data,
    )
    _lame_fem_pd_comparison_write_summary(
        summary_path,
        pd_time,
        pd_data,
        fem_data,
        lame_data,
    )
    print(f'figure: {figure_path}')
    print(f'summary: {summary_path}')
    return figure_path, summary_path

# ---- End merged legacy plot scripts ----
