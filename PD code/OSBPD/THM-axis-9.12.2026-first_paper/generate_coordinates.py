import numpy as np
import matplotlib.pyplot as plt
import geometry_utils as geom

def compute_layer_dr_r_nr(
        r, dr, ghost_node,
        r_inner=0.0,
        generate_core=True,
        modify_coordinates=False,
        coordinate_x_scale=geom.DEFAULT_COORDINATE_X_SCALE,
):
    """
    Calculate dr, delta, r, Nr, and length for the single region.
    """
    r_inner = float(r_inner)
    generate_core = bool(generate_core)
    radius = float(r)
    dr = float(dr)
    coordinate_x_scale = float(coordinate_x_scale)

    if modify_coordinates:
        outer_axis_x = coordinate_x_scale * radius
        outer_axis_z = radius
    else:
        outer_axis_x = radius
        outer_axis_z = radius

    if r_inner > 0.0:
        shell_thickness = radius - r_inner
        if modify_coordinates:
            inner_axis_x = coordinate_x_scale * radius - shell_thickness
            inner_axis_z = radius - shell_thickness
        else:
            inner_axis_x = radius - shell_thickness
            inner_axis_z = radius - shell_thickness
    else:
        inner_axis_x, inner_axis_z = 0.0, 0.0

    Nr = int(outer_axis_x / dr + 1e-12)
    delta = ghost_node * dr
    results = [{
        "layer": 0,
        "dr": dr,
        "delta": delta,
        "r": r,
        "r_inner": r_inner,
        "generate_core": generate_core,
        "modify_coordinates": bool(modify_coordinates),
        "coordinate_x_scale": coordinate_x_scale,
        "outer_axis_x": outer_axis_x,
        "outer_axis_z": outer_axis_z,
        "inner_axis_x": inner_axis_x,
        "inner_axis_z": inner_axis_z,
        "Nr": Nr,
        "length": 2 * r
    }]

    return results

from decimal import Decimal, getcontext, ROUND_HALF_UP
import numpy as np
import matplotlib.pyplot as plt

def generate_one_slice_coordinates(
        R, Nr, ghost_nodes_r, zone,
        r_ghost_left, r_ghost_right,
        graph,
        r_start,
        modify_coordinates=None,
        coordinate_x_scale=None,
):
    getcontext().prec = 50  # 楂樼簿搴﹀崄杩涘埗璁＄畻

    def _quantize_step(x, step):
        """Quantize x to the nearest integer multiple of step."""
        xd = Decimal(str(x))
        sd = Decimal(str(step))
        return float((xd / sd).to_integral_value(rounding=ROUND_HALF_UP) * sd)

    def snap_until_clean(x, step, tol=1e-18, max_iter=5):
        """Snap x to the grid until the residual is within tolerance."""
        for _ in range(max_iter):
            x = _quantize_step(x, step)
            k = round(x / step)
            resid = abs(x - k * step)
            if resid <= tol:
                return x
        raise RuntimeError(f"Failed to snap value {x} to grid with step={step} within tol={tol}")

    ghost_dict = {
        'left': np.empty((0, 2)),
        'right': np.empty((0, 2)),
        'top': np.empty((0, 2)),
        'bot': np.empty((0, 2)),
    }

    dz_layer = zone["length"]
    dr = zone["dr"]
    r_inner = float(zone.get("r_inner", 0.0))
    generate_core = bool(zone.get("generate_core", True))
    if modify_coordinates is None:
        modify_coordinates = bool(zone.get("modify_coordinates", False))
    if coordinate_x_scale is None:
        coordinate_x_scale = float(
            zone.get("coordinate_x_scale", geom.DEFAULT_COORDINATE_X_SCALE)
        )

    r_center = r_start
    z_center = R
    radius = float(R)
    coordinate_x_scale = float(coordinate_x_scale)
    if modify_coordinates:
        default_outer_axis_x = coordinate_x_scale * radius
        default_outer_axis_z = radius
    else:
        default_outer_axis_x = radius
        default_outer_axis_z = radius
    outer_axis_x = float(zone.get("outer_axis_x", default_outer_axis_x))
    outer_axis_z = float(zone.get("outer_axis_z", default_outer_axis_z))
    if r_inner > 0.0:
        shell_thickness = radius - r_inner
        if modify_coordinates:
            default_inner_axis_x = coordinate_x_scale * radius - shell_thickness
            default_inner_axis_z = radius - shell_thickness
        else:
            default_inner_axis_x = radius - shell_thickness
            default_inner_axis_z = radius - shell_thickness
        inner_axis_x = float(zone.get("inner_axis_x", default_inner_axis_x))
        inner_axis_z = float(zone.get("inner_axis_z", default_inner_axis_z))
    else:
        inner_axis_x, inner_axis_z = 0.0, 0.0


    r_all = np.linspace(
        r_start - ghost_nodes_r * dr + dr / 2,
        r_start + Nr * dr - dr / 2 + ghost_nodes_r * dr,
        Nr + 2 * ghost_nodes_r
    )


    z_bot = snap_until_clean(0.0, dr) - 1e-14
    z_top = snap_until_clean(dz_layer, dr) - 1e-14
    Nz = int((dz_layer + 1e-12) / dr)

    z_all = np.linspace(
        z_bot - ghost_nodes_r * dr + dr / 2,
        z_top + dr / 2 + (ghost_nodes_r - 1) * dr,
        Nz + 2 * ghost_nodes_r
    )

    # Full coordinate grid
    rr, zz = np.meshgrid(r_all, z_all, indexing='xy')
    coords_all = np.column_stack([rr.ravel(), zz.ravel()])

    # Level <= 1 means the point is inside the circular/elliptical boundary.
    outer_level = geom.ellipse_level(
        coords_all[:, 0], coords_all[:, 1],
        r_center, z_center,
        outer_axis_x, outer_axis_z,
    )
    if generate_core or r_inner <= 0.0:
        mask_core_allowed = np.ones(coords_all.shape[0], dtype=bool)
    else:
        inner_level = geom.ellipse_level(
            coords_all[:, 0], coords_all[:, 1],
            r_center, z_center,
            inner_axis_x, inner_axis_z,
        )
        mask_core_allowed = inner_level >= 1.0

    # Main area mask
    mask_phys = (
            (coords_all[:, 0] >= r_start) &
            (coords_all[:, 1] >= z_bot) & (coords_all[:, 1] <= z_top) &
            (outer_level <= 1.0) &
            mask_core_allowed
    )
    coords_phys = coords_all[mask_phys]

    # Left ghost
    if r_ghost_left:
        mask_left = (
            (coords_all[:, 0] < r_start) &
            (coords_all[:, 1] >= z_bot - ghost_nodes_r * dr) &
            (coords_all[:, 1] <= z_top + ghost_nodes_r * dr) &
            mask_core_allowed
        )
        ghost_dict['left'] = coords_all[mask_left]

    # Right ghost
    if r_ghost_right:
        expanded_outer_level = geom.ellipse_level(
            coords_all[:, 0], coords_all[:, 1],
            r_center, z_center,
            outer_axis_x + ghost_nodes_r * dr,
            outer_axis_z + ghost_nodes_r * dr,
        )
        mask_right = (
                (coords_all[:, 0] >= r_start) &
                (outer_level > 1.0) &
                (expanded_outer_level < 1.0) &
                mask_core_allowed
        )
        ghost_dict['right'] = coords_all[mask_right]

    total_points = len(coords_phys) + sum(len(coords) for coords in ghost_dict.values())

    # Visualization
    if graph:
        plt.figure(figsize=(6, 6))
        plt.scatter(coords_phys[:, 0], coords_phys[:, 1], s=8, label='Physical', color='blue')

        color_map = {
            'left': 'green',
            'right': 'orange',
            'top': 'purple',
            'bot': 'cyan',
        }
        for key, coords in ghost_dict.items():
            if len(coords) > 0:
                plt.scatter(coords[:, 0], coords[:, 1], s=8, label=f'Ghost {key}', color=color_map[key])


        plt.scatter([r_center], [z_center], color='red', s=30, label='Circle center')

        plt.gca().set_aspect('equal')
        plt.xlabel('r')
        plt.ylabel('z')
        plt.title(
            f'Single region, Z 鈭?({z_bot:.2e}, {z_top:.2e}), '
            f'center=({r_center:.2e}, {z_center:.2e})'
        )
        plt.legend(
            title=f'Total particles: {total_points}',
            loc='center left',
            bbox_to_anchor=(1.05, 0.5)
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    ghost_coords = np.vstack([
        ghost_dict['left'],
        ghost_dict['right'],
        ghost_dict['top'],
        ghost_dict['bot'],
    ])

    return coords_phys, ghost_coords, total_points, ghost_dict

