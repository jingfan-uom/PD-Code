import numpy as np


DEFAULT_COORDINATE_X_SCALE = 1.05


def capsule_axes(
        radius,
        dshell=0.0,
        modify_coordinates=False,
        coordinate_x_scale=DEFAULT_COORDINATE_X_SCALE,
        inner=False,
):
    radius = float(radius)
    dshell = float(dshell)

    if modify_coordinates:
        axis_x = float(coordinate_x_scale) * radius
        axis_z = radius
        if inner:
            axis_x -= dshell
            axis_z -= dshell
    else:
        if inner:
            radius -= dshell
        axis_x = radius
        axis_z = radius

    return axis_x, axis_z


def ellipse_level(x, z, center_x, center_z, axis_x, axis_z):
    x = np.asarray(x)
    z = np.asarray(z)
    return ((x - float(center_x)) / float(axis_x)) ** 2 + (
        (z - float(center_z)) / float(axis_z)
    ) ** 2


def capsule_level(
        coords,
        radius,
        dshell=0.0,
        center_x=0.0,
        center_z=None,
        modify_coordinates=False,
        coordinate_x_scale=DEFAULT_COORDINATE_X_SCALE,
        inner=False,
):
    coords = np.asarray(coords)
    if center_z is None:
        center_z = radius
    axis_x, axis_z = capsule_axes(
        radius,
        dshell=dshell,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
        inner=inner,
    )
    return ellipse_level(coords[:, 0], coords[:, 1], center_x, center_z, axis_x, axis_z)


def core_mask_from_coords(
        coords,
        outer_radius,
        dshell,
        center_x=0.0,
        center_z=None,
        modify_coordinates=False,
        coordinate_x_scale=DEFAULT_COORDINATE_X_SCALE,
):
    return capsule_level(
        coords,
        outer_radius,
        dshell=dshell,
        center_x=center_x,
        center_z=center_z,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
        inner=True,
    ) < 1.0


def outer_surface_mask_from_coords(
        coords,
        outer_radius,
        dr,
        dshell=0.0,
        center_x=0.0,
        center_z=None,
        modify_coordinates=False,
        coordinate_x_scale=DEFAULT_COORDINATE_X_SCALE,
):
    coords = np.asarray(coords)
    axis_x, axis_z = capsule_axes(
        outer_radius,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
    )
    level = capsule_level(
        coords,
        outer_radius,
        center_x=center_x,
        center_z=center_z,
        modify_coordinates=modify_coordinates,
        coordinate_x_scale=coordinate_x_scale,
    )
    band = abs(float(dr)) / max(min(axis_x, axis_z), 1.0e-30)
    mask = (level >= (1.0 - band) ** 2) & (level <= (1.0 + band) ** 2)
    if dshell > 0.0:
        mask &= ~core_mask_from_coords(
            coords,
            outer_radius,
            dshell,
            center_x=center_x,
            center_z=center_z,
            modify_coordinates=modify_coordinates,
            coordinate_x_scale=coordinate_x_scale,
        )
    return mask


def capsule_outward_normals(
        x,
        z,
        center_x,
        center_z,
        axis_x,
        axis_z,
):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nx = (x - float(center_x)) / (float(axis_x) ** 2)
    nz = (z - float(center_z)) / (float(axis_z) ** 2)
    norm = np.maximum(np.sqrt(nx ** 2 + nz ** 2), 1.0e-30)
    return np.column_stack([nx / norm, nz / norm])
