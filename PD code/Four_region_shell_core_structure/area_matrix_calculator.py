import numpy as np
from numba import jit
@jit(nopython=True)
def get_cell_corners(x_center, z_center, dx, dz):
    """
    Given the cell center (x_center, z_center) and cell dimensions (dx, dz),
    return the coordinates of the corners (2D) or endpoints (1D).
    """
    if dz == 0:
        # 1D case: only left and right endpoints
        x_left = x_center - 0.5 * dx
        x_right = x_center + 0.5 * dx
        corners = [
            (x_left, z_center),
            (x_right, z_center),
        ]
    else:
        # 2D case: four corners of the rectangle
        x_left = x_center - 0.5 * dx
        x_right = x_center + 0.5 * dx
        z_down = z_center - 0.5 * dz
        z_up = z_center + 0.5 * dz
        corners = [
            (x_left, z_down),
            (x_left, z_up),
            (x_right, z_down),
            (x_right, z_up),
        ]
    return corners


def is_all_in_out(corners, cx, cz, delta, tolerance):
    """
    Determine the spatial relationship between the given corners and a circle
    centered at (cx, cz) with radius delta.

    Returns:
        - "all_in"   -> all corners are inside the circle
        - "all_out"  -> all corners are outside the circle
        - "partial"  -> some corners are inside and some are outside
    """
    in_flags = []
    for (x_pt, z_pt) in corners:
        dist2 = (x_pt - cx) ** 2 + (z_pt - cz) ** 2
        in_flags.append(dist2 <= delta ** 2 + tolerance)

    if all(in_flags):
        return "all_in"
    elif not any(in_flags):
        return "all_out"
    else:
        return "partial"

@jit(nopython=True)
def partial_area_of_cell_in_circle(
        x_center, z_center,
        dx, dz,
        cx, cz,
        delta,
        tolerance):
    """
    Estimate the overlapped area between a grid cell and a circle (partial intersection),
    using subgrid sampling.

    Parameters:
        - dz = 0 => 1D segment sampling
        - dz ≠ 0 => 2D cell sampling
        - sub: number of subdivisions in each direction
    """
    sub = 10
    if dz == 0:
        # 1D segment
        x_left = x_center - 0.5 * dx
        step_x = dx / sub
        count_in = 0
        for ix in range(sub):
            x_samp = x_left + (ix + 0.5) * step_x
            dist2 = (x_samp - cx) ** 2
            if dist2 <= delta ** 2 + tolerance:
                count_in += 1
        frac = count_in / sub
        return dx * frac

    else:
        sub = 10  # Number of subdivisions per dimension (10x10 sampling grid for 2D cell)
        x_left = x_center - 0.5 * dx  # Compute the x-coordinate of the left edge of the cell
        z_down = z_center - 0.5 * dz  # Compute the z-coordinate of the bottom edge of the cell
        step_x = dx / sub  # Width of each subcell in the x-direction
        step_z = dz / sub  # Height of each subcell in the z-direction
        count_in = 0  # Counter for the number of sampling points falling inside the circle
        total_pts = sub * sub  # Total number of sampling points (subgrid resolution)
        # Loop over all subcells within the cell
        for ix in range(sub):
            for iz in range(sub):
                # Compute the (x, z) coordinates of the subcell center
                x_samp = x_left + (ix + 0.5) * step_x
                z_samp = z_down + (iz + 0.5) * step_z
                # Compute squared distance from sampling point to the center of the circle
                dist2 = (x_samp - cx) ** 2 + (z_samp - cz) ** 2
                # Check if the sampling point lies within the circular horizon (radius delta)
                if dist2 <= delta ** 2 + tolerance:
                    count_in += 1  # Increment the count of inside points
        # Compute the fraction of the cell area that lies within the circle
        frac = count_in / total_pts
        # Return the estimated overlapping area using the fractional coverage
        return (dx * dz) * frac



def compute_partial_area_matrix(
        x_flat, z_flat,
        dx, dz,
        delta,
        distance_matrix,
        tolerance
):
    """
    Compute the area overlap matrix between all nodes and their neighbors
    based on corner geometry and partial area sampling.

    For each (i, j):
      1) If distance > delta + diagonal/2 + tolerance => definitely outside
      2) Otherwise:
         - Use get_cell_corners() to get the four corners of cell j
         - Use is_all_in_out() to classify the corner status
         - If all_in   => full cell area (dx * dz or dx in 1D)
         - If all_out  => 0
         - If partial  => compute area using partial_area_of_cell_in_circle()
    """
    N = len(x_flat)
    area_mat = np.zeros((N, N), dtype=float)
    print("Calculating partial area matrix...")
    for i in range(N):
        cx = x_flat[i]
        cz = z_flat[0] if dz == 0 else z_flat[i]
        for j in range(N):
            dist = distance_matrix[i, j]

            if dist > delta + 0.5 * np.sqrt(dx**2 + dz**2) + tolerance:
                # Early exclusion: cell j is too far from point i
                area_mat[i, j] = 0.0
                continue

            # Get the center of cell j
            xj = x_flat[j]
            zj = z_flat[0] if dz == 0 else z_flat[j]

            # Compute corners of cell j
            corners_j = get_cell_corners(xj, zj, dx, dz)

            # Check relationship between corners and horizon
            status = is_all_in_out(corners_j, cx, cz, delta, tolerance)

            if status == "all_in":
                area_mat[i, j] = dx if dz == 0 else dx * dz
            elif status == "all_out":
                area_mat[i, j] = 0.0
            else:
                area_mat[i, j] = partial_area_of_cell_in_circle(
                    xj, zj, dx, dz,
                    cx, cz, delta,
                    tolerance
                )
    print("Partial area matrix calculation completed.")
    return area_mat
