import numpy as np
from numba import njit, prange
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
        # 2D cell
        x_left = x_center - 0.5 * dx
        z_down = z_center - 0.5 * dz
        step_x = dx / sub
        step_z = dz / sub

        count_in = 0
        total_pts = sub * sub

        for ix in range(sub):
            for iz in range(sub):
                x_samp = x_left + (ix + 0.5) * step_x
                z_samp = z_down + (iz + 0.5) * step_z
                dist2 = (x_samp - cx) ** 2 + (z_samp - cz) ** 2
                if dist2 <= delta ** 2 + tolerance:
                    count_in += 1

        frac = count_in / total_pts
        return (dx * dz) * frac



def compute_partial_area_matrix(
        x_flat, z_flat,
        dx, dz,
        delta,
        indptr, indices, dist,   # ✅ 用 CSR 替代 distance_matrix
        tolerance,
):

    x_flat = np.asarray(x_flat)
    z_flat = np.asarray(z_flat)

    N = x_flat.size

    # 早排除阈值（与你原函数一致）
    cutoff_exclude = delta + 0.5 * np.sqrt(dx**2 + dz**2) + tolerance
    nnz = indices.size
    partial_area_flat = np.zeros(nnz, dtype=float)

    for i in range(N):
        cx = x_flat[i]
        cz = z_flat[i]

        a, b = indptr[i], indptr[i + 1]
        if a == b:
            continue

        js = indices[a:b]
        dij = dist[a:b]

        for t, j in enumerate(js):
            d = dij[t]

            # 早排除：太远则面积=0
            if d > cutoff_exclude:
                partial_area_flat[a + t] = 0.0
                continue

            # cell j 的中心
            xj = x_flat[j]
            zj = z_flat[0] if dz == 0 else z_flat[j]

            corners_j = get_cell_corners(xj, zj, dx, dz)
            status = is_all_in_out(corners_j, cx, cz, delta, tolerance)

            if status == "all_in":
                partial_area_flat[a + t] = dx if dz == 0 else dx * dz
            elif status == "all_out":
                partial_area_flat[a + t] = 0.0
            else:
                partial_area_flat[a + t] = partial_area_of_cell_in_circle(
                    xj, zj, dx, dz,
                    cx, cz, delta,
                    tolerance
                )
    return partial_area_flat


@njit(parallel=True, fastmath=True)
def compute_partial_area_flat_csr_numba(
    x_flat, z_flat, dx, dz, delta, tolerance,
    indptr, indices, dist,
    sx, sz
):
    N = x_flat.size
    nnz = indices.size
    out = np.zeros(nnz, dtype=np.float64)

    # 常量
    diag = (dx*dx + dz*dz) ** 0.5
    cutoff_exclude = delta + 0.5 * diag + tolerance
    delta2 = delta*delta + tolerance

    sub = sx.size  # = 10

    if dz == 0.0:
        # -------- 1D 情况 --------
        z0 = z_flat[0]

        for i in prange(N):
            cx = x_flat[i]
            cz = z0

            a = indptr[i]
            b = indptr[i+1]
            if a == b:
                continue

            for p in range(a, b):
                j = indices[p]
                d = dist[p]
                if d > cutoff_exclude:
                    out[p] = 0.0
                    continue

                xj = x_flat[j]
                # 1D cell endpoints
                x_left = xj - 0.5*dx

                # corners (2 endpoints)
                # all_in/all_out/partial
                # left
                dl = (x_left - cx)
                in_l = (dl*dl) <= delta2
                # right
                dr_ = (x_left + dx - cx)
                in_r = (dr_*dr_) <= delta2

                if in_l and in_r:
                    out[p] = dx
                elif (not in_l) and (not in_r):
                    out[p] = 0.0
                else:
                    # partial: sampling along x
                    count_in = 0
                    for ix in range(sub):
                        x_samp = x_left + sx[ix]*dx
                        dd = x_samp - cx
                        if (dd*dd) <= delta2:
                            count_in += 1
                    out[p] = dx * (count_in / sub)

        return out

    else:
        # -------- 2D 情况 --------
        for i in prange(N):
            cx = x_flat[i]
            cz = z_flat[i]

            a = indptr[i]
            b = indptr[i+1]
            if a == b:
                continue

            for p in range(a, b):
                j = indices[p]
                d = dist[p]
                if d > cutoff_exclude:
                    out[p] = 0.0
                    continue

                xj = x_flat[j]
                zj = z_flat[j]

                # cell corners (no list/tuple)
                x_left  = xj - 0.5*dx
                x_right = xj + 0.5*dx
                z_down  = zj - 0.5*dz
                z_up    = zj + 0.5*dz

                # corner in/out flags
                dx1 = x_left  - cx; dz1 = z_down - cz
                in1 = (dx1*dx1 + dz1*dz1) <= delta2
                dx2 = x_left  - cx; dz2 = z_up   - cz
                in2 = (dx2*dx2 + dz2*dz2) <= delta2
                dx3 = x_right - cx; dz3 = z_down - cz
                in3 = (dx3*dx3 + dz3*dz3) <= delta2
                dx4 = x_right - cx; dz4 = z_up   - cz
                in4 = (dx4*dx4 + dz4*dz4) <= delta2

                all_in = in1 and in2 and in3 and in4
                any_in = in1 or in2 or in3 or in4

                if all_in:
                    out[p] = dx * dz
                elif not any_in:
                    out[p] = 0.0
                else:
                    # partial: sub×sub sampling
                    count_in = 0
                    for ix in range(sub):
                        x_samp = x_left + sx[ix]*dx
                        dxs = x_samp - cx
                        for iz in range(sub):
                            z_samp = z_down + sz[iz]*dz
                            dzs = z_samp - cz
                            if (dxs*dxs + dzs*dzs) <= delta2:
                                count_in += 1
                    out[p] = (dx * dz) * (count_in / (sub*sub))

        return out
