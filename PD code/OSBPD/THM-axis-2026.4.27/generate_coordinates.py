import numpy as np
import matplotlib.pyplot as plt

def compute_layer_dr_r_nr(
        r, n_slices, dr1, dr2, dr3, dr_l,
        len1, len2, len3, size, ghost_node,
        only_one_slice=False,
        delta_factor=None,
):
    """
    Calculate dr, delta, r, Nr, and length for each region.

    If only_one_slice=True:
        generate only one region with
            dr = 0.6 * size
            length = 2 * r
    """
    # ---------------- single-slice mode ----------------
    if delta_factor is None:
        delta_factor = ghost_node

    if only_one_slice:
        dr = dr1
        Nr = int(r / dr + 1e-12)
        delta = delta_factor * dr

        results = [{
            "layer": 0,
            "dr": dr,
            "delta": delta,
            "r": r,
            "Nr": Nr,
            "length": 2 * r
        }]

        n_slices = 1
        print(f"final n_slices={n_slices}")
        return results, n_slices

    # ---------------- normal multi-slice mode ----------------
    assert n_slices >= 3, "At least three layers are required to use dr1, dr2, dr3"

    for _ in range(100):  # avoid infinite loop
        remaining_layers = n_slices - 3
        used_height = len1 + len2 + len3
        remaining_height = 2 * r - used_height
        if remaining_height <= 0:
            raise ValueError("len1 + len2 + len3 cannot exceed total height")

        # 把高度换算成“size”的整数个数，避免浮点误差
        remaining_height_um = int(round(remaining_height / size))

        # 均分（整除）+ 余数，都是整数
        len_rest_um, leftover_um = divmod(remaining_height_um, remaining_layers)

        # 只要求余数为偶数；允许 len_rest_um 是奇数
        if leftover_um % 2 != 0:
            n_slices += 1
            continue
        break
    else:
        raise RuntimeError("Failed to find a valid n_slices within 100 iterations")

    results = []
    lock_dr = False  # Lock dr to dr_l if reached

    for i in range(n_slices):
        if i == 0:
            dr = dr1
            length = len1
        elif i == 1:
            dr = dr2
            length = len2
        elif i == 2:
            dr = dr3
            length = len3
        else:
            # Determine dr
            if not lock_dr:
                dr_candidate = dr3 * (2 ** (i - 2))
                if dr_candidate >= dr_l:
                    dr = dr_l
                    lock_dr = True
                else:
                    dr = dr_candidate
            else:
                dr = dr_l

            # Assign length (convert back to meters)
            if i < n_slices - 1:
                length = len_rest_um * size
            else:
                length = (len_rest_um + leftover_um) * size

        Nr = int(r / dr + 1e-12)
        delta = delta_factor * dr
        results.append({
            "layer": i,
            "dr": dr,
            "delta": delta,
            "r": r,
            "Nr": Nr,
            "length": length
        })

    total_height = sum(layer["length"] for layer in results)
    assert abs(total_height - 2 * r) < 1e-12, "Total height does not match 2*r"
    print(f"final n_slices={n_slices}")

    return results, n_slices

from math import fsum
from decimal import Decimal, getcontext, ROUND_HALF_UP
import numpy as np
import matplotlib.pyplot as plt

def generate_one_slice_coordinates(
        R, Nr, ghost_nodes_r, zones,
        r_ghost_left, r_ghost_right,
        r_ghost_top, r_ghost_bot,
        n_slices,
        slice_id,
        graph,
        r_start   # 新增：几何在 r 方向的起始位置（也是半圆最左端）
):
    getcontext().prec = 50  # 高精度十进制计算

    def _quantize_step(x, step):
        """按 step 的十进制刻度量化到最近整数倍，返回 float。"""
        xd = Decimal(str(x))
        sd = Decimal(str(step))
        return float((xd / sd).to_integral_value(rounding=ROUND_HALF_UP) * sd)

    def snap_until_clean(x, step, tol=1e-18, max_iter=5):
        """循环量化，直到 |x - round(x/step)*step| <= tol，否则报错。"""
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

    dz_layer = zones[slice_id]["length"]
    dr = zones[slice_id]["dr"]
    # 新圆心
    r_center = r_start
    z_center = R

    # r方向总范围（含ghost）
    # 物理区中心点范围：r_start+dr/2 到 r_start+(Nr-1/2)dr
    r_all = np.linspace(
        r_start - ghost_nodes_r * dr + dr / 2,
        r_start + Nr * dr - dr / 2 + ghost_nodes_r * dr,
        Nr + 2 * ghost_nodes_r
    )

    # z方向
    z_bot = snap_until_clean(2 * R - fsum(zone['length'] for zone in zones[:slice_id + 1]), dr) - 1e-14
    z_top = snap_until_clean(z_bot + zones[slice_id]['length'], dr) - 1e-14
    Nz = int((dz_layer + 1e-12) / dr)

    z_all = np.linspace(
        z_bot - ghost_nodes_r * dr + dr / 2,
        z_top + dr / 2 + (ghost_nodes_r - 1) * dr,
        Nz + 2 * ghost_nodes_r
    )

    # Full coordinate grid
    rr, zz = np.meshgrid(r_all, z_all, indexing='xy')
    coords_all = np.column_stack([rr.ravel(), zz.ravel()])

    # 相对新圆心的半径平方
    dist2_from_center = (coords_all[:, 0] - r_center) ** 2 + (coords_all[:, 1] - z_center) ** 2

    # Main area mask
    mask_phys = (
            (coords_all[:, 0] >= r_start) &
            (coords_all[:, 1] >= z_bot) & (coords_all[:, 1] <= z_top) &
            (dist2_from_center <= R ** 2)
    )
    coords_phys = coords_all[mask_phys]

    # Left ghost
    if r_ghost_left:
        mask_left = (
            (coords_all[:, 0] < r_start) &
            (coords_all[:, 1] >= z_bot - ghost_nodes_r * dr) &
            (coords_all[:, 1] <= z_top + ghost_nodes_r * dr)
        )
        ghost_dict['left'] = coords_all[mask_left]

    # Bottom ghost
    if r_ghost_bot and slice_id != n_slices - 1:
        mask_bot = (
                (coords_all[:, 1] < z_bot) &
                (coords_all[:, 0] >= r_start) &
                (dist2_from_center <= R ** 2)
        )
        ghost_dict['bot'] = coords_all[mask_bot]

    # Top ghost
    if r_ghost_top and slice_id != 0:
        mask_top = (
                (coords_all[:, 1] > z_top) &
                (coords_all[:, 0] >= r_start) &
                (dist2_from_center <= R ** 2)
        )
        ghost_dict['top'] = coords_all[mask_top]

    # Right ghost
    if r_ghost_right:
        if slice_id == 0:
            mask_right = (
                    (coords_all[:, 0] >= r_start) &
                    (dist2_from_center > R ** 2) &
                    (dist2_from_center < (R + ghost_nodes_r * dr) ** 2)
            )
        elif slice_id == n_slices - 1:
            mask_right = (
                    (coords_all[:, 0] >= r_start) &
                    (dist2_from_center > R ** 2) &
                    (dist2_from_center < (R + ghost_nodes_r * dr) ** 2)
            )
        else:
            mask_right = (
                    (coords_all[:, 0] >= r_start) &
                    (coords_all[:, 1] > z_bot - ghost_nodes_r * dr) &
                    (coords_all[:, 1] < z_top + ghost_nodes_r * dr) &
                    (dist2_from_center > R ** 2) &
                    (dist2_from_center < (R + ghost_nodes_r * dr) ** 2)
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

        # 可选：画出圆心
        plt.scatter([r_center], [z_center], color='red', s=30, label='Circle center')

        plt.gca().set_aspect('equal')
        plt.xlabel('r')
        plt.ylabel('z')
        plt.title(
            f'Slice {slice_id}, Z ∈ ({z_bot:.2e}, {z_top:.2e}), '
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

