import numpy as np
import matplotlib.pyplot as plt

def generate_coordinates(
    r_start, z_start, dr, dz, Lr, Lz, Nr, Nz,
    ghost_nodes_x, ghost_nodes_z,
    r_ghost_left, r_ghost_right,
    z_ghost_top, z_ghost_bot
):
    """
    Generate r and z coordinate arrays including ghost regions.
    Each ghost region can be selectively added based on boolean flags.

    Parameters:
    - r_start, z_start: starting coordinates of the physical domain
    - dr, dz: grid spacing in r and z directions
    - Lr, Lz: length of the physical domain in r and z directions
    - Nr, Nz: number of grid nodes in the physical domain
    - ghost_nodes_x, ghost_nodes_z: number of ghost nodes in r and z directions
    - r_ghost_left, r_ghost_right, z_ghost_top, z_ghost_bot:
        booleans indicating whether to include ghost regions on each side

    Returns:
    - r_all, z_all: coordinate arrays including ghost nodes
    - Nr_tot, Nz_tot: total number of nodes including ghost regions
    """

    # r-direction (horizontal)
    r_phys = np.linspace(r_start + dr / 2, r_start + Lr - dr / 2, Nr)

    r_parts = []
    if r_ghost_left:
        r_ghost_l = np.linspace(r_start - ghost_nodes_x * dr + dr / 2, r_start - dr / 2, ghost_nodes_x)
        r_parts.append(r_ghost_l)
    r_parts.append(r_phys)
    if r_ghost_right:
        r_ghost_r = np.linspace(
            r_start + Lr + dr / 2,
            r_start + Lr + dr / 2 + (ghost_nodes_x - 1) * dr,
            ghost_nodes_x
        )
        r_parts.append(r_ghost_r)

    r_all = np.concatenate(r_parts)
    Nr_tot = len(r_all)

    # z-direction (vertical, reversed order: top to bottom)
    z_phys = np.linspace(z_start + Lz - dz / 2, z_start + dz / 2, Nz)

    z_parts = []
    if z_ghost_top:
        z_ghost_t = np.linspace(
            z_start + Lz + (ghost_nodes_z - 1) * dz + dz / 2,
            z_start + Lz + dz / 2,
            ghost_nodes_z
        )
        z_parts.append(z_ghost_t)
    z_parts.append(z_phys)
    if z_ghost_bot:
        z_ghost_b = np.linspace(
            z_start - dz / 2,
            z_start - ghost_nodes_z * dz + dz / 2,
            ghost_nodes_z
        )
        z_parts.append(z_ghost_b)

    z_all = np.concatenate(z_parts)
    Nz_tot = len(z_all)

    return r_all, z_all, Nr_tot, Nz_tot



def generate_half_circle_coordinates(
    dr, R, Nr, ghost_nodes_r,
    r_ghost_left,
    half_circle
):
    """
    Generate coordinates for a semi-circular region with optional ghost nodes.

    Returns:
        coords_phys: Physical points inside the semi-circle
        coords_ghost_left: Ghost nodes on the left (r < 0)
        coords_ghost_bot: Empty (not used)
        coords_ghost_circle: Ghost nodes outside the semi-circular boundary
    """
    coords_ghost_circle = np.empty((0, 2))
    coords_ghost_left = np.empty((0, 2))


    if half_circle:
        # Define full coordinate range
        r_all = np.linspace(-ghost_nodes_r * dr + dr / 2, R + dr / 2 + (ghost_nodes_r - 1) * dr, Nr + 2 * ghost_nodes_r)
        z_all = np.linspace(-ghost_nodes_r * dr + dr / 2, 2 * R + dr / 2 + (ghost_nodes_r - 1) * dr, 2 * Nr + 2 * ghost_nodes_r)

        # Generate full mesh grid
        rr, zz = np.meshgrid(r_all, z_all, indexing='xy')
        coords_all = np.column_stack([rr.ravel(), zz.ravel()])

        # Mask for physical region (half circle centered at (0, R))
        mask_phys = (
            (coords_all[:, 0] >= 0) &
            (coords_all[:, 1] >= 0) &
            (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 <= R ** 2)
        )
        coords_phys = coords_all[mask_phys]

        # Left ghost region
        if r_ghost_left:
            mask_ghost_left = (coords_all[:, 0] < 0) & (coords_all[:, 1] >= 0) & (coords_all[:, 1] <= 2* R)
            coords_ghost_left = coords_all[mask_ghost_left]

        # Circular ghost region outside the physical semi-circle
        mask_circle = (
            (coords_all[:, 0] >= 0) &
            (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 > R ** 2 - 1e-12) &
            (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 <= (R + 3 * dr) ** 2 + 1e-12)
        )
        coords_ghost_circle = coords_all[mask_circle]

    else:
        # Define full coordinate range for rectangular domain
        r_all = np.linspace(-ghost_nodes_r * dr + dr / 2, R - dr / 2, Nr + ghost_nodes_r)+5e-14
        z_all = np.linspace(-ghost_nodes_r * dr + dr / 2, 2 * R - dr / 2, 2 * Nr + ghost_nodes_r)-1.5e-14

        # Generate full mesh grid
        rr, zz = np.meshgrid(r_all, z_all, indexing='xy')
        coords_all = np.column_stack([rr.ravel(), zz.ravel()])

        # Mask for physical rectangular region (0 <= r <= R, 0 <= z <= 2R)
        mask_phys = (
                (coords_all[:, 0] >= 0) &
                (coords_all[:, 1] >= 0) &
                (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 <= R ** 2)
        )
        coords_phys = coords_all[mask_phys]

        # Left ghost region
        if r_ghost_left:
            mask_ghost_left = (coords_all[:, 0] < 0) & (coords_all[:, 1] >= 0)
            coords_ghost_left = coords_all[mask_ghost_left]


    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(coords_phys[:, 0], coords_phys[:, 1], s=8, color='blue', label='Physical')
    plt.scatter(coords_ghost_left[:, 0], coords_ghost_left[:, 1], s=8, color='green', label='Ghost Left')

    if coords_ghost_circle is not None and len(coords_ghost_circle) > 0:
        plt.scatter(coords_ghost_circle[:, 0], coords_ghost_circle[:, 1], s=8, color='red', label='Ghost Circle')

    plt.gca().set_aspect('equal')
    plt.title("Coordinate Categories in Half Circle Region")
    plt.xlabel("r (radius direction)")
    plt.ylabel("z (axial direction)")
    plt.grid(True)

    x_max = np.max([coords_phys[:, 0].max(), coords_ghost_left[:, 0].max()]) + R / 2
    x_min = np.min([coords_phys[:, 0].min(), coords_ghost_left[:, 0].min()]) - 2 * dr
    plt.xlim(x_min, x_max)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return coords_phys, coords_ghost_left, coords_ghost_circle


def generate_circle_coordinates(
    dr, R, Nr, ghost_nodes_r,
    full_circle=False
):
    """
    Generate coordinates for a circular or semi-circular region with optional circular ghost nodes.

    Parameters:
        dr: grid spacing
        R: radius of the circle
        Nr: number of nodes in radius direction
        ghost_nodes_r: number of ghost layers
        full_circle: whether to include ghost nodes around the circular boundary

    Returns:
        coords_phys: Physical points inside the domain (NumPy array of shape [N, 2])
        coords_ghost_circle: Ghost nodes outside the circular boundary (NumPy array of shape [M, 2], only if full_circle=True)
    """
    coords_ghost_circle = np.empty((0, 2))

    # Define full coordinate range; circle is centered at (R, R)
    domain_size = 2 * R
    grid_pts = 2 * Nr + 2 * ghost_nodes_r
    x_all = np.linspace(-ghost_nodes_r * dr + dr / 2, domain_size + ghost_nodes_r * dr - dr / 2, grid_pts)
    y_all = np.linspace(-ghost_nodes_r * dr + dr / 2, domain_size + ghost_nodes_r * dr - dr / 2, grid_pts)

    # Generate mesh grid
    xx, yy = np.meshgrid(x_all, y_all, indexing='xy')
    coords_all = np.column_stack([xx.ravel(), yy.ravel()])

    # Center of the circle
    center = (R, R)

    # Mask for physical region (inside the circle)
    mask_phys = (
        (coords_all[:, 0] - center[0]) ** 2 + (coords_all[:, 1] - center[1]) ** 2 <= R ** 2 + 1e-8
    )
    coords_phys = coords_all[mask_phys]

    # Optional circular ghost region (only when full_circle=True)
    if full_circle:
        mask_ghost_circle = (
            (coords_all[:, 0] - center[0]) ** 2 + (coords_all[:, 1] - center[1]) ** 2 > R ** 2 + 1e-8
        ) & (
            (coords_all[:, 0] - center[0]) ** 2 + (coords_all[:, 1] - center[1]) ** 2 <= (R + 3 * dr) ** 2 + 1e-8
        )
        coords_ghost_circle = coords_all[mask_ghost_circle]

    plt.figure(figsize=(6, 6))
    plt.scatter(coords_phys[:, 0], coords_phys[:, 1], s=8, color='blue',
                label=f'Circle particles ({len(coords_phys)})')
    if full_circle and len(coords_ghost_circle) > 0:
        plt.scatter(coords_ghost_circle[:, 0], coords_ghost_circle[:, 1], s=8, color='red',
                    label=f'Ghost particles ({len(coords_ghost_circle)})')

    plt.gca().set_aspect('equal')
    plt.title("Coordinate Categories in Circle Region")
    plt.xlabel("r (radius direction)")
    plt.ylabel("z (axial direction)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return coords_phys, coords_ghost_circle


def compute_layer_dr_r_nr(r, n_slices, dr1, dr2, dr3, dr_l, len1, len2, len3, size,ghost_node):
    """
    Calculate dr, delta, r, Nr, and length for each region.
    First 3 layers use custom dr and length.
    Remaining layers have integer-micron length, last layer gets extra leftover.
    """
    # Convert to microns for integer calculation
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
        delta = ghost_node * dr
        results.append({
            "layer": i,
            "dr": dr,
            "delta": delta,
            "r": r,
            "Nr": Nr,
            "length": length
        })

    # Optional safety check
    total_height = sum(layer["length"] for layer in results)
    assert abs(total_height - 2 * r) < 1e-12, "Total height does not match 2*r"
    print(f"final n_slices={n_slices}")

    return results,n_slices

from math import fsum
from decimal import Decimal, getcontext, ROUND_HALF_UP
def generate_one_slice_coordinates(
        R, Nr, ghost_nodes_r, zones,
        r_ghost_left, r_ghost_right,
        r_ghost_top, r_ghost_bot,
        n_slices,
        slice_id,
        graph
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
    R = snap_until_clean(R, dr)+1e-14

    # r direction range (including ghost)

    r_all = np.linspace(
        -ghost_nodes_r * dr + dr / 2,
        R + dr / 2 + (ghost_nodes_r - 1) * dr,
        Nr + 2 * ghost_nodes_r
    )

    # Single layer thickness in the z direction
    z_bot = snap_until_clean(2 * R - fsum(zone['length'] for zone in zones[:slice_id + 1]), dr)-1e-14
    z_top = snap_until_clean(z_bot + zones[slice_id]['length'], dr)-1e-14
    Nz = int((dz_layer + 1e-12) / dr)

    z_all = np.linspace(
        z_bot - ghost_nodes_r * dr + dr / 2,
        z_top + dr / 2 + (ghost_nodes_r - 1) * dr ,
        Nz + 2 * ghost_nodes_r
    )

    # Full coordinate grid
    rr, zz = np.meshgrid(r_all, z_all, indexing='xy')
    coords_all = np.column_stack([rr.ravel(), zz.ravel()])

    # Main area mask
    mask_phys = (
            (coords_all[:, 0] >= 0) &
            (coords_all[:, 1] >= z_bot) & (coords_all[:, 1] <= z_top ) &
            (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 <= R ** 2 )
    )
    coords_phys = coords_all[mask_phys]

    # Left ghost
    if r_ghost_left:
        mask_left = (coords_all[:, 0] < 0) & (coords_all[:, 1] >= z_bot - 3 *dr ) & (coords_all[:, 1] <= z_top + 3 *dr)
        ghost_dict['left'] = coords_all[mask_left]


    if r_ghost_bot and slice_id != n_slices - 1:

        mask_bot = (
                (coords_all[:, 1] < z_bot) &
                (coords_all[:, 0] > 0) &
                (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 <= R ** 2)
        )
        ghost_dict['bot'] = coords_all[mask_bot]

    # top ghost(Generation is only allowed when it is not the topmost slice.)
    if r_ghost_top and slice_id != 0:
        mask_top = (
                (coords_all[:, 1] > z_top) &
                (coords_all[:, 0] > 0) &
                (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 <= R ** 2 )
        )
        ghost_dict['top'] = coords_all[mask_top]

    # Right side ghost (note to widen the z range)
    if r_ghost_right:
        if slice_id == 0:
            # Top layer, z range extends upward
            mask_right = (
                    (coords_all[:, 0] >= 0) &
                    (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 > R ** 2 ) &
                    (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 < (R + 3 * dr) ** 2 )
            )
        elif slice_id == n_slices - 1:
            # Bottom layer, z range extends downward
            mask_right = (
                    (coords_all[:, 0] >= 0) &
                    (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 > R ** 2 ) &
                    (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 < (R + 3 * dr) ** 2 )
            )
        else:
            # Middle layer, normal range
            mask_right = (
                    (coords_all[:, 0] > 0) &
                    (coords_all[:, 1] > z_bot - 3 * dr) &
                    (coords_all[:, 1] < z_top + 3 * dr) &
                    (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 > R ** 2 ) &
                    (coords_all[:, 0] ** 2 + (coords_all[:, 1] - R) ** 2 < (R + 3 * dr) ** 2 )
            )
        ghost_dict['right'] = coords_all[mask_right]

    # Top ghost

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
            'circle': 'red'
        }
        for key, coords in ghost_dict.items():
            if len(coords) > 0:
                plt.scatter(coords[:, 0], coords[:, 1], s=8, label=f'Ghost {key}', color=color_map[key])

        # Total number of particles

        total_points = len(coords_phys) + sum(len(coords) for coords in ghost_dict.values())

        plt.gca().set_aspect('equal')
        plt.xlabel('r')
        plt.ylabel('z')
        plt.title(f'Slice {slice_id}, Z ∈ ({z_bot:.2e}, {z_top:.2e})')
        # Set the legend and add total information
        plt.legend(
            title=f'Total particles: {total_points}',
            loc='center left',  # 图例锚点在绘图区的左中
            bbox_to_anchor=(1.05, 0.5)  # 往右移 1.05 倍轴宽，垂直居中
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


