import numpy as np

def generate_fine_and_coarse_grids(
        r_start, z_start,
        Lr, Lz,
        dr_coarse, dz_coarse,
        dx_fine, dz_fine,
        fine_mask_func,
        delta
):

    # 1) 分别生成 fine 网格 / coarse 网格 (1D)
    r_all_fine = np.arange(r_start + dx_fine / 2, r_start + Lr, dx_fine)
    z_all_fine = np.arange(z_start + dz_fine / 2, z_start + Lz, dz_fine)

    r_all_coarse = np.arange(r_start + dr_coarse / 2, r_start + Lr, dr_coarse)
    z_all_coarse = np.arange(z_start + dz_coarse / 2, z_start + Lz, dz_coarse)

    # 2) 分别创建 2D 网格
    rr_fine, zz_fine = np.meshgrid(r_all_fine, z_all_fine, indexing='xy')
    rr_coarse, zz_coarse = np.meshgrid(r_all_coarse, z_all_coarse, indexing='xy')
    rr_fine = np.round(rr_fine, decimals=12)
    zz_fine = np.round(zz_fine, decimals=12)

    rr_coarse = np.round(rr_coarse, decimals=12)
    zz_coarse = np.round(zz_coarse, decimals=12)

    # 3) 细网格物理点(phys_mask)
    fine_mask_phys = np.zeros_like(rr_fine, dtype=bool)
    for i in range(rr_fine.shape[0]):
        for j in range(rr_fine.shape[1]):
            if fine_mask_func(rr_fine[i, j], zz_fine[i, j]):
                fine_mask_phys[i, j] = True
    fine_mask_non_phys = ~fine_mask_phys

    # 4) 粗网格物理点(可定义与fine互斥,这里只做简单示例)
    coarse_mask_phys = np.zeros_like(rr_coarse, dtype=bool)
    for i in range(rr_coarse.shape[0]):
        for j in range(rr_coarse.shape[1]):
            if not fine_mask_func(rr_coarse[i, j], zz_coarse[i, j]):
                coarse_mask_phys[i, j] = True
    coarse_mask_non_phys = ~coarse_mask_phys

    # 5) 计算界面掩码
    fine_mask_interface = compute_interface_mask(fine_mask_phys, fine_mask_non_phys)
    coarse_mask_interface = compute_interface_mask(coarse_mask_phys, coarse_mask_non_phys)

    # 6) 计算“界面边界”掩码 (上下左右扩展delta)
    fine_mask_interface_boundary = mark_interface_boundary_points_2d(
        mask_interface=fine_mask_interface,
        mask_non_phys=fine_mask_non_phys,
        delta=delta
    )
    coarse_mask_interface_boundary = mark_interface_boundary_points_2d(
        mask_interface=coarse_mask_interface,
        mask_non_phys=coarse_mask_non_phys,
        delta=delta
    )

    # 7) 汇总结果并返回
    fine_phys_coords = np.column_stack((rr_fine[fine_mask_phys],zz_fine[fine_mask_phys]))
    coarse_phys_coords = np.column_stack((rr_coarse[coarse_mask_phys],zz_coarse[coarse_mask_phys]))

    fine_intf_coords = np.column_stack((rr_fine[fine_mask_interface_boundary],zz_fine[fine_mask_interface_boundary]))
    coarse_intf_coords = np.column_stack((rr_coarse[coarse_mask_interface_boundary],zz_coarse[coarse_mask_interface_boundary]))

    return {
        # 掩码
        "fine_mask_phys": fine_mask_phys,
        "coarse_mask_phys": coarse_mask_phys,
        "fine_mask_intf_boundary": fine_mask_interface_boundary,
        "coarse_mask_intf_boundary": coarse_mask_interface_boundary,

        # 对应坐标矩阵
        "fine_phys_coords": fine_phys_coords,
        "coarse_phys_coords": coarse_phys_coords,
        "fine_intf_coords": fine_intf_coords,
        "coarse_intf_coords": coarse_intf_coords,}

def compute_interface_mask(phys_mask, non_phys_mask):
    """
    给定 phys_mask 和 non_phys_mask（同一网格下的掩码），
    找到邻居有非物理点的物理点，标记为 True。
    """
    assert phys_mask.shape == non_phys_mask.shape, "两个掩码尺寸不一致！"
    nrows, ncols = phys_mask.shape
    interface_mask = np.zeros_like(phys_mask, dtype=bool)

    def neighbors(i, j):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < nrows and 0 <= nj < ncols:
                yield ni, nj

    for i in range(nrows):
        for j in range(ncols):
            if phys_mask[i, j]:
                # 若相邻点中有非物理点，就标记该点为 interface
                for (ni, nj) in neighbors(i, j):
                    if non_phys_mask[ni, nj]:
                        interface_mask[i, j] = True
                        break
    return interface_mask


def mark_interface_boundary_points_2d(mask_interface, mask_non_phys, delta=0):
    """
    在上下左右各扩展 delta 的邻域内，
    如果非物理点距离(行列差小于等于delta)任意界面点不超过 delta，则标记为 True。
    """
    assert mask_interface.shape == mask_non_phys.shape, "两个掩码尺寸必须一致"
    nrows, ncols = mask_interface.shape
    boundary_mask = np.zeros_like(mask_non_phys, dtype=bool)

    for i in range(nrows):
        for j in range(ncols):
            if mask_non_phys[i, j]:
                # 在 [i-delta, i+delta], [j-delta, j+delta] 范围内搜索
                rmin, rmax = max(0, i - delta), min(nrows - 1, i + delta)
                cmin, cmax = max(0, j - delta), min(ncols - 1, j + delta)

                found_interface = False
                for rr in range(rmin, rmax + 1):
                    for cc in range(cmin, cmax + 1):
                        if mask_interface[rr, cc]:
                            found_interface = True
                            break
                    if found_interface:
                        break

                if found_interface:
                    boundary_mask[i, j] = True
    return boundary_mask

import numpy as np

def interpolate_coarse_interface_temperature(
    coarse_intf_coords: np.ndarray,
    fine_phys_coords:  np.ndarray,
    fine_phys_temps:   np.ndarray,
    dr_fine: float,
    dz_fine: float,
    tol: float = 1e-12,
) -> np.ndarray:

    # —— 把细网格坐标 -> 温度 做成快速字典 —— #
    # 为避免浮点误差，坐标先按 tol 做 round
    def key(p):
        return (round(p[0] / tol) , round(p[1] / tol))

    coord2temp = { key(p): t for p, t in zip(fine_phys_coords, fine_phys_temps) }

    # —— 对每个 coarse 接口点插值 —— #
    result = np.empty(coarse_intf_coords.shape[0])

    for i, (r, z) in enumerate(coarse_intf_coords):
        # 目标 4 个邻点坐标
        neighbors = [
            (r - dr_fine/2, z - dz_fine/2),  # 左上
            (r - dr_fine/2, z + dz_fine/2),  # 左下
            (r + dr_fine/2, z - dz_fine/2),  # 右上
            (r + dr_fine/2, z + dz_fine/2),  # 右下
        ]

        temps = []
        for p in neighbors:
            k = key(p)
            if k in coord2temp:
                temps.append(coord2temp[k])
            else:
                raise ValueError(
                    f"找不到细网格物理点 {p}；检查 dr_fine/dz_fine 是否正确 "
                    "或该接口粒子是否落在细网格之外。"
                )

        # 四点均值
        result[i] = np.mean(temps)

    return result
import numpy as np

def interpolate_fine_interface_temperature(
    fine_intf_coords:  np.ndarray,
    coarse_phys_coords: np.ndarray,
    coarse_phys_temps:  np.ndarray,
    dr_coarse: float,
    dz_coarse: float,
    power: float = 1.0,
    tol: float = 1e-12
) -> np.ndarray:

    R = np.sqrt(dr_coarse**2 + dz_coarse**2) + tol
    Nf = fine_intf_coords.shape[0]
    temps_out = np.empty(Nf)

    for i, Pf in enumerate(fine_intf_coords):
        # 与所有粗点的欧氏距离
        diffs = coarse_phys_coords - Pf
        dists = np.hypot(diffs[:, 0], diffs[:, 1])

        # 选出半径 R 内的邻居
        mask = dists <= R
        if not np.any(mask):
            raise ValueError(
                f"Fine‑intf 点 {Pf} 周围半径 {R:.4g} 内无 coarse‑phys 点，"
                "请检查网格覆盖范围或增大半径。"
            )

        d_sel = dists[mask]
        T_sel = coarse_phys_temps[mask]

        # 若有重合点（距离≈0），直接取该温度
        if np.any(d_sel < tol):
            temps_out[i] = T_sel[d_sel < tol][0]
            continue

        # 反距离加权
        w = 1.0 / np.power(d_sel, power)
        temps_out[i] = np.dot(w, T_sel) / w.sum()

    return temps_out
