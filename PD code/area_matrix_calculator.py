import numpy as np


def get_cell_corners(x_center, z_center, dx, dz):
    """
    给定单元中心 (x_center, z_center) 与网格尺寸 (dx, dz),
    返回该单元的角点(2D)或端点(1D)的坐标列表 corners。
    """
    if dz == 0:
        # 1D 情况，只有左右两个点
        x_left = x_center - 0.5 * dx
        x_right = x_center + 0.5 * dx
        corners = [
            (x_left, z_center),
            (x_right, z_center),
        ]
    else:
        # 2D 情况, 上下左右四个角点
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


def is_all_in_out(corners, cx, cz, delta,tolerance):
    """
    判断 corners 里的所有角点相对于圆心 (cx, cz) 半径 delta 的位置。
    返回:
      "all_in"   -> 角点都在圆内
      "all_out"  -> 角点都在圆外
      "partial"  -> 部分在内, 部分在外
    """
    in_flags = []
    for (x_pt, z_pt) in corners:
        dist2 = (x_pt - cx) ** 2 + (z_pt - cz) ** 2
        in_flags.append(dist2 <= delta ** 2 + tolerance )

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
    对"部分交叠"的网格单元, 用子网格采样方式估计交叠面积。
    - dz=0 => 1D 线段采样
    - dz!=0 => 2D 网格采样
    sub: 每维采样细分次数
    """
    sub = 10
    if dz == 0:
        # 1D 线段
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
        # 2D 情况
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
        distance_matrix,tolerance
):
    """
    根据"四角点"判断每个网格单元 与 圆心 (cx,cz) 半径 delta 的交叠面积。
    distance_matrix[i,j] 表示第 i 个圆心 与 第 j 个单元中心 的中心距离(仅供快速剔除).

    对于每个 (i, j):
      1) 若 dist>delta + 1e-6 => 完全在外
      2) 否则:
         - 用 get_cell_corners() 得到 j 单元的角点
         - 用 is_all_in_out() 判断 "all_in/all_out/partial"
         - 若 all_in => dx*dz (或 dx, 1D)
         - 若 all_out => 0
         - 若 partial => partial_area_of_cell_in_circle_sampling()
    """
    N = len(x_flat)
    area_mat = np.zeros((N, N), dtype=float)

    for i in range(N):
        cx = x_flat[i]
        if dz == 0:
            cz = z_flat[0]  # 1D: z不变
        else:
            cz = z_flat[i]

        for j in range(N):
            dist = distance_matrix[i, j]
            if dist > delta + 0.5* np.sqrt(dx**2+ dz**2)+ tolerance:
                # 快速判定在外
                area_mat[i, j] = 0.0
                continue

            # j 单元的中心
            xj = x_flat[j]
            zj = z_flat[j] if dz != 0 else z_flat[0]

            # 获取 j 单元的角点
            corners_j = get_cell_corners(xj, zj, dx, dz)

            # 判断这些角点与 (cx, cz) 的关系
            status = is_all_in_out(corners_j, cx, cz, delta,tolerance)

            if status == "all_in":
                # 全部在 horizon 内
                area_mat[i, j] = (dx if dz == 0 else dx * dz)
            elif status == "all_out":
                # 全部在外
                area_mat[i, j] = 0.0
            else:
                # 部分 => 采样估算
                area_mat[i, j] = partial_area_of_cell_in_circle(
                    xj, zj, dx, dz,
                    cx, cz, delta,
                    tolerance
                )

    return area_mat
