# region_utils.py

import area_matrix_calculator  # 确保已导入
import core_funcs as cf
import numpy as np
import bc_funcs as bc

def compute_region_matrices(args):
    coords, dr, delta, tolerance, slice_id = args

    r_flat = coords[:, 0]
    z_flat = coords[:, 1]

    N = len(r_flat)
    distance_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        dx_r = r_flat[i] - r_flat  # shape (N,)
        dx_z = z_flat[i] - z_flat
        distance_matrix[i, :] = np.sqrt(dx_r ** 2 + dx_z ** 2)

    partial_area_matrix = area_matrix_calculator.compute_partial_area_matrix(
        r_flat, z_flat, dr, dr, delta, distance_matrix, tolerance
    )

    horizon_mask = ((distance_matrix > tolerance) & (partial_area_matrix != 0.0))
    true_indices = np.where(horizon_mask)

    # 保存为 npz 文件
    np.savez_compressed(f"matrix_slice_{slice_id}.npz",
                        distance=distance_matrix,
                        area=partial_area_matrix,
                        mask=horizon_mask,
                        indices=true_indices)
    return f"matrix_slice_{slice_id}.npz"

# Multiprocess_task_function.py

def update_temperature_for_region(args):
    (
        region_id,
        Tcurr,
        Hcurr,
        Kmat,
        factor_mat,
        partial_area_matrix,
        shape_factor_matrix,
        distance_matrix,
        horizon_mask,
        true_indices,
        delta,
        dt,
        rho_s, cs, cl, L, Ts, Tl, ks, kl  # 材料参数
    ) = args

    # 1. 非局部通量更新焓
    flux = Kmat @ Tcurr
    Hnew = Hcurr + flux
    # 2. 焓转温度
    Tnew = cf.get_temperature(Hnew, rho_s, cs, cl, L, Ts, Tl)
    # 3. 构建新的导热矩阵
    Knew = cf.build_K_matrix(
        Tnew,
        cf.compute_thermal_conductivity_matrix,
        factor_mat,
        partial_area_matrix,
        shape_factor_matrix,
        distance_matrix,
        horizon_mask,
        true_indices,
        ks, kl, Ts, Tl, delta, dt
    )

    return region_id, Tnew, Hnew, Knew
