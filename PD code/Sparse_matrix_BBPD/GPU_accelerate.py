import numpy as np
from scipy.spatial import cKDTree

def build_neighbor_csr(
    r_flat: np.ndarray,
    z_flat: np.ndarray,
    delta,
    dx, dz,
    tolerance,
):
    """
    只存储 cutoff 内的相互作用数据（不构造 NxN 距离矩阵），并同时返回：
    1) CSR 连续存储：indptr, indices, dx_r, dx_z, dist
    2) 按行拆开的列表视图：idx_list, dx_r_list, dx_z_list, dist_list

    你的筛选条件 dist <= delta + 0.5*sqrt(dx^2+dz^2) + tol
    等价于 dist <= 2*(delta + tol)，因此 cutoff = 2*(delta + tolerance)
    """
    r_flat = np.asarray(r_flat)
    z_flat = np.asarray(z_flat)

    N = r_flat.size
    coords = np.column_stack((r_flat, z_flat))  # (N, 2)

    cutoff = delta + 0.5 * np.sqrt(dx**2 + dz**2) + tolerance

    tree = cKDTree(coords)
    neigh_lists = tree.query_ball_point(coords, r=cutoff)  # list[list[int]]

    # ---- 第1遍：统计每行邻居数（用于构建 indptr / nnz）----
    counts = np.empty(N, dtype=np.int64)

    for i, js in enumerate(neigh_lists):
        counts[i] = len(js)

    indptr = np.empty(N + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])

    indices = np.empty(nnz, dtype=np.int64)
    dist = np.empty(nnz, dtype=np.float64)

    # ---- 第2遍：填充 indices / dist ----
    k = 0
    for i, js in enumerate(neigh_lists):
        m = counts[i]
        if m == 0:
            continue
        js2 = js
        js_arr = np.asarray(js2, dtype=np.int64)

        indices[k:k + m] = js_arr
        dr = r_flat[js_arr] - r_flat[i]
        dz = z_flat[js_arr] - z_flat[i]
        dist[k:k + m] = np.sqrt(dr * dr + dz * dz)

        k += m

    return indptr, indices, dist
