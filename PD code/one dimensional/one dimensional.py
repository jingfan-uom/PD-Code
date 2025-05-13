import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Material properties (Based on Case 5.2 in the paper)
# ==============================
Kl = 0.2  # Thermal conductivity in liquid state (W/m-K)
Ks = 0.2  # Thermal conductivity in solid state (W/m-K)
Cs = 2000  # Specific heat capacity in solid state (J/kg-K)
Cl = 2000  # Specific heat capacity in liquid state (J/kg-K)
rho = 780  # PCM density (kg/m^3)
L = 168000  # PCM latent heat (J/kg)

# Phase change temperature range (46.5 - 48.5°C)
T_low = 46.5  # Solid phase temperature (°C)
T_high = 48.5  # Liquid phase temperature (°C)
T_left = 65.0  # Left boundary temperature (°C)
T_right = 25.0  # Right boundary temperature (°C)

# ==============================
# Spatial and time parameters
# ==============================
L_dom = 0.14  # Computational domain length (m)
N_particles = 112  # 56 particles used in the paper
dx = L_dom / N_particles  # **Note: No longer using N_particles - 1**
dt = 1  # Time step size (s)
delta = 4.5 * dx  # Horizon radius


ghost_nodes = 3  # ghost_nodes are the boundary nodes

# **Modification**: Center the physical grid points within each cell
x_phys = np.linspace(0.06 + dx / 2, 0.20 - dx / 2, N_particles)  # Physical grid points

x_ghost_left = np.arange(0.06 - (ghost_nodes) * dx + dx / 2, 0.06, dx)  # **Left ghost nodes**
x_ghost_right = np.arange(0.20 + dx / 2, 0.20 + dx / 2 + (ghost_nodes - 1) * dx, dx)  # **Right ghost nodes**
# Redefine the entire x-axis, including ghost nodes
x = np.concatenate([x_ghost_left, x_phys, x_ghost_right])

# ==============================
# Initialize temperature field
# ==============================
T = np.full_like(x, 25.0)  # Initial temperature set to 25°C

# 第二步：计算距离矩阵
distance_matrix = np.abs(x[:, None] - x[None, :])

# 第三步: 根据距离筛选有效节点（超过delta的都为0）这是布尔矩阵
horizon_mask = distance_matrix <= delta

# 第四步: 轴对称修正系数矩阵
axisymmetric_factor = 2 * x[None, :] / (x[:, None] + x[None, :])



print("距离矩阵 distance_matrix：\n", distance_matrix)
print("\n视域掩码 horizon_mask：\n", horizon_mask)
print("\n轴对称修正因子 axisymmetric_factor：\n", axisymmetric_factor)
print("\nx坐标矩阵 x_matrix：\n",x )


# ==============================
# Compute initial enthalpy H
# ==============================
def enthalpy(T):
    """ Compute enthalpy H (based on phase transition region handling) """
    H = np.copy(T)
    for i in range(len(T)):
        if T[i] < T_low:  # Solid phase
            H[i] = Cs * T[i]
        elif T[i] > T_high:  # Liquid phase
            H[i] = Cs * T_low + L + Cl * (T[i] - T_high)
        else:  # Phase transition region (linear interpolation)
            H[i] = Cs * T_low + (L / (T_high - T_low)) * (T[i] - T_low)
    return H


H = enthalpy(T)  # Compute initial enthalpy


# ==============================
# Convert enthalpy H to temperature T
# ==============================
def temperature(H):
    """ Convert enthalpy to temperature """
    T_new = np.copy(H)
    for i in range(len(H)):
        if H[i] <= Cs * T_low:  # Solid state
            T_new[i] = H[i] / Cs
        elif H[i] >= Cs * T_low + L:  # Liquid state
            T_new[i] = (H[i] - (Cs * T_low + L)) / Cl + T_high
        else:  # Phase transition region
            T_new[i] = ((H[i] - Cs * T_low) * (T_high - T_low) / L) + T_low
    return T_new


# ==============================
# Compute thermal conductivity K(T)
# ==============================
def thermal_conductivity(T):
    """ Compute thermal conductivity based on temperature """
    return Ks if T < T_low else Kl


# ==============================
# Update temperature and enthalpy

def apply_dirichlet_bc_mirror(T, ghost_nodes):
    """
    Apply Dirichlet boundary conditions using the mirror method (Ghost Cell approach).
    T_left and T_right represent the fixed boundary temperatures.
    """
    # Left ghost nodes
    # ghost_nodes are the boundary nodes
    for i in range(ghost_nodes):
        # interior_index corresponds to the "mirrored" interior node for the i-th ghost node
        # When ghost_nodes=3:
        #   i=0 => interior_index = 2*(3) - 0 - 1 = 5 => (counting i+1 nodes inward)
        #   i=1 => interior_index = 4
        #   i=2 => interior_index = 3
        interior_index = 2 * ghost_nodes - i - 1

        # Formula: T_ghost = 2*T_boundary - T_interior
        T[i] = 2.0 * T_left - T[interior_index]

    # Right ghost nodes
    n = len(T)
    for i in range(n - ghost_nodes, n):
        # dist represents the offset relative to the right-side start index (n - ghost_nodes)
        dist = i - (n - ghost_nodes)
        # interior_index is the corresponding mirrored interior node
        #   e.g., if i = n-1 (last ghost node) => dist = ghost_nodes-1 => interior_index moves dist+1 left
        interior_index = (n - ghost_nodes - 1) - dist

        T[i] = 2.0 * T_right - T[interior_index]

    return T
""" 这里初始化温度矩阵"""
T = apply_dirichlet_bc_mirror(T, ghost_nodes)

""" 下面这个函数的思路是首先计算两点间的平均导热系数K_avg_matrix，然后乘以轴对称系数和时间步等，避免计算除0，再对自热乘以了
1.5得到系数矩阵K"""
def build_K_matrix(T_array, ghost_nodes):
    N = len(T_array)
    # 第一步: 导热系数矩阵
    K_values = np.array([thermal_conductivity(T) for T in T_array])
    K_avg_matrix = (K_values[:, None] + K_values[None, :]) / 2 /delta

    # 第二步：K矩阵 用于计算焓
    """ 这里用轴对称系数矩阵和距离矩阵 还有horizon矩阵"""
    # 先初始化 K_matrix 全为 0
    K_matrix = np.zeros((N, N))

    # 只对 valid 位置 (distance!=0 且在 horizon 内) 进行运算
    valid = (distance_matrix != 0) & horizon_mask
    # 这里也可以先把分子、分母拆出来，看起来更清晰：
    numerator = K_avg_matrix * axisymmetric_factor * dt / rho
    denominator = (distance_matrix ** 2)/ dx
    # 只对 valid 元素执行除法并赋值到 K_matrix
    K_matrix[valid] = numerator[valid] / denominator[valid]

    # 第三步：相邻节点乘以1.5 (最近邻节点)
    for i in range(ghost_nodes, N - ghost_nodes):

        K_matrix[i, i - 1] *= 1.5
        K_matrix[i, i + 1] *= 1.5
    return K_matrix

# ==============================
def update_temperature(T, H):
    """ Update temperature field """
    new_T = np.copy(T)
    new_H = np.copy(H)
    new_K=  build_K_matrix(new_T, ghost_nodes)
    # 温度差矩阵 (new_T[j]-new_T[i])
    # ----------------------------------------------------------
    # 1) 计算温度差矩阵 (new_T[j] - new_T[i])，维度 NxN
    #    注意顺序是 j - i
    delta_T_matrix = new_T[None, :] - new_T[:, None]

    # 2) 用horizon_mask屏蔽不在视域内的节点，然后进行相乘
    #    adj_matrix[i, j] 表示节点 i 与 j 的有效（视域内）传热贡献
    adj_matrix = new_K * delta_T_matrix

    """3) 沿着行方向(axis=1)求和，得到每个节点 i 的净热通量(或焓增量),这里的求和得整明白了，注意矩阵方向，我用的是K*T=H，所以是对行求和"""
    #    flux[i] = sum_j( adj_matrix[i, j] )
    flux = np.sum(adj_matrix, axis=1)

    # 4) 将这个一维向量加到new_H上，得到更新后的焓值
    #    注意 new_H 本身是一维的 shape=(N,)
    new_H = new_H + flux

    new_T = temperature(new_H)  # Convert enthalpy to temperature
    new_T = apply_dirichlet_bc_mirror(new_T, ghost_nodes) # Maintain boundary conditions


    return new_T, new_H


import time
start_time = time.time()  # 记录开始时间

# ==============================
# Time-stepping loop
# ==============================
time_steps = int(21601 / dt)

# Store temperature distributions at 1h (3600s), 3h (10800s), 6h (21600s)
time_points = {
    3600: None,
    10800: None,
    21600: None
}

for step in range(time_steps):
    T, H = update_temperature(T, H)

    current_time = step * dt  # Current time (seconds)

    # Save data if the current time matches one of the target time points
    if int(current_time) in time_points:
        time_points[int(current_time)] = np.copy(T)

    # Print debug information
    if step % 3601 == 0:
        print(f"t = {current_time / 3600:.2f} h, Temperature (first 10 points):", T[:10])

# ==============================
# Extract temperature at 1h, 3h, and 6h
# ==============================
T_1h = time_points[3600]
T_3h = time_points[10800]
T_6h = time_points[21600]

# ==============================
# Plot temperature profiles at 1h, 3h, 6h
# ==============================
plt.plot(x_phys, T_1h[ghost_nodes: -ghost_nodes], label="1 Hour")
plt.plot(x_phys, T_3h[ghost_nodes: -ghost_nodes], label="3 Hours")
plt.plot(x_phys, T_6h[ghost_nodes: -ghost_nodes], label="6 Hours")

plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Profile at 1h, 3h, 6h')
plt.legend()
plt.grid(True)
plt.show()



end_time = time.time()  # 记录结束时间
elapsed = end_time - start_time  # 计算耗时
print(f"计算结束，共耗时 {elapsed:.2f} 秒")
