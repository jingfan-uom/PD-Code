import matplotlib.pyplot as plt
import numpy as np

def temperature(R_all, Z_all, T, total_time, nsteps,dt):
    T_min = np.min(T)
    T_max = np.max(T)

    plt.figure(figsize=(6, 5))
    levels = np.linspace(340, 500, 17)
    ctf = plt.contourf(R_all, Z_all, T, levels=levels, cmap='jet')
    plt.xlim([0.2, 1.0])
    plt.ylim([0.0, 0.8])
    cbar = plt.colorbar(ctf)
    cbar.set_label(f"Temperature (K)\nMin: {T_min:.2f} K, Max: {T_max:.2f} K")
    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature after {total_time-dt:.1f}s ({nsteps} steps)")
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def plot_z_profile(T_record, z_all, r_all, save_times_hours):
    z_target = 0.4
    z_index = np.argmin(np.abs(z_all - z_target))

    plt.figure(figsize=(8, 5))
    for T, t_hour in zip(T_record, save_times_hours):
        plt.plot(r_all, T[z_index, :], label=f"{t_hour} h")

    plt.xlabel("r (m)")
    plt.ylabel("Temperature at z = 0.4 m (K)")
    plt.title("z = 0.4 m Cross-sectional Temperature Evolution")
    plt.xlim(0.2, 1.0)
    plt.ylim(200, 500)

    # 设置主刻度：每 50K
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    # 设置次刻度：每 25K（即在主刻度之间再加一个）
    plt.gca().yaxis.set_minor_locator(MultipleLocator(25))
    # 开启次刻度显示
    plt.tick_params(axis='y', which='minor', length=4, color='gray')
    plt.tick_params(axis='y', which='major', length=7)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

