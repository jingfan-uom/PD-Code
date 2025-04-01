import matplotlib.pyplot as plt
import numpy as np

def temperature(R_all, Z_all, T, total_time, nsteps):
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
    plt.title(f"Temperature after {total_time:.1f}s ({nsteps} steps)")
    plt.show()
