import generate_coordinates as gc
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import Multiprocess_task_function as mt
import bc_funcs as bc
import core_funcs as cf

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Physical and thermal parameters
rho_s, cs, ks = 1000.0, 2060.0, 1
rho_l, cl, kl = 1000.0, 4182.0, 0.6
Ts = 312.65
Tl = 313.65
L = 333
tolerance = 1e-14
Tsurr = 400
Tinit = 283.15

# Initialization of coarse regions and temperatures
# dr1, dr2, dr3 are used to generate regional particle density by calling functions to create coordinates.
r = 20 * 1e-6  # Domain radius in meters
dr1 = 0.05 * 1e-6
dr2 = 0.05 * 1e-6
dr3 = 2 * dr2
dr_l = 0.4 * 1e-6

len1 = 0.5 * 1e-6
len2 = 0.5 * 1e-6
len3 = 1 * 1e-6

ghost_nodes_r = 3  # Number of ghost cells in the r direction
n_slices = 8
num_processes = 4
dt = 1e-8  # Time step (s)
total_time = 100e-8  # Total simulation time (s)

if __name__ == "__main__":
    start_time2 = time.time()

    # 1. Define regional coordinates and compute area matrices
    zones = gc.compute_layer_dr_r_nr(r, n_slices, dr1, dr2, dr3, dr_l, len1, len2, len3)

    for z in zones:
        print(f"Layer {z['layer']:2d}: dr = {z['dr']:.2e}, delta = {z['delta']:.2e}, "
              f"length = {z['length']:.2e}, Nr = {z['Nr']}")

    phys_coords_list = []
    ghost_coords_list = []
    ghost_dict_list = []
    start_time = time.time()

    for i in range(n_slices):
        zone = zones[i]
        dr_i = zone["dr"]
        Nr_i = zone["Nr"]

        coords_phys, ghost_coords, n_points, ghost_dict = gc.generate_one_slice_coordinates(
            r, Nr_i, ghost_nodes_r,
            zones,
            r_ghost_left=True,
            r_ghost_right=True,
            r_ghost_top=True,
            r_ghost_bot=True,
            n_slices=n_slices,
            slice_id=i,
            graph=False
        )
        print(f"The number of particles in region {i} is: {n_points}")

        phys_labeled = np.hstack([coords_phys, np.full((coords_phys.shape[0], 1), i)])
        ghost_labeled = np.hstack([ghost_coords, np.full((ghost_coords.shape[0], 1), i)])
        phys_coords_list.append(phys_labeled)
        ghost_coords_list.append(ghost_labeled)
        ghost_dict_list.append(ghost_dict)

    task_args = []
    for i in range(n_slices):
        coords = np.vstack([phys_coords_list[i], ghost_coords_list[i]])
        dr = zones[i]["dr"]
        delta = zones[i]["delta"]
        slice_id = zones[i]["layer"]
        task_args.append((coords, dr, delta, tolerance, slice_id))

    if num_processes == 1:
        results = [mt.compute_region_matrices(args) for args in task_args]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(mt.compute_region_matrices, task_args)

    distance_matrices = []
    partial_area_matrices = []
    horizon_masks = []
    true_indices_list = []

    for i, file_path in enumerate(results):
        data = np.load(file_path)
        distance_matrices.append(data["distance"])
        partial_area_matrices.append(data["area"])
        horizon_masks.append(data["mask"])
        true_indices_list.append(tuple(data["indices"]))

    end_time = time.time()
    print(f"Calculation of partial_area_matrices finished, elapsed real time = {end_time - start_time:.2f}s")
