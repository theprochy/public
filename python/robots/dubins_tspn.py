# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import math
import time

from invoke_LKH import solve_TSP
import dubins

def dist_euclidean_squared(coord1, coord2):
    (x1, y1) = coord1
    (x2, y2) = coord2
    return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)

def dist_euclidean(coord1, coord2):
    return math.sqrt(dist_euclidean_squared(coord1, coord2))

def configurations_to_path(configurations, turning_radius):  
    """
    Compute a closed tour through the given configurations and turning radius, 
    and return densely sampled configurations and length.  

    Parameters
    ----------
    configurations: list (float, float, float)
        list of robot configurations (x,y,phi), one for each goal
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    list (float,float,float), path_length (float)
        tour as a list of densely sampled robot configurations (x, y, phi)
    """  
    N = len(configurations)
    path = []
    path_len = 0.
    for a in range(N):
        b = (a+1) % N
        start = configurations[a]
        end = configurations[b]
        step_size = 0.01 * turning_radius
        dubins_path = dubins.shortest_path(start, end, turning_radius)
        segment_len = dubins_path.path_length()
        step_configurations, _ = dubins_path.sample_many(step_size)
        path = path + step_configurations
        path_len += segment_len
    return path, path_len

def create_samples(goals, sensing_radius, position_resolution, heading_resolution):
    """
    Sample the goal regions on the boundary using uniform distribution.

    Parameters
    ----------
    goals: list (float, float)
        list of the TSP goal coordinates (x, y)
    sensing_radius: float
        neighborhood of TSP goals  
    position_resolution: int
        number of location at the region's boundary
    heading_resolution: int
        number of heading angles per location
    
    Returns
    -------
    matrix[target_idx][sample_idx] 
        2D matrix of configurations (x, y, phi)
    """ 
    samples = []
    for idx, g in enumerate(goals):
        samples.append([])
        for sp in range(position_resolution):
            alpha = sp * 2*math.pi / position_resolution
            position = g + sensing_radius * np.array([math.cos(alpha), math.sin(alpha)])
            for sh in range(heading_resolution):
                heading = sh * 2*math.pi / heading_resolution
                sample = (position[0], position[1], heading)
                samples[idx].append(sample)
    return samples

def plan_tour_decoupled(goals, sensing_radius, turning_radius):
    """
    Compute a DTSPN tour using the decoupled approach.  

    Parameters
    ----------
    goals: list (float, float)
        list of the TSP goal coordinates (x, y)
    sensing_radius: float
        neighborhood of TSP goals  
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    list (float,float,float), path_length (float)
        tour as a list of robot configurations (x, y, phi) densely sampled
    """

    N = len(goals)

    # find path between each pair of goals (a,b)
    etsp_distances = np.zeros((N,N))	
    for a in range(0,N): 
        for b in range(0,N):
            g1 = goals[a]
            g2 = goals[b]
            etsp_distances[a][b] = dist_euclidean(g1, g2) 
        
    # Example how to print a small matrix with fixed precision
    # np.set_printoptions(precision=2)
    # print("ETSP distances")
    # print(etsp_distances)

    sequence = solve_TSP(etsp_distances)
    # print("ETSP sequence")
    # print(sequence)
    position_resolution = heading_resolution = 8
    samples = create_samples(goals, sensing_radius, position_resolution, heading_resolution)
    n_conf = len(samples[sequence[0]])
    #      starting conf | step | step end conf
    M = np.full((n_conf, N, n_conf), np.inf)
    M[:, 0, :] = 0
    prev = np.full((n_conf, N + 1, n_conf), np.nan)
    lengths = np.full(n_conf, np.inf)
    # starting config
    for i in range(n_conf):
        # step
        for j in range(1, N + 1):
            # each start config
            for k in ([i] if j == 1 else range(n_conf)):
                # each goal config
                start = samples[sequence[j - 1]][k]
                for l in ([i] if j == N else range(n_conf)):
                    end = samples[sequence[j % N]][l]
                    dubins_path = dubins.shortest_path(start, end, turning_radius)
                    if j == N:
                        if M[i][j - 1][k] + dubins_path.path_length() < lengths[i]:
                            lengths[i] = M[i][j - 1][k] + dubins_path.path_length()
                            prev[i][j][l] = k
                    elif M[i][j - 1][k] + dubins_path.path_length() < M[i][j][l]:
                        M[i][j][l] = M[i][j - 1][k] + dubins_path.path_length()
                        prev[i][j][l] = k
    best_conf = np.where(lengths == min(lengths))[0][0]
    selected_samples = []
    selected = best_conf
    for i in range(N, 0, -1):
        selected = int(prev[best_conf][i][selected])
        selected_samples.append(selected)
    selected_samples.reverse()

    configurations = []
    for idx in range(N):
        configurations.append(samples[sequence[idx]][selected_samples[idx]])

    return configurations_to_path(configurations, turning_radius)

def plan_tour_noon_bean(goals, sensing_radius, turning_radius):
    """
    Compute a DTSPN tour using the NoonBean approach.  

    Parameters
    ----------
    goals: list (float, float)
        list of the TSP goal coordinates (x, y)
    sensing_radius: float
        neighborhood of TSP goals  
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    list (float,float,float), path_length (float)
        tour as a list of robot configurations (x, y, phi) densely sampled
    """

    N = len(goals)
    position_resolution = heading_resolution = 8
    samples = create_samples(goals, sensing_radius, position_resolution, heading_resolution)

    # Number of samples per location
    n_conf = position_resolution * heading_resolution
    distances = np.zeros((N*n_conf, N*n_conf))
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(n_conf):
                    for l in range(n_conf):
                        start = samples[i][k]
                        end = samples[j][l]
                        dubins_path = dubins.shortest_path(start, end, turning_radius)
                        distances[i * n_conf + k][j * n_conf + (l + 1) % n_conf] = dubins_path.path_length()
    M = 1.2 * np.max(distances)
    INF = N * M / 2
    distances_f = np.full((N*n_conf, N*n_conf), INF, dtype=float)
    for i in range(N):
        for j in range(N):
            # zero cycle block
            if i == j:
                for k in range(n_conf):
                    distances_f[i * n_conf + k][i * n_conf + (k + 1) % n_conf] = 0
            # distances to next set block
            else:
                for k in range(n_conf):
                    for l in range(n_conf):
                        distances_f[i * n_conf + k][j * n_conf + (l + 1) % n_conf] = distances[i * n_conf + k][j * n_conf + (l + 1) % n_conf] + M
    start = time.time()
    sequence = solve_TSP(distances_f)
    print(time.time() - start)
    final_sequence = []
    selected_samples = []
    tsp_cost = 0
    for i in range(N * n_conf):
        tsp_cost = tsp_cost + distances_f[sequence[i]][sequence[(i + 1) % (N * n_conf)]]
        if sequence[i] // n_conf != sequence[(i + 1) % (N * n_conf)] // n_conf:
            final_sequence.append(sequence[i] // n_conf)
            selected_samples.append(sequence[i] % n_conf)
    configurations = []
    for idx in range(N):
        configurations.append(samples[final_sequence[idx]][selected_samples[idx]])

    return configurations_to_path(configurations, turning_radius)
