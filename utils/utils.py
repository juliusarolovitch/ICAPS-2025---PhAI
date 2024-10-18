# utils.py

import numpy as np
import torch
import random
import heapq

class Node:
    def __init__(self, pos, g=float('inf'), h=0, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def is_valid(pos, map_grid):
    return 0 <= pos[0] < map_grid.shape[0] and 0 <= pos[1] < map_grid.shape[1] and map_grid[pos] == 0

def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(pos, map_grid):
    x, y = pos
    neighbors = [
        ((x-1, y), 1),
        ((x+1, y), 1),
        ((x, y-1), 1),
        ((x, y+1), 1),
        ((x-1, y-1), np.sqrt(2)),
        ((x-1, y+1), np.sqrt(2)),
        ((x+1, y-1), np.sqrt(2)),
        ((x+1, y+1), np.sqrt(2))
    ]
    return [(neighbor, cost) for neighbor, cost in neighbors if is_valid(neighbor, map_grid)]

def generate_start_goal_biased(map_grid, bias_factor=0.05):
    size = map_grid.shape[0]
    all_positions = [(x, y) for x in range(size)
                     for y in range(size) if map_grid[x, y] == 0]

    if not all_positions:
        raise ValueError("No valid positions found in the map")

    distances = np.array([distance_to_nearest_obstacle(
        pos, map_grid) for pos in all_positions])
    probabilities = np.exp(-bias_factor * distances)
    probabilities /= np.sum(probabilities)

    while True:
        start = all_positions[np.random.choice(
            len(all_positions), p=probabilities)]
        goal = all_positions[np.random.choice(
            len(all_positions), p=probabilities)]

        if start != goal:
            return start, goal

def distance_to_nearest_obstacle(pos, map_grid):
    obstacle_positions = np.argwhere(map_grid == 1)
    if obstacle_positions.size == 0:
        return np.inf  # No obstacles in the map
    distances = np.sqrt(np.sum((obstacle_positions - pos)**2, axis=1))
    return np.min(distances)

def generate_map(height=128, width=128, num_obstacles=150, min_obstacle_size=4, max_obstacle_size=8):
    map_grid = np.zeros((height, width), dtype=int)
    for _ in range(num_obstacles):
        obstacle_size = random.randint(min_obstacle_size, max_obstacle_size)
        x = random.randint(0, height - obstacle_size)
        y = random.randint(0, width - obstacle_size)
        map_grid[x:x+obstacle_size, y:y+obstacle_size] = 1
    return map_grid
