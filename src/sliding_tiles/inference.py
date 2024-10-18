import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
import matplotlib.pyplot as plt
import heapq
import os
import csv
from tqdm import tqdm
import math
import random
import time
from itertools import count
import bisect

# import imageio  # Commented out as visualization is optional

# ---------------------------
# Utility Functions for Sliding Tile Puzzle
# ---------------------------

def get_possible_moves(blank_pos, grid_size=5):
    moves = []
    row = blank_pos // grid_size
    col = blank_pos % grid_size
    if row > 0:  # Up
        moves.append(blank_pos - grid_size)
    if row < grid_size - 1:  # Down
        moves.append(blank_pos + grid_size)
    if col > 0:  # Left
        moves.append(blank_pos - 1)
    if col < grid_size - 1:  # Right
        moves.append(blank_pos + 1)
    return moves

def get_neighbors(state, grid_size=5):
    neighbors = []
    blank_pos = state.index(0)
    possible_moves = get_possible_moves(blank_pos, grid_size)
    for move in possible_moves:
        new_state = list(state)
        new_state[blank_pos], new_state[move] = new_state[move], new_state[blank_pos]
        neighbors.append((tuple(new_state), 1))  # Cost is 1 per move
    return neighbors

def manhattan_distance(state, goal_state, grid_size=5):
    distance = 0
    for i in range(1, grid_size * grid_size):  # Tiles numbered from 1 to N-1
        idx_current = state.index(i)
        idx_goal = goal_state.index(i)
        x_current, y_current = idx_current % grid_size, idx_current // grid_size
        x_goal, y_goal = idx_goal % grid_size, idx_goal // grid_size
        distance += abs(x_current - x_goal) + abs(y_current - y_goal)
    return distance

def linear_conflict(state, goal_state, grid_size=5):
    """Calculate the number of linear conflicts in rows and columns."""
    conflict = 0
    state_grid = np.array(state).reshape(grid_size, grid_size)
    goal_grid = np.array(goal_state).reshape(grid_size, grid_size)
    
    # Analyze rows
    for row in range(grid_size):
        state_row = state_grid[row, :]
        goal_row = goal_grid[row, :]
        # Create mapping from tile value to goal position in the row
        goal_positions = {tile: idx for idx, tile in enumerate(goal_row) if tile != 0}
        for i in range(grid_size):
            for j in range(i + 1, grid_size):
                tile_i = state_row[i]
                tile_j = state_row[j]
                if tile_i != 0 and tile_j != 0:
                    if (tile_i in goal_positions and tile_j in goal_positions and
                        goal_positions[tile_i] > goal_positions[tile_j]):
                        conflict += 1

    # Analyze columns
    for col in range(grid_size):
        state_col = state_grid[:, col]
        goal_col = goal_grid[:, col]
        # Create mapping from tile value to goal position in the column
        goal_positions = {tile: idx for idx, tile in enumerate(goal_col) if tile != 0}
        for i in range(grid_size):
            for j in range(i + 1, grid_size):
                tile_i = state_col[i]
                tile_j = state_col[j]
                if tile_i != 0 and tile_j != 0:
                    if (tile_i in goal_positions and tile_j in goal_positions and
                        goal_positions[tile_i] > goal_positions[tile_j]):
                        conflict += 1

    return conflict * 2  # Each conflict adds two moves

def misplaced_tiles(state, goal_state):
    """Calculate the number of misplaced tiles."""
    return sum(1 for s, g in zip(state, goal_state) if s != 0 and s != g)

def is_solvable(state, grid_size=5):
    inversion_count = 0
    state_wo_blank = [tile for tile in state if tile != 0]
    for i in range(len(state_wo_blank)):
        for j in range(i + 1, len(state_wo_blank)):
            if state_wo_blank[i] > state_wo_blank[j]:
                inversion_count += 1

    if grid_size % 2 == 1:
        return inversion_count % 2 == 0
    else:
        blank_pos = state.index(0)
        blank_row_from_bottom = grid_size - (blank_pos // grid_size)
        if blank_row_from_bottom % 2 == 0:
            return inversion_count % 2 == 1
        else:
            return inversion_count % 2 == 0

def generate_random_puzzle_state(num_moves=100, grid_size=5):
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    state = list(goal_state)
    blank_pos = state.index(0)
    for _ in range(num_moves):
        moves = get_possible_moves(blank_pos, grid_size)
        move = random.choice(moves)
        state[blank_pos], state[move] = state[move], state[blank_pos]
        blank_pos = move
    return tuple(state)

# ---------------------------
# Normalization Functions
# ---------------------------

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# ---------------------------
# MLP Model Class
# ---------------------------

# class MLPModel(nn.Module):
#     def __init__(self, input_size, output_size, hidden_sizes=[128, 64]):
#         super(MLPModel, self).__init__()
#         layers = []
#         in_size = input_size
#         for h_size in hidden_sizes:
#             layers.append(nn.Linear(in_size, h_size))
#             layers.append(nn.ReLU())
#             in_size = h_size
#         layers.append(nn.Linear(in_size, output_size))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         output = self.net(x)
#         return output  # Removed x as it's not used in the loss here

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[256, 128, 64]):
        super(MLPModel, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        return output  # Return only output tensor



# ---------------------------
# A* Search Classes
# ---------------------------

class ModelNode:
    def __init__(self, state, f_star, g, h=None, parent=None):
        self.state = state
        self.f_star = f_star
        self.g = g
        self.h = h
        self.parent = parent

    def __lt__(self, other):
        return self.f_star < other.f_star


class AStarNode:
    def __init__(self, state, g, h, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

# ---------------------------
# Inference Function
# ---------------------------

def run_inference(start_state, goal_state, current_state, g, h, model, normalization_values, device):
    f_star_min = normalization_values['f_star_min']
    f_star_max = normalization_values['f_star_max']
    g_min = normalization_values['g_min']
    g_max = normalization_values['g_max']
    h_min = normalization_values['h_min']
    h_max = normalization_values['h_max']

    g_normalized = normalize(g, g_min, g_max)
    h_normalized = normalize(h, h_min, h_max)

    # Prepare input tensor
    grid_size = int(np.sqrt(len(start_state)))  # Assuming square grid
    encoded_start = np.array(start_state, dtype=np.float32)
    encoded_goal = np.array(goal_state, dtype=np.float32)
    current_state_array = np.array(current_state, dtype=np.float32)

    input_tensor = np.concatenate([
        encoded_start,       # Shape: (grid_size * grid_size,)
        encoded_goal,        # Shape: (grid_size * grid_size,)
        current_state_array, # Shape: (grid_size * grid_size,)
        np.array([g_normalized, h_normalized], dtype=np.float32)  # Shape: (2,)
    ])  # Total length: (grid_size * grid_size) * 3 + 2

    input_tensor = torch.from_numpy(input_tensor).float().to(device)
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, input_size]

    model.eval()
    with torch.no_grad():
        f_star_predicted = model(input_tensor)
    f_star_denormalized = denormalize(f_star_predicted.item(), f_star_min, f_star_max)

    return f_star_denormalized

# ---------------------------
# A* Search Algorithms
# ---------------------------

# Traditional A* Search
def astar_traditional(start_state, goal_state, max_runtime=60, grid_size=5):
    open_list = []
    closed_set = set()
    g_score = {start_state: 0}
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    start_node = AStarNode(start_state, g=0, h=h_start)
    heapq.heappush(open_list, start_node)
    expansions = 0
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        current = heapq.heappop(open_list)
        expansions += 1

        # if expansions>=30000:
        #     return None, None, expansions

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g = g_score[current.state] + cost

            if neighbor_state in g_score and tentative_g >= g_score[neighbor_state]:
                continue  # Not a better path

            g_score[neighbor_state] = tentative_g
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            next_node = AStarNode(neighbor_state, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions  # No path found

# Weighted A* Algorithm
def weighted_a_star(start_state, goal_state, epsilon=1.5, max_runtime=60, grid_size=5):
    open_list = []
    closed_set = set()
    g_score = {start_state: 0}
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    f_start = g_score[start_state] + epsilon * h_start
    start_node = AStarNode(start_state, g=0, h=h_start)
    heapq.heappush(open_list, (f_start, start_node))
    expansions = 0
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        current_f, current = heapq.heappop(open_list)
        expansions += 1

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g = current.g + cost

            if neighbor_state in g_score and tentative_g >= g_score[neighbor_state]:
                continue  # Not a better path

            g_score[neighbor_state] = tentative_g
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            f = tentative_g + epsilon * h
            next_node = AStarNode(neighbor_state, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, (f, next_node))

    return None, None, expansions  # No solution found

def ida_star(start_state, goal_state, max_runtime=60, grid_size=5):
    threshold = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    start_time = time.time()
    path = [start_state]
    expansions = 0

    def dfs(current_state, g, threshold, path, visited):
        nonlocal expansions
        f = g + manhattan_distance(current_state, goal_state, grid_size) + linear_conflict(current_state, goal_state, grid_size)
        if f > threshold:
            return f
        if current_state == goal_state:
            return 'FOUND', g
        expansions += 1
        min_threshold = float('inf')
        for neighbor_state, cost in get_neighbors(current_state, grid_size):
            if neighbor_state in visited:
                continue
            visited.add(neighbor_state)
            path.append(neighbor_state)
            t = dfs(neighbor_state, g + cost, threshold, path, visited)
            if isinstance(t, tuple) and t[0] == 'FOUND':
                return t
            if t == 'TIMEOUT':
                return 'TIMEOUT'
            if t < min_threshold:
                min_threshold = t
            path.pop()
            visited.remove(neighbor_state)
            if time.time() - start_time > max_runtime:
                return 'TIMEOUT'
        return min_threshold

    while True:
        visited = set([start_state])
        t = dfs(start_state, 0, threshold, path, visited)
        if isinstance(t, tuple) and t[0] == 'FOUND':
            total_cost = t[1]
            return path, total_cost, expansions
        if t == 'TIMEOUT':
            return None, None, expansions
        if t == float('inf'):
            return None, None, expansions  # No solution
        threshold = t


# Simplified Memory-Bounded A* (SMA*) Algorithm
def sma_star(start_state, goal_state, max_memory=10000, max_runtime=60, grid_size=5):
    open_list = []
    closed_set = {}
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    start_node = AStarNode(state=start_state, g=0, h=h_start)
    heapq.heappush(open_list, (start_node.f, start_node))
    expansions = 0
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout
        if len(open_list) > max_memory:
            # Remove the node with the highest f-value (least promising)
            open_list.pop()
            continue
        current_f, current = heapq.heappop(open_list)
        expansions += 1

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions

        closed_set[current.state] = current.g

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            tentative_g = current.g + cost
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            neighbor_node = AStarNode(state=neighbor_state, g=tentative_g, h=h, parent=current)
            f = neighbor_node.f

            if neighbor_state in closed_set and tentative_g >= closed_set[neighbor_state]:
                continue

            heapq.heappush(open_list, (f, neighbor_node))

    return None, None, expansions  # No solution found

# Recursive Best-First Search (RBFS) Algorithm
def rbfs(start_state, goal_state, max_runtime=60, grid_size=5):
    start_time = time.time()
    expansions = 0

    def rbfs_recursive(node, f_limit):
        nonlocal expansions
        if time.time() - start_time > max_runtime:
            return None, float('inf')  # Timeout
        if node.state == goal_state:
            return [node.state], node.f
        successors = []
        for neighbor_state, cost in get_neighbors(node.state, grid_size):
            g = node.g + cost
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            f = max(g + h, node.f)
            successors.append(AStarNode(state=neighbor_state, g=g, h=h, parent=node))
        if not successors:
            return None, float('inf')
        while True:
            # Sort successors based on their f-value
            successors.sort(key=lambda n: n.f)
            best = successors[0]
            if best.f > f_limit:
                return None, best.f
            alternative = successors[1].f if len(successors) > 1 else float('inf')
            result, best.f = rbfs_recursive(best, min(f_limit, alternative))
            expansions += 1
            if result is not None:
                return [node.state] + result, best.f
        return None, float('inf')

    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    root_node = AStarNode(state=start_state, g=0, h=h_start)
    path, _ = rbfs_recursive(root_node, float('inf'))
    if path:
        total_cost = len(path) - 1
        return path, total_cost, expansions
    else:
        return None, None, expansions

def gbfs(start_state, goal_state, max_runtime=60, grid_size=5):
    open_list = []
    closed_set = set()
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    heapq.heappush(open_list, (h_start, 0, start_state, [start_state]))  # Added g=0
    expansions = 0
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        _, current_g, current_state, path = heapq.heappop(open_list)
        expansions += 1

        if current_state == goal_state:
            total_cost = current_g
            return path, total_cost, expansions

        closed_set.add(current_state)

        for neighbor_state, cost in get_neighbors(current_state, grid_size):
            if neighbor_state in closed_set:
                continue
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            heapq.heappush(open_list, (h, current_g + cost, neighbor_state, path + [neighbor_state]))


def astar_with_model(start_state, goal_state, model, normalization_values, device, max_runtime=60, grid_size=5):
    open_list = []
    closed_set = set()
    g_score = {start_state: 0}
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    f_star_start = run_inference(start_state, goal_state, start_state, 0, h_start, model, normalization_values, device)
    h_model = f_star_start - 0  # Since g=0 for start node
    h = max(h_start, h_model)
    start_node = AStarNode(state=start_state, g=0, h=h)
    heapq.heappush(open_list, start_node)
    expansions = 0
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        current = heapq.heappop(open_list)
        expansions += 1

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g = current.g + cost

            if neighbor_state in g_score and tentative_g >= g_score[neighbor_state]:
                continue  # Not a better path

            g_score[neighbor_state] = tentative_g
            h_standard = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            f_star = run_inference(start_state, goal_state, neighbor_state, tentative_g, h_standard, model, normalization_values, device)
            h_model = f_star - tentative_g
            h = max(h_standard, h_model)
            next_node = AStarNode(state=neighbor_state, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions  # No path found


def focal_astar_with_model(start_state, goal_state, model, normalization_values, device, epsilon=1.1, max_runtime=60, grid_size=5):
    """
    Focal A* Search Algorithm utilizing a neural network model for heuristic predictions.

    Parameters:
        start_state (tuple): The initial state of the puzzle.
        goal_state (tuple): The goal state of the puzzle.
        model (MLPModel): The neural network model to predict f* values.
        normalization_values (dict): Normalization parameters for the model.
        device (torch.device): The device to run the model on (CPU or CUDA).
        epsilon (float): The epsilon factor for the focal list.
        max_runtime (int): Maximum allowed runtime in seconds.
        grid_size (int): Size of the puzzle grid.

    Returns:
        path (list): The sequence of states from start to goal.
        total_cost (float): The total cost of the solution path.
        expansions (int): Number of nodes expanded during the search.
    """
    class OpenList:
        def __init__(self):
            self.elements = []
            self.entry_finder = {}
            self.counter = count()

        def add_node(self, node):
            f = node.g + node.h
            if node.state in self.entry_finder:
                existing_count, existing_node = self.entry_finder[node.state]
                if node.g < existing_node.g:
                    self.remove_node(existing_node)
                else:
                    return
            count_value = next(self.counter)
            bisect.insort_left(self.elements, (f, count_value, node))
            self.entry_finder[node.state] = (count_value, node)

        def remove_node(self, node):
            if node.state in self.entry_finder:
                count_value, existing_node = self.entry_finder[node.state]
                f = existing_node.g + existing_node.h
                idx = bisect.bisect_left(self.elements, (f, count_value, existing_node))
                # Ensure the correct node is removed
                while idx < len(self.elements):
                    if self.elements[idx][2].state == node.state:
                        self.elements.pop(idx)
                        break
                    idx += 1
                del self.entry_finder[node.state]

        def get_f_min(self):
            if self.elements:
                return self.elements[0][0]
            else:
                return float('inf')

        def get_focal_nodes(self, f_min, epsilon):
            # Find the index where f > f_min * epsilon
            upper_bound = f_min * epsilon
            idx = bisect.bisect_right(self.elements, (upper_bound, float('inf'), None))
            return [node for (_, _, node) in self.elements[:idx]]

        def is_empty(self):
            return not self.elements

    start_time = time.time()
    open_list = OpenList()
    closed_set = set()
    g_score = {start_state: 0}

    # Compute the initial heuristic
    h_standard_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    # Run the model to get f_star for the start node
    f_star_start = run_inference(start_state, goal_state, start_state, 0, h_standard_start, model, normalization_values, device)
    # Compute h_model = f_star - g
    h_model_start = f_star_start - 0  # Since g = 0 for start node
    # Ensure admissible heuristic
    h_start = max(h_standard_start, h_model_start)

    start_node = ModelNode(state=start_state, f_star=f_star_start, g=0, h=h_start, parent=None)
    open_list.add_node(start_node)
    expansions = 0

    while not open_list.is_empty():
        current_time = time.time()
        if current_time - start_time > max_runtime:
            print("Focal A* with model: Timeout reached.")
            return None, None, expansions  # Timeout

        f_min = open_list.get_f_min()
        focal_nodes = open_list.get_focal_nodes(f_min, epsilon)

        if not focal_nodes:
            print("Focal A* with model: No nodes in focal list.")
            return None, None, expansions  # No nodes to expand

        # Select node from focal list with minimum f_star
        current = min(focal_nodes, key=lambda node: node.f_star)
        open_list.remove_node(current)
        expansions += 1

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g = current.g + cost

            if neighbor_state in g_score and tentative_g >= g_score[neighbor_state]:
                continue  # Not a better path

            # Update g_score
            g_score[neighbor_state] = tentative_g

            # Compute standard heuristic
            h_standard = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            # Run the model to get f_star
            f_star = run_inference(start_state, goal_state, neighbor_state, tentative_g, h_standard, model, normalization_values, device)
            # Compute h_model
            h_model = f_star - tentative_g
            # Ensure admissible heuristic
            h = max(h_standard, h_model)

            # Create the new node
            neighbor_node = ModelNode(state=neighbor_state, f_star=f_star, g=tentative_g, h=h, parent=current)

            # Add the neighbor to the open list
            open_list.add_node(neighbor_node)

    print("Focal A* with model: No solution found.")
    return None, None, expansions  # No path found



# ---------------------------
# Multi-Heuristic A* (MHA*) Algorithm
# ---------------------------

def generate_mha_heuristics(goal_state, grid_size=5, num_heuristics=4):
    """
    Generate multiple heuristics for MHA*.
    Parameters:
        goal_state (tuple): The goal state of the puzzle.
        grid_size (int): Size of the puzzle grid.
        num_heuristics (int): Number of additional heuristics to generate.
    Returns:
        heuristics (list): List containing h0 and additional heuristics [h0, h1, ..., hn].
    """
    heuristics = []
    # Base heuristic h0: MD + LC
    def h0(state):
        return manhattan_distance(state, goal_state, grid_size) + linear_conflict(state, goal_state, grid_size)
    heuristics.append(h0)

    for _ in range(num_heuristics):
        r1 = random.uniform(1.0, 5.0)
        r2 = random.uniform(1.0, 5.0)
        r3 = random.uniform(1.0, 5.0)
        def heuristic(state, r1=r1, r2=r2, r3=r3):
            md = manhattan_distance(state, goal_state, grid_size)
            lc = linear_conflict(state, goal_state, grid_size)
            mt = misplaced_tiles(state, goal_state)
            return r1 * md + r2 * lc + r3 * mt
        heuristics.append(heuristic)
    return heuristics


# ---------------------------
# Independent Multi-Heuristic A* (IMHA*) Algorithm
# ---------------------------

def imha_star(start_state, goal_state, heuristics, w1=1.5, w2=2.0, max_runtime=60, grid_size=5):
    """
    Independent Multi-Heuristic A* (IMHA*) Search Algorithm.

    Parameters:
        start_state (tuple): The initial state of the puzzle.
        goal_state (tuple): The goal state of the puzzle.
        heuristics (list): List of heuristic functions [h0, h1, ..., hn].
        w1 (float): Heuristic weight factor.
        w2 (float): Priority weight factor.
        max_runtime (int): Maximum allowed runtime in seconds.
        grid_size (int): Size of the puzzle grid.

    Returns:
        path (list): The sequence of states from start to goal.
        total_cost (float): The total cost of the solution path.
        expansions (int): Number of nodes expanded during the search.
    """
    import time
    start_time = time.time()
    n = len(heuristics) - 1  # Number of inadmissible heuristics

    # Initialize searches
    OPEN = [{} for _ in range(n+1)]  # OPEN lists for each search
    CLOSED = [set() for _ in range(n+1)]  # CLOSED sets for each search
    g_values = [{} for _ in range(n+1)]  # g-values for each search
    bp_values = [{} for _ in range(n+1)]  # Backpointers for each search

    # Initialize all searches
    for i in range(n+1):
        g_values[i][start_state] = 0
        bp_values[i][start_state] = None
        heap = []
        h = heuristics[i](start_state)
        key = g_values[i][start_state] + w1 * h
        heapq.heappush(heap, (key, start_state))
        OPEN[i] = heap

    expansions = 0

    while OPEN[0]:
        current_time = time.time()
        if current_time - start_time > max_runtime:
            print("IMHA*: Timeout reached.")
            return None, None, expansions  # Timeout

        for i in range(1, n+1):
            # Check if inadmissible search should be prioritized
            if OPEN[i]:
                inadmissible_key = OPEN[i][0][0]
                anchor_key = OPEN[0][0][0] if OPEN[0] else float('inf')
                if inadmissible_key <= w2 * anchor_key:
                    # Check termination condition
                    if goal_state in g_values[i] and g_values[i][goal_state] <= inadmissible_key:
                        # Reconstruct path
                        path = []
                        state = goal_state
                        while state is not None:
                            path.append(state)
                            state = bp_values[i][state]
                        return path[::-1], g_values[i][goal_state], expansions
                    # Expand node from OPEN_i
                    key, s = heapq.heappop(OPEN[i])
                    if s in CLOSED[i]:
                        continue
                    CLOSED[i].add(s)
                    expansions += 1
                    # Expand s in search i
                    for neighbor_state, cost in get_neighbors(s, grid_size):
                        if neighbor_state in CLOSED[i]:
                            continue
                        tentative_g = g_values[i][s] + cost
                        if neighbor_state not in g_values[i] or tentative_g < g_values[i][neighbor_state]:
                            g_values[i][neighbor_state] = tentative_g
                            bp_values[i][neighbor_state] = s
                            h = heuristics[i](neighbor_state)
                            key = g_values[i][neighbor_state] + w1 * h
                            heapq.heappush(OPEN[i], (key, neighbor_state))
                    continue  # Proceed to next inadmissible search

        # If no inadmissible search is prioritized, expand from anchor search
        if OPEN[0]:
            anchor_key = OPEN[0][0][0]
            # Check termination condition
            if goal_state in g_values[0] and g_values[0][goal_state] <= anchor_key:
                # Reconstruct path
                path = []
                state = goal_state
                while state is not None:
                    path.append(state)
                    state = bp_values[0][state]
                return path[::-1], g_values[0][goal_state], expansions
            # Expand node from OPEN_0
            key, s = heapq.heappop(OPEN[0])
            if s in CLOSED[0]:
                continue
            CLOSED[0].add(s)
            expansions += 1
            # Expand s in anchor search
            for neighbor_state, cost in get_neighbors(s, grid_size):
                if neighbor_state in CLOSED[0]:
                    continue
                tentative_g = g_values[0][s] + cost
                if neighbor_state not in g_values[0] or tentative_g < g_values[0][neighbor_state]:
                    g_values[0][neighbor_state] = tentative_g
                    bp_values[0][neighbor_state] = s
                    h = heuristics[0](neighbor_state)
                    key = g_values[0][neighbor_state] + w1 * h
                    heapq.heappush(OPEN[0], (key, neighbor_state))
        else:
            # OPEN[0] is empty, no solution
            print("IMHA*: No solution found.")
            return None, None, expansions  # No solution found

    print("IMHA*: No solution found.")
    return None, None, expansions  # No solution found

# ---------------------------
# Shared Multi-Heuristic A* (SMHA*) Algorithm
# ---------------------------

def smha_star(start_state, goal_state, heuristics, w1=1.5, w2=2.0, max_runtime=60, grid_size=5):
    """
    Shared Multi-Heuristic A* (SMHA*) Search Algorithm.

    Parameters:
        start_state (tuple): The initial state of the puzzle.
        goal_state (tuple): The goal state of the puzzle.
        heuristics (list): List of heuristic functions [h0, h1, ..., hn].
        w1 (float): Heuristic weight factor.
        w2 (float): Priority weight factor.
        max_runtime (int): Maximum allowed runtime in seconds.
        grid_size (int): Size of the puzzle grid.

    Returns:
        path (list): The sequence of states from start to goal.
        total_cost (float): The total cost of the solution path.
        expansions (int): Number of nodes expanded during the search.
    """
    import time
    start_time = time.time()
    n = len(heuristics) - 1  # Number of inadmissible heuristics

    # Initialize
    OPEN = [{} for _ in range(n+1)]  # OPEN lists for each search
    CLOSED = set()  # Shared CLOSED set
    g = {}  # Shared g-values
    bp = {}  # Shared backpointers

    g[start_state] = 0
    bp[start_state] = None

    # Insert start_state into all OPEN lists
    for i in range(n+1):
        heap = []
        h = heuristics[i](start_state)
        key = g[start_state] + w1 * h
        heapq.heappush(heap, (key, start_state))
        OPEN[i] = heap

    expansions = 0

    while OPEN[0]:
        current_time = time.time()
        if current_time - start_time > max_runtime:
            print("SMHA*: Timeout reached.")
            return None, None, expansions  # Timeout

        for i in range(1, n+1):
            # Check if inadmissible search should be prioritized
            if OPEN[i]:
                inadmissible_key = OPEN[i][0][0]
                anchor_key = OPEN[0][0][0] if OPEN[0] else float('inf')
                if inadmissible_key <= w2 * anchor_key:
                    # Check termination condition
                    if goal_state in g and g[goal_state] <= inadmissible_key:
                        # Reconstruct path
                        path = []
                        state = goal_state
                        while state is not None:
                            path.append(state)
                            state = bp[state]
                        return path[::-1], g[goal_state], expansions
                    # Expand node from OPEN_i
                    key, s = heapq.heappop(OPEN[i])
                    if s in CLOSED:
                        continue
                    CLOSED.add(s)
                    expansions += 1
                    # Expand s
                    for neighbor_state, cost in get_neighbors(s, grid_size):
                        if neighbor_state in CLOSED:
                            continue
                        tentative_g = g[s] + cost
                        if neighbor_state not in g or tentative_g < g[neighbor_state]:
                            g[neighbor_state] = tentative_g
                            bp[neighbor_state] = s
                            # Insert/update in OPEN_0
                            if neighbor_state not in CLOSED:
                                h_anchor = heuristics[0](neighbor_state)
                                key_anchor = g[neighbor_state] + w1 * h_anchor
                                heapq.heappush(OPEN[0], (key_anchor, neighbor_state))
                            # Insert/update in OPEN_i if key ≤ w2 * key_anchor
                            h_i = heuristics[i](neighbor_state)
                            key_i = g[neighbor_state] + w1 * h_i
                            if key_i <= w2 * key_anchor:
                                heapq.heappush(OPEN[i], (key_i, neighbor_state))
                    continue  # Proceed to next inadmissible search

        # If no inadmissible search is prioritized, expand from anchor search
        if OPEN[0]:
            anchor_key = OPEN[0][0][0]
            # Check termination condition
            if goal_state in g and g[goal_state] <= anchor_key:
                # Reconstruct path
                path = []
                state = goal_state
                while state is not None:
                    path.append(state)
                    state = bp[state]
                return path[::-1], g[goal_state], expansions
            # Expand node from OPEN_0
            key, s = heapq.heappop(OPEN[0])
            if s in CLOSED:
                continue
            CLOSED.add(s)
            expansions += 1
            # Expand s
            for neighbor_state, cost in get_neighbors(s, grid_size):
                if neighbor_state in CLOSED:
                    continue
                tentative_g = g[s] + cost
                if neighbor_state not in g or tentative_g < g[neighbor_state]:
                    g[neighbor_state] = tentative_g
                    bp[neighbor_state] = s
                    # Insert/update in OPEN_0
                    if neighbor_state not in CLOSED:
                        h_anchor = heuristics[0](neighbor_state)
                        key_anchor = g[neighbor_state] + w1 * h_anchor
                        heapq.heappush(OPEN[0], (key_anchor, neighbor_state))
                    # Insert/update in OPEN_i if key ≤ w2 * key_anchor
                    for i in range(1, n+1):
                        h_i = heuristics[i](neighbor_state)
                        key_i = g[neighbor_state] + w1 * h_i
                        if key_i <= w2 * key_anchor:
                            heapq.heappush(OPEN[i], (key_i, neighbor_state))
        else:
            # OPEN[0] is empty, no solution
            print("SMHA*: No solution found.")
            return None, None, expansions  # No solution found

    print("SMHA*: No solution found.")
    return None, None, expansions  # No solution found




# ---------------------------
# Potential Search Algorithm
# ---------------------------

class PotentialNode:
    def __init__(self, state, g, h, flnr, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.flnr = flnr
        self.parent = parent

    def __lt__(self, other):
        return self.flnr < other.flnr

def potential_search(start_state, goal_state, C, max_runtime=60, grid_size=5):
    import time
    start_time = time.time()
    open_list = []
    closed_set = set()
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    g_start = 0

    if g_start >= C:
        return None, None, 0

    flnr_start = h_start / (C - g_start + 1e-8) if C - g_start > 0 else float('inf')
    start_node = PotentialNode(state=start_state, g=g_start, h=h_start, flnr=flnr_start, parent=None)
    heapq.heappush(open_list, start_node)
    expansions = 0

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        current = heapq.heappop(open_list)
        expansions += 1

        if current.state == goal_state and current.g <= C:
            # Reconstruct path and compute total cost
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1], current.g, expansions

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            tentative_g = current.g + cost

            if tentative_g > C:
                continue  # Prune nodes that exceed cost bound

            if neighbor_state in closed_set:
                continue

            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)

            if C - tentative_g > 0:
                flnr = h / (C - tentative_g + 1e-8)  # Added epsilon to avoid division by zero
            else:
                continue  # Cannot proceed if denominator is zero or negative

            neighbor_node = PotentialNode(state=neighbor_state, g=tentative_g, h=h, flnr=flnr, parent=current)
            heapq.heappush(open_list, neighbor_node)

    return None, None, expansions  # No solution found

# ---------------------------
# Optimistic Search Algorithm
# ---------------------------

class OptimisticNode:
    def __init__(self, state, g, h, h_hat, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.h_hat = h_hat
        self.f = g + h
        self.f_hat = g + h_hat
        self.parent = parent

    def __lt__(self, other):
        # The comparison will be based on f or f_hat depending on the mode
        if OptimisticNode.use_f_hat:
            return self.f_hat < other.f_hat
        else:
            return self.f < other.f

    # Class variable to determine whether to use f or f_hat
    use_f_hat = True

def optimistic_search(start_state, goal_state, w=1.5, max_runtime=60, grid_size=5):
    import time
    start_time = time.time()
    open_list = []
    closed_set = set()
    expansions = 0
    incumbent_solution_cost = float('inf')
    best_solution_node = None

    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    h_hat_start = w * h_start

    start_node = OptimisticNode(state=start_state, g=0, h=h_start, h_hat=h_hat_start, parent=None)
    heapq.heappush(open_list, start_node)

    OptimisticNode.use_f_hat = True  # Start by using f_hat

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        current = heapq.heappop(open_list)
        expansions += 1

        if current.state == goal_state:
            if current.g < incumbent_solution_cost:
                incumbent_solution_cost = current.g
                best_solution_node = current
            # Return the first solution found
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1], current.g, expansions

        closed_set.add(current.state)

        # Decide whether to use f_hat or f for the priority
        if open_list:
            best_f_hat = open_list[0].f_hat
            if best_f_hat >= incumbent_solution_cost:
                # Switch to using f (A*)
                OptimisticNode.use_f_hat = False
                # Rebuild the heap based on f
                temp_list = list(open_list)
                open_list.clear()
                heapq.heapify(open_list)  # Empty the heap
                for node in temp_list:
                    heapq.heappush(open_list, node)
        else:
            OptimisticNode.use_f_hat = False

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g = current.g + cost
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            h_hat = w * h

            neighbor_node = OptimisticNode(state=neighbor_state, g=tentative_g, h=h, h_hat=h_hat, parent=current)
            heapq.heappush(open_list, neighbor_node)

    return None, None, expansions  # No solution found

# ---------------------------
# Multi-Heuristic A* (MHA*) Algorithm
# ---------------------------

def generate_mha_heuristics(goal_state, grid_size=5, num_heuristics=4):
    """Generate multiple heuristics for MHA*."""
    heuristics = []
    # Base heuristic h0: MD + LC
    def h0(state):
        return manhattan_distance(state, goal_state, grid_size) + linear_conflict(state, goal_state, grid_size)
    heuristics.append(h0)

    for _ in range(num_heuristics):
        r1 = random.uniform(1.0, 5.0)
        r2 = random.uniform(1.0, 5.0)
        r3 = random.uniform(1.0, 5.0)
        def heuristic(state, r1=r1, r2=r2, r3=r3):
            md = manhattan_distance(state, goal_state, grid_size)
            lc = linear_conflict(state, goal_state, grid_size)
            mt = misplaced_tiles(state, goal_state)
            return r1 * md + r2 * lc + r3 * mt
        heuristics.append(heuristic)
    return heuristics

class MHAStarNode:
    def __init__(self, state, g, h_values, parent=None):
        self.state = state
        self.g = g
        self.h_values = h_values  # List of heuristic values
        self.parent = parent
        self.f = min([g + h for h in h_values])  # Use the minimum f-value across heuristics

    def __lt__(self, other):
        return self.f < other.f

def multi_heuristic_a_star(start_state, goal_state, heuristics, max_runtime=60, grid_size=5):
    open_list = []
    closed_set = set()
    g_score = {start_state: 0}
    h_values_start = [h(start_state) for h in heuristics]
    start_node = MHAStarNode(start_state, g=0, h_values=h_values_start)
    heapq.heappush(open_list, start_node)
    expansions = 0
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions  # Timeout

        current = heapq.heappop(open_list)
        expansions += 1

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1], total_cost, expansions

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g = current.g + cost

            if neighbor_state in g_score and tentative_g >= g_score[neighbor_state]:
                continue  # Not a better path

            g_score[neighbor_state] = tentative_g
            h_values = [h(neighbor_state) for h in heuristics]
            next_node = MHAStarNode(neighbor_state, tentative_g, h_values, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions  # No path found

# ---------------------------
# Generate Puzzles Function
# ---------------------------

def generate_puzzles(num_puzzles, grid_size):
    puzzles = []
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    for _ in range(num_puzzles):
        start_state = generate_random_puzzle_state(num_moves=random.randint(120, 130), grid_size=grid_size)
        while not is_solvable(start_state, grid_size):
            start_state = generate_random_puzzle_state(num_moves=random.randint(120, 130), grid_size=grid_size)
        puzzles.append((start_state, goal_state))
    return puzzles

# ---------------------------
# Assessment Function
# ---------------------------

def run_experiments(models, normalization_values, device, grid_size, num_puzzles=5, max_runtime=60, epsilons=[1.5, 2.0, 3.0], output_csv="algorithm_results.csv", min_expansions=5000, max_expansions=30000, max_attempts_per_puzzle=10):
    import math  # Ensure math is imported
    
    sizes = [grid_size]
    max_runtime = max_runtime  # e.g., 60 seconds

    # Initialize results dictionary
    results = {}
    algorithms = [
        'Traditional A*',
        'Weighted A* w=1.5', 'Weighted A* w=2.0', 'Weighted A* w=3.0',
        'IDA*',
        'SMA*',
        'Potential Search',
        'Optimistic Search w=1.5', 'Optimistic Search w=2.0', 'Optimistic Search w=3.0',
        'IMHA*',
        'SMHA*'
    ]
    # Add model-based algorithms (A* and Focal A*)
    for model_name in models.keys():
        algorithms.append(f'{model_name} A*')
        algorithms.append(f'{model_name} Focal A*')
    for algo in algorithms:
        results[algo] = {}
        for size in sizes:
            tile_count = size * size - 1
            results[algo][tile_count] = {
                'Puzzles_Solved': 0,
                'Total_Solution_Cost': 0.0,
                'Solution_Costs': [],
                'Total_Expansions': 0.0,
                'Expansions': []
            }

    for size in sizes:
        tile_count = size * size - 1
        print(f"Running experiments for {tile_count}-tile puzzles.")
        puzzles_solved = 0
        attempts = 0

        while puzzles_solved < num_puzzles and attempts < num_puzzles * max_attempts_per_puzzle:
            attempts += 1
            print(f"\nAttempt {attempts} for puzzle {puzzles_solved + 1}/{num_puzzles}")

            # Generate a new puzzle
            start_state = generate_random_puzzle_state(num_moves=random.randint(100, 130), grid_size=size)
            while not is_solvable(start_state, grid_size):
                start_state = generate_random_puzzle_state(num_moves=random.randint(100, 130), grid_size=size)
            goal_state = tuple(range(1, size * size)) + (0,)

            # Run Traditional A* to check expansions
            print("Running Traditional A* to check expansion criteria")
            path, cost, expansions = astar_traditional(start_state, goal_state, max_runtime=max_runtime, grid_size=size)
            print(f"Expanded {expansions} nodes, cost is {cost}")

            if expansions < min_expansions or expansions > max_expansions:
                print(f"Expansion count {expansions} not within [{min_expansions}, {max_expansions}]. Regenerating puzzle.")
                continue  # Regenerate a new puzzle

            if path is not None:
                # Accepted puzzle
                puzzles_solved += 1
                print(f"Puzzle {puzzles_solved} accepted.")

                # Record Traditional A* results
                results['Traditional A*'][tile_count]['Puzzles_Solved'] += 1
                results['Traditional A*'][tile_count]['Total_Solution_Cost'] += cost
                results['Traditional A*'][tile_count]['Solution_Costs'].append(cost)
                results['Traditional A*'][tile_count]['Total_Expansions'] += expansions
                results['Traditional A*'][tile_count]['Expansions'].append(expansions)
            else:
                print("Traditional A* did not find a solution within the time limit.")
                continue  # Regenerate a new puzzle

            # Weighted A* with different weights
            for w in epsilons:
                print(f"Running Weighted A* with weight {w}")
                path, cost, expansions = weighted_a_star(start_state, goal_state, epsilon=w, max_runtime=max_runtime, grid_size=size)
                print(f"Expanded {expansions} nodes, cost is {cost}")
                algo_name = f'Weighted A* w={w}'
                if path is not None:
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                else:
                    print(f"Weighted A* with weight {w} did not find a solution within the time limit.")

            # Potential Search
            print("Running Potential Search")
            # For Potential Search, set cost bound C based on Traditional A* cost
            C = cost * max(epsilons)
            path, cost, expansions = potential_search(start_state, goal_state, C=C, max_runtime=max_runtime, grid_size=size)
            print(f"Expanded {expansions} nodes, cost is {cost}")
            if path is not None:
                results['Potential Search'][tile_count]['Puzzles_Solved'] += 1
                results['Potential Search'][tile_count]['Total_Solution_Cost'] += cost
                results['Potential Search'][tile_count]['Solution_Costs'].append(cost)
                results['Potential Search'][tile_count]['Total_Expansions'] += expansions
                results['Potential Search'][tile_count]['Expansions'].append(expansions)
            else:
                print("Potential Search did not find a solution within the time limit.")

            # Optimistic Search with different weights
            for w in epsilons:
                print(f"Running Optimistic Search with weight {w}")
                path, cost, expansions = optimistic_search(start_state, goal_state, w=w, max_runtime=max_runtime, grid_size=size)
                print(f"Expanded {expansions} nodes, cost is {cost}")
                algo_name = f'Optimistic Search w={w}'
                if path is not None:
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                else:
                    print(f"Optimistic Search with weight {w} did not find a solution within the time limit.")

            # IMHA* and SMHA* require w1 and w2
            # Here, we define w1 and w2 based on a chosen epsilon, say the first epsilon
            chosen_epsilon = epsilons[0]  # You can choose a different strategy
            w2 = min(2.0, math.sqrt(chosen_epsilon))
            w1 = chosen_epsilon / w2
            print(f"Using w1={w1} and w2={w2} for MHA* algorithms.")

            # Generate heuristics
            heuristics = generate_mha_heuristics(goal_state, grid_size=size, num_heuristics=4)

            # IMHA*
            print("Running Independent Multi-Heuristic A* (IMHA*)")
            imha_path, imha_cost, imha_expansions = imha_star(
                start_state,
                goal_state,
                heuristics,
                w1=w1,
                w2=w2,
                max_runtime=max_runtime,
                grid_size=size
            )
            print(f"Expanded {imha_expansions} nodes, cost is {imha_cost}")
            # IMHA*
            algo_name = 'IMHA*'
            if imha_path is not None:
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += imha_cost
                results[algo_name][tile_count]['Solution_Costs'].append(imha_cost)
                results[algo_name][tile_count]['Total_Expansions'] += imha_expansions
                results[algo_name][tile_count]['Expansions'].append(imha_expansions)
            else:
                print("IMHA* did not find a solution within the time limit.")

            # SMHA*
            print("Running Shared Multi-Heuristic A* (SMHA*)")
            smha_path, smha_cost, smha_expansions = smha_star(
                start_state,
                goal_state,
                heuristics,
                w1=w1,
                w2=w2,
                max_runtime=max_runtime,
                grid_size=size
            )
            print(f"Expanded {smha_expansions} nodes, cost is {smha_cost}")
            algo_name = 'SMHA*'
            if smha_path is not None:
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += smha_cost
                results[algo_name][tile_count]['Solution_Costs'].append(smha_cost)
                results[algo_name][tile_count]['Total_Expansions'] += smha_expansions
                results[algo_name][tile_count]['Expansions'].append(smha_expansions)
            else:
                print("SMHA* did not find a solution within the time limit.")

            # Model-based A* and Focal A*
            for model_name, model_data in models.items():
                model = model_data['model']
                norm_values = normalization_values[model_name]

                # Regular A* with model
                print(f"Running A* with model {model_name}")
                path, cost, expansions = astar_with_model(start_state, goal_state, model, norm_values, device, max_runtime=max_runtime, grid_size=size)
                print(f"Expanded {expansions} nodes, cost is {cost}")
                algo_name = f'{model_name} A*'
                if path is not None:
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                else:
                    print(f"A* with model {model_name} did not find a solution within the time limit.")

                # Focal A* with model
                epsilon = 1.5  # Chosen epsilon value for Focal A*
                print(f"Running Focal A* with model {model_name} and epsilon {epsilon}")
                path, cost, expansions = focal_astar_with_model(start_state, goal_state, model, norm_values, device, epsilon=epsilon, max_runtime=max_runtime, grid_size=size)
                print(f"Expanded {expansions} nodes, cost is {cost}")
                algo_name = f'{model_name} Focal A*'
                if path is not None:
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                else:
                    print(f"Focal A* with model {model_name} did not find a solution within the time limit.")

    # Save the stats to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'Algorithm',
            'Puzzle_Size',
            'Puzzles_Solved',
            'Total_Puzzles',
            'Success_Rate',  # New Column
            'Average_Solution_Cost',
            'Solution_Cost_Percent_Difference',
            'Average_Expansions',
            'Expansion_Percent_Difference'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for algo_name, data in results.items():
            for tile_count, metrics in data.items():
                if metrics['Puzzles_Solved'] > 0:
                    avg_solution_cost = metrics['Total_Solution_Cost'] / metrics['Puzzles_Solved']
                    avg_expansions = metrics['Total_Expansions'] / metrics['Puzzles_Solved']
                else:
                    avg_solution_cost = 0
                    avg_expansions = 0
                # Fetch Traditional A* average cost and expansions for comparison
                a_star_metrics = results['Traditional A*'][tile_count]
                if a_star_metrics['Puzzles_Solved'] > 0:
                    a_star_avg_cost = a_star_metrics['Total_Solution_Cost'] / a_star_metrics['Puzzles_Solved']
                    a_star_avg_expansions = a_star_metrics['Total_Expansions'] / a_star_metrics['Puzzles_Solved']
                else:
                    a_star_avg_cost = 1  # Prevent division by zero
                    a_star_avg_expansions = 1  # Prevent division by zero
                if algo_name == 'Traditional A*':
                    solution_cost_percent_diff = 0.00
                    expansion_percent_diff = 0.00
                else:
                    if a_star_avg_cost > 0:
                        solution_cost_percent_diff = ((avg_solution_cost - a_star_avg_cost) / a_star_avg_cost) * 100
                    else:
                        solution_cost_percent_diff = 0.00
                    if a_star_avg_expansions > 0:
                        expansion_percent_diff = ((avg_expansions - a_star_avg_expansions) / a_star_avg_expansions) * 100
                    else:
                        expansion_percent_diff = 0.00
                # Calculate Success Rate
                success_rate = (metrics['Puzzles_Solved'] / num_puzzles) * 100
                writer.writerow({
                    'Algorithm': algo_name,
                    'Puzzle_Size': tile_count,
                    'Puzzles_Solved': metrics['Puzzles_Solved'],
                    'Total_Puzzles': num_puzzles,
                    'Success_Rate': f"{success_rate:.2f}%",  # New Field
                    'Average_Solution_Cost': round(avg_solution_cost, 2),
                    'Solution_Cost_Percent_Difference': f"{round(solution_cost_percent_diff, 2)}%",
                    'Average_Expansions': round(avg_expansions, 2),
                    'Expansion_Percent_Difference': f"{round(expansion_percent_diff, 2)}%"
                })

    print(f"Assessment results saved to {output_csv}")

# ---------------------------
# [Main Execution remains mostly unchanged with added parameters]
# ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run search algorithms on sliding tile puzzles.")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the puzzle grid (e.g., 5 for 5x5 grid)")
    parser.add_argument("--num_puzzles", type=int, default=10, help="Number of puzzles to solve")
    parser.add_argument("--max_runtime", type=int, default=6000, help="Maximum runtime per puzzle in seconds")
    parser.add_argument("--epsilons", type=float, nargs='+', default=[3.0], help="Epsilon weights for Weighted A* and Optimistic Search algorithms")
    parser.add_argument("--output_csv", type=str, default="algorithm_results.csv", help="Path to the output CSV file")
    parser.add_argument("--model_paths", type=str, nargs='+', help="Paths to the pre-trained MLPModels")
    parser.add_argument("--norm_paths", type=str, nargs='+', help="Paths to the normalization values pickle files corresponding to the models")
    # New arguments for expansion criteria
    parser.add_argument("--min_expansions", type=int, default=10000, help="Minimum number of node expansions required for A*")
    parser.add_argument("--max_expansions", type=int, default=50000, help="Maximum number of node expansions allowed for A*")
    parser.add_argument("--max_attempts_per_puzzle", type=int, default=10, help="Maximum attempts to generate a valid puzzle per required puzzle")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Models
    models = {}
    normalization_values = {}
    if args.model_paths and args.norm_paths:
        if len(args.model_paths) != len(args.norm_paths):
            print("The number of model paths and normalization paths must be the same.")
            exit(1)
        input_size = (args.grid_size * args.grid_size) * 3 + 2  # Adjusted input size
        for model_path, norm_path in zip(args.model_paths, args.norm_paths):
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model = MLPModel(input_size=input_size, output_size=1).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            # Load Normalization Values
            with open(norm_path, 'rb') as f:
                normalization_values_model = pickle.load(f)
            models[model_name] = {
                'model': model,
                'norm_values': normalization_values_model
            }
            normalization_values[model_name] = normalization_values_model
    else:
        print("No models provided. Proceeding without model-based A* search.")
        models = {}
        normalization_values = {}

    # Run the experiments
    run_experiments(
        models=models,
        normalization_values=normalization_values,
        device=device,
        grid_size=args.grid_size,
        num_puzzles=args.num_puzzles,
        max_runtime=args.max_runtime,
        epsilons=args.epsilons,
        output_csv=args.output_csv,
        min_expansions=args.min_expansions,
        max_expansions=args.max_expansions,
        max_attempts_per_puzzle=args.max_attempts_per_puzzle
    )