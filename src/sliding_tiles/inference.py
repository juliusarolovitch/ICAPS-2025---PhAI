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
    def __init__(self, state, g, h, f=None, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.f = f if f is not None else g + h
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
        encoded_start,
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
def astar_traditional(start_state, goal_state, max_runtime=60, grid_size=5, max_expansions=50000):
    open_list = []
    g_score = {start_state: 0}
    closed_set = set()  # Initialize the closed set
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    start_node = AStarNode(start_state, g=0, h=h_start)
    heapq.heappush(open_list, start_node)
    expansions = 0
    expansion_counts = {}  # Track how many times each state is expanded
    start_time = time.time()

    while open_list:
        if time.time() - start_time > max_runtime:
            return None, None, expansions, expansion_counts  # Timeout

        current = heapq.heappop(open_list)

        # Check if the current state has already been expanded
        if current.state in closed_set:
            continue  # Skip re-expansion

        closed_set.add(current.state)  # Mark the current state as expanded
        expansions += 1
        expansion_counts[current.state] = expansion_counts.get(current.state, 0) + 1

        if expansions > max_expansions:
            return None, None, expansions, expansion_counts

        if current.state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions, expansion_counts

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            tentative_g = current.g + cost  # Use current.g instead of g_score[current.state] + cost

            # If neighbor has been expanded, skip it
            if neighbor_state in closed_set:
                continue

            # If this path to neighbor is not better, skip it
            if neighbor_state in g_score and tentative_g >= g_score[neighbor_state]:
                continue  # Not a better path

            # Update the best g_score for the neighbor
            g_score[neighbor_state] = tentative_g

            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            next_node = AStarNode(neighbor_state, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions, expansion_counts  # No path found



# Weighted A* Algorithm
def weighted_a_star_with_reopening_tracking(start_state, goal_state, epsilon=1.5, max_runtime=60, grid_size=5, max_expansions=50000, allow_reexpansions=True):
    open_list = []
    closed_set = set()
    g_score = {start_state: 0}
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    f_start = g_score[start_state] + epsilon * h_start
    start_node = AStarNode(start_state, g=0, h=h_start, f=f_start)
    heapq.heappush(open_list, (f_start, start_node))
    expansions = 0
    reopening_count = 0
    start_time = time.time()

    # Use a dictionary to keep track of nodes in the open list
    open_dict = {start_state: start_node}

    while open_list:
        if time.time() - start_time > max_runtime or expansions > max_expansions:
            return None, None, expansions, reopening_count  # Timeout

        current_f, current = heapq.heappop(open_list)
        current_state = current.state

        if current_state in closed_set:
            continue  # Skip nodes already expanded

        expansions += 1

        if current_state == goal_state:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions, reopening_count

        closed_set.add(current_state)
        open_dict.pop(current_state, None)  # Remove from open_dict

        for neighbor_state, cost in get_neighbors(current_state, grid_size):
            tentative_g = current.g + cost

            if not allow_reexpansions and neighbor_state in closed_set:
                continue

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
                f = tentative_g + epsilon * h
                next_node = AStarNode(neighbor_state, g=tentative_g, h=h, f=f, parent=current)

                if neighbor_state in closed_set:
                    if allow_reexpansions:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        heapq.heappush(open_list, (f, next_node))
                        open_dict[neighbor_state] = next_node
                elif neighbor_state in open_dict:
                    existing_node = open_dict[neighbor_state]
                    if tentative_g < existing_node.g:
                        # Update node in open_list
                        existing_node.g = tentative_g
                        existing_node.f = f
                        existing_node.parent = current
                        heapq.heappush(open_list, (existing_node.f, existing_node))
                else:
                    heapq.heappush(open_list, (f, next_node))
                    open_dict[neighbor_state] = next_node

    return None, None, expansions, reopening_count  # No solution found



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

def astar_with_model(start_state, goal_state, model, normalization_values, device, max_runtime=60, grid_size=5, max_expansions=50000, allow_reexpansions=True):
    open_list = []
    g_score = {start_state: 0}
    expansions = 0
    expansion_counts = {}  # Track expansions per state
    reopening_count = 0
    closed_set = set()
    open_dict = {}

    # Compute initial heuristic
    h_standard = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    f_star_start = run_inference(start_state, goal_state, start_state, g=0, h=h_standard, model=model, normalization_values=normalization_values, device=device)
    h_model = f_star_start - 0
    h = max(h_standard, h_model)

    start_node = AStarNode(state=start_state, g=0, h=h)
    heapq.heappush(open_list, (start_node.f, start_node))
    open_dict[start_state] = start_node
    start_time = time.time()

    while open_list:
        if (time.time() - start_time > max_runtime) or (expansions > max_expansions):
            return None, None, expansions, reopening_count  # Timeout

        current_f, current = heapq.heappop(open_list)
        current_state = current.state

        if current_state in closed_set:
            continue

        expansions += 1
        expansion_counts[current_state] = expansion_counts.get(current_state, 0) + 1

        if current_state == goal_state:
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions, reopening_count

        closed_set.add(current_state)
        open_dict.pop(current_state, None)

        for neighbor_state, cost in get_neighbors(current_state, grid_size):
            tentative_g = current.g + cost

            if not allow_reexpansions and neighbor_state in closed_set:
                continue

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                h_standard = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
                f_star = run_inference(start_state, goal_state, neighbor_state, tentative_g, h_standard, model, normalization_values, device)
                h_model = f_star - tentative_g
                h = max(h_standard, h_model)
                neighbor_node = AStarNode(neighbor_state, g=tentative_g, h=h, parent=current)

                if neighbor_state in closed_set:
                    if allow_reexpansions:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                        open_dict[neighbor_state] = neighbor_node
                elif neighbor_state in open_dict:
                    existing_node = open_dict[neighbor_state]
                    if tentative_g < existing_node.g:
                        existing_node.g = tentative_g
                        existing_node.h = h
                        existing_node.f = tentative_g + h
                        existing_node.parent = current
                        heapq.heappush(open_list, (existing_node.f, existing_node))
                else:
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                    open_dict[neighbor_state] = neighbor_node

    return None, None, expansions, reopening_count  # No solution found


def focal_astar_with_model(start_state, goal_state, model, normalization_values, device, epsilon=1.1, max_runtime=60, grid_size=5, max_expansions=50000, allow_reexpansions=True):
    class OpenList:
        def __init__(self):
            self.elements = []
            self.entry_finder = {}
            self.counter = count()

        def add_or_update_node(self, node):
            f = node.g + node.h
            node_key = node.state
            if node_key in self.entry_finder:
                existing_count, existing_node = self.entry_finder[node_key]
                if node.g < existing_node.g:
                    self.remove_node(existing_node)
                else:
                    return
            count_value = next(self.counter)
            bisect.insort_left(self.elements, (f, count_value, node))
            self.entry_finder[node_key] = (count_value, node)

        def remove_node(self, node):
            node_key = node.state
            if node_key in self.entry_finder:
                count_value, existing_node = self.entry_finder[node_key]
                f = existing_node.g + existing_node.h
                idx = bisect.bisect_left(self.elements, (f, count_value, existing_node))
                while idx < len(self.elements):
                    if self.elements[idx][2].state == node.state:
                        self.elements.pop(idx)
                        break
                    idx += 1
                del self.entry_finder[node_key]

        def get_f_min(self):
            if self.elements:
                return self.elements[0][0]
            else:
                return float('inf')

        def get_focal_nodes(self, f_min, epsilon):
            upper_bound = f_min * epsilon
            idx = bisect.bisect_right(self.elements, (upper_bound, float('inf'), None))
            return [node for (_, _, node) in self.elements[:idx]]

        def is_empty(self):
            return not self.elements

    start_time = time.time()
    open_list = OpenList()
    g_score = {start_state: 0}
    expansions = 0
    expansion_counts = {}
    reopening_count = 0
    closed_set = set()

    h_standard = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    f_star_start = run_inference(start_state, goal_state, start_state, g=0, h=h_standard, model=model, normalization_values=normalization_values, device=device)
    h_model = f_star_start - 0
    h = max(h_standard, h_model)

    start_node = ModelNode(state=start_state, f_star=f_star_start, g=0, h=h, parent=None)
    open_list.add_or_update_node(start_node)

    while not open_list.is_empty():
        if (time.time() - start_time > max_runtime) or (expansions > max_expansions):
            print("Focal A* with model: Timeout reached.")
            return None, None, expansions, reopening_count  # Timeout

        f_min = open_list.get_f_min()
        focal_nodes = open_list.get_focal_nodes(f_min, epsilon)

        if not focal_nodes:
            print("Focal A* with model: No nodes in focal list.")
            return None, None, expansions, reopening_count  # No nodes to expand

        current = min(focal_nodes, key=lambda node: node.f_star)
        open_list.remove_node(current)
        current_state = current.state

        if current_state in closed_set:
            continue

        expansions += 1
        expansion_counts[current_state] = expansion_counts.get(current_state, 0) + 1

        if current_state == goal_state:
            path = []
            total_cost = current.g
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1], total_cost, expansions, reopening_count

        closed_set.add(current_state)

        for neighbor_state, cost in get_neighbors(current_state, grid_size):
            tentative_g = current.g + cost

            if not allow_reexpansions and neighbor_state in closed_set:
                continue

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                h_standard = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
                f_star = run_inference(start_state, goal_state, neighbor_state, tentative_g, h_standard, model, normalization_values, device)
                h_model = f_star - tentative_g
                h = max(h_standard, h_model)
                neighbor_node = ModelNode(state=neighbor_state, f_star=f_star, g=tentative_g, h=h, parent=current)

                if neighbor_state in closed_set:
                    if allow_reexpansions:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        open_list.add_or_update_node(neighbor_node)
                else:
                    open_list.add_or_update_node(neighbor_node)

    print("Focal A* with model: No solution found.")
    return None, None, expansions, reopening_count  # No path found



# ---------------------------
# Independent Multi-Heuristic A* (IMHA*) Algorithm
# ---------------------------

def imha_star_with_reopening_tracking(start_state, goal_state, heuristics, w1=1.5, w2=2.0, max_runtime=60, grid_size=5, max_expansions=50000, allow_reexpansions=True):
    """
    Independent Multi-Heuristic A* (IMHA*) Search Algorithm with re-opening tracking.
    """
    start_time = time.time()
    n = len(heuristics) - 1  # Number of inadmissible heuristics

    # Initialize searches
    OPEN = [[] for _ in range(n+1)]  # OPEN lists for each search
    open_dicts = [{} for _ in range(n+1)]  # Dictionaries to track nodes in open lists
    CLOSED = [set() for _ in range(n+1)]  # CLOSED sets for each search
    g_values = [{} for _ in range(n+1)]  # g-values for each search
    bp_values = [{} for _ in range(n+1)]  # Backpointers for each search
    expansions = 0
    reopening_count = 0

    # Initialize all searches
    for i in range(n+1):
        g_values[i][start_state] = 0
        bp_values[i][start_state] = None
        h = heuristics[i](start_state)
        key = g_values[i][start_state] + w1 * h
        node = (key, start_state)
        heapq.heappush(OPEN[i], node)
        open_dicts[i][start_state] = node

    while any(OPEN):
        current_time = time.time()
        if current_time - start_time > max_runtime or expansions > max_expansions:
            print("IMHA*: Timeout reached.")
            return None, None, expansions, reopening_count  # Timeout

        for i in range(1, n+1):
            if OPEN[i]:
                inadmissible_key, s = OPEN[i][0]
                anchor_key = OPEN[0][0][0] if OPEN[0] else float('inf')
                if inadmissible_key <= w2 * anchor_key:
                    if goal_state in g_values[i] and g_values[i][goal_state] <= inadmissible_key:
                        # Reconstruct path
                        path = []
                        state = goal_state
                        while state is not None:
                            path.append(state)
                            state = bp_values[i][state]
                        return path[::-1], g_values[i][goal_state], expansions, reopening_count
                    # Expand node from OPEN_i
                    key, s = heapq.heappop(OPEN[i])
                    open_dicts[i].pop(s, None)
                    if s in CLOSED[i]:
                        continue
                    expansions += 1
                    CLOSED[i].add(s)
                    for neighbor_state, cost in get_neighbors(s, grid_size):
                        tentative_g = g_values[i][s] + cost
                        if tentative_g < g_values[i].get(neighbor_state, float('inf')):
                            g_values[i][neighbor_state] = tentative_g
                            bp_values[i][neighbor_state] = s
                            h = heuristics[i](neighbor_state)
                            key = tentative_g + w1 * h
                            node = (key, neighbor_state)
                            if neighbor_state in CLOSED[i]:
                                if allow_reexpansions:
                                    CLOSED[i].remove(neighbor_state)
                                    reopening_count += 1
                                    heapq.heappush(OPEN[i], node)
                                    open_dicts[i][neighbor_state] = node
                            elif neighbor_state in open_dicts[i]:
                                existing_node = open_dicts[i][neighbor_state]
                                if key < existing_node[0]:
                                    # Update node in open_list
                                    heapq.heappush(OPEN[i], node)
                                    open_dicts[i][neighbor_state] = node
                            else:
                                heapq.heappush(OPEN[i], node)
                                open_dicts[i][neighbor_state] = node
                    continue  # Proceed to next inadmissible search

        # If no inadmissible search is prioritized, expand from anchor search
        if OPEN[0]:
            anchor_key, s = heapq.heappop(OPEN[0])
            open_dicts[0].pop(s, None)
            if s in CLOSED[0]:
                continue
            expansions += 1
            CLOSED[0].add(s)
            if goal_state in g_values[0] and g_values[0][goal_state] <= anchor_key:
                # Reconstruct path
                path = []
                state = goal_state
                while state is not None:
                    path.append(state)
                    state = bp_values[0][state]
                return path[::-1], g_values[0][goal_state], expansions, reopening_count
            for neighbor_state, cost in get_neighbors(s, grid_size):
                tentative_g = g_values[0][s] + cost
                if tentative_g < g_values[0].get(neighbor_state, float('inf')):
                    g_values[0][neighbor_state] = tentative_g
                    bp_values[0][neighbor_state] = s
                    h = heuristics[0](neighbor_state)
                    key = tentative_g + w1 * h
                    node = (key, neighbor_state)
                    if neighbor_state in CLOSED[0]:
                        if allow_reexpansions:
                            CLOSED[0].remove(neighbor_state)
                            reopening_count += 1
                            heapq.heappush(OPEN[0], node)
                            open_dicts[0][neighbor_state] = node
                    elif neighbor_state in open_dicts[0]:
                        existing_node = open_dicts[0][neighbor_state]
                        if key < existing_node[0]:
                            # Update node in open_list
                            heapq.heappush(OPEN[0], node)
                            open_dicts[0][neighbor_state] = node
                    else:
                        heapq.heappush(OPEN[0], node)
                        open_dicts[0][neighbor_state] = node
        else:
            print("IMHA*: No solution found.")
            return None, None, expansions, reopening_count  # No solution found

    print("IMHA*: No solution found.")
    return None, None, expansions, reopening_count  # No solution found


# ---------------------------
# Shared Multi-Heuristic A* (SMHA*) Algorithm
# ---------------------------

def smha_star(start_state, goal_state, heuristics, w1=1.5, w2=2.0, max_runtime=60, grid_size=5, max_expansions=50000):
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
        if current_time - start_time > max_runtime or expansions>max_expansions:
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

from collections import defaultdict
import heapq
import time

class PotentialNode:
    def __init__(self, state, g, h, flnr, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.flnr = flnr  # Potential function value
        self.parent = parent

    def __lt__(self, other):
        return self.flnr < other.flnr

def potential_search_with_reopening_tracking(start_state, goal_state, B=3.0, max_runtime=60, grid_size=5, max_expansions=50000, allow_reexpansions=True):
    start_time = time.time()
    open_list = []
    g_score = {start_state: 0}
    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    f_min = h_start  # Since g=0 at the start
    C = B * f_min

    # Initialize the starting node
    start_node = PotentialNode(state=start_state, g=0, h=h_start, flnr=h_start / (C - 0 + 1e-8), parent=None)
    heapq.heappush(open_list, start_node)
    expansions = 0
    expansion_counts = defaultdict(int)
    reshuffles = 0  # Initialize reshuffle counter
    reopening_count = 0
    closed_set = set()
    open_dict = {start_state: start_node}

    while open_list:
        # Check for runtime timeout or expansion limit
        if (time.time() - start_time > max_runtime) or (expansions > max_expansions):
            print("Potential Search: Timeout or expansion limit reached.")
            re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
            print(f"Number of reshuffles: {reshuffles}")
            return None, None, expansions, reopening_count, reshuffles

        # Pop the node with the lowest potential
        current = heapq.heappop(open_list)
        current_state = current.state

        if current_state in closed_set:
            continue  # Skip already expanded nodes

        expansions += 1
        expansion_counts[current_state] += 1

        if current_state == goal_state and current.g <= C:
            # Reconstruct path
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            print(f"Potential Search: Path found with cost {current.g} in {expansions} expansions.")
            return path[::-1], current.g, expansions, reopening_count, reshuffles

        closed_set.add(current_state)
        open_dict.pop(current_state, None)

        for neighbor_state, cost in get_neighbors(current_state, grid_size):
            tentative_g = current.g + cost

            if tentative_g > C:
                continue

            if not allow_reexpansions and neighbor_state in closed_set:
                continue

            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            f = tentative_g + h

            # Update f_min if necessary
            cost_diff = f_min - f
            if f < f_min and cost_diff >= math.sqrt(2):
                f_min = f
                C = B * f_min
                reshuffles += 1

                # Recompute flnr for all nodes in the open list
                new_open_list = []
                open_dict.clear()
                while open_list:
                    node = heapq.heappop(open_list)
                    updated_flnr = node.h / (C - node.g + 1e-8) if (C - node.g) > 0 else float('inf')
                    node.flnr = updated_flnr
                    heapq.heappush(new_open_list, node)
                    open_dict[node.state] = node
                open_list = new_open_list

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                flnr = h / (C - tentative_g + 1e-8) if (C - tentative_g) > 0 else float('inf')
                neighbor_node = PotentialNode(state=neighbor_state, g=tentative_g, h=h, flnr=flnr, parent=current)

                if neighbor_state in closed_set:
                    if allow_reexpansions:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        heapq.heappush(open_list, neighbor_node)
                        open_dict[neighbor_state] = neighbor_node
                elif neighbor_state in open_dict:
                    existing_node = open_dict[neighbor_state]
                    if tentative_g < existing_node.g:
                        # Update node in open_list
                        existing_node.g = tentative_g
                        existing_node.flnr = flnr
                        existing_node.parent = current
                        heapq.heappush(open_list, existing_node)
                else:
                    heapq.heappush(open_list, neighbor_node)
                    open_dict[neighbor_state] = neighbor_node

    print("Potential Search: No solution found.")
    return None, None, expansions, reopening_count, reshuffles  # No solution found


def potential_search_with_model(start_state, goal_state, model, normalization_values, device, B=3.0, max_runtime=60, grid_size=5):
    """
    Dynamic Potential Search (DPS) Algorithm with model for sliding tile puzzles,
    modified to prevent reshuffles when the cost difference is less than sqrt(2).
    
    Parameters:
        start_state (tuple): The initial state of the puzzle.
        goal_state (tuple): The goal state of the puzzle.
        model: The neural network model used to predict f-values.
        normalization_values (dict): Normalization values used by the model.
        device: The torch device.
        B (float): Suboptimality bound.
        max_runtime (int): Maximum allowed runtime in seconds.
        grid_size (int): Size of the puzzle grid.
    
    Returns:
        path (list): The sequence of states from start to goal.
        total_cost (float): The total cost of the solution path.
        expansions (int): Number of nodes expanded during the search.
        re_expansions (int): Number of states that were expanded more than once.
        reshuffles (int): Number of times the queue was reshuffled.
    """
    import math  # Ensure math is imported
    start_time = time.time()
    open_list = []
    g_score = {start_state: 0}
    
    # Compute h_start using standard heuristic
    h_standard_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    
    # Compute f_star for start node using the model
    f_star_start = run_inference(start_state, goal_state, start_state, g=0, h=h_standard_start, model=model, normalization_values=normalization_values, device=device)
    
    # Compute h_model
    h_model_start = f_star_start - 0  # Since g=0 for start node
    
    # Use h = max(h_standard, h_model) to ensure admissibility
    h_start = max(h_standard_start, h_model_start)
    
    # Compute f = g + h
    f_start = 0 + h_start  # g=0
    
    f_min = f_start  # Initialize f_min
    C = B * f_min
    
    # Initialize the starting node
    start_node = PotentialNode(state=start_state, g=0, h=h_start, flnr=h_start / (C - 0 + 1e-8), parent=None)
    heapq.heappush(open_list, start_node)
    expansions = 0
    expansion_counts = defaultdict(int)
    reshuffles = 0  # Initialize reshuffle counter
    
    while open_list:
        # Check for runtime timeout
        if time.time() - start_time > max_runtime:
            print("Dynamic Potential Search with Model: Timeout reached.")
            re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
            print(f"Number of reshuffles: {reshuffles}")
            return None, None, expansions, re_expansions, reshuffles
    
        # Expand the node with the lowest potential
        current = heapq.heappop(open_list)
        expansions += 1
        expansion_counts[current.state] += 1
    
        # Check if the goal has been reached within cost bound C
        if current.state == goal_state and current.g <= C:
            # Reconstruct path
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            print(f"DPS with Model: Path found with cost {current.g} in {expansions} expansions.")
            re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
            print(f"Number of reshuffles: {reshuffles}")
            return path[::-1], current.g, expansions, re_expansions, reshuffles
    
        # Explore neighbors
        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            tentative_g = current.g + cost
    
            # Prune paths that exceed the current cost bound
            if tentative_g > C:
                continue
    
            # Calculate h_standard for the neighbor
            h_standard = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
    
            # Compute f_star using the model
            f_star = run_inference(start_state, goal_state, neighbor_state, tentative_g, h_standard, model, normalization_values, device)
    
            # Compute h_model
            h_model = f_star - tentative_g
    
            # Use h = max(h_standard, h_model) to ensure admissibility
            h = max(h_standard, h_model)
    
            # Compute f = tentative_g + h
            f = tentative_g + h
    
            # Calculate the difference between current f_min and new f
            cost_diff = f_min - f
    
            # Update f_min and reshuffles only if the cost difference is >= sqrt(2)
            if f < f_min and cost_diff >= .1:
                f_min = f
                C = B * f_min
                reshuffles += 1  # Increment reshuffle counter
    
                # Recompute flnr for all nodes in the open list
                new_open_list = []
                while open_list:
                    node = heapq.heappop(open_list)
                    updated_flnr = node.h / (C - node.g + 1e-8) if (C - node.g) > 0 else float('inf')
                    node.flnr = updated_flnr
                    heapq.heappush(new_open_list, node)
                open_list = new_open_list
    
            # If a better path to the neighbor is found, update and add to OPEN
            if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                g_score[neighbor_state] = tentative_g
                flnr = h / (C - tentative_g + 1e-8) if (C - tentative_g) > 0 else float('inf')
                neighbor_node = PotentialNode(state=neighbor_state, g=tentative_g, h=h, flnr=flnr, parent=current)
                heapq.heappush(open_list, neighbor_node)
    
    print("Dynamic Potential Search with Model: No solution found.")
    re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
    print(f"Number of reshuffles: {reshuffles}")
    return None, None, expansions, re_expansions, reshuffles  # No solution found

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

def optimistic_search(start_state, goal_state, w=1.5, max_runtime=60, grid_size=5, max_expansions=50000, allow_reexpansions=True):
    import time
    start_time = time.time()
    open_list = []
    closed_set = set()
    expansions = 0
    incumbent_solution_cost = float('inf')
    best_solution_node = None

    h_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)
    h_hat_start = w * h_start  # Aggressive heuristic

    start_node = OptimisticNode(state=start_state, g=0, h=h_start, h_hat=h_hat_start, parent=None)
    heapq.heappush(open_list, start_node)

    OptimisticNode.use_f_hat = True  # Start with aggressive search

    while open_list:
        current_time = time.time()
        if current_time - start_time > max_runtime:
            print("Optimistic Search: Timeout reached.")
            return None, None, expansions  # Timeout

        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            print("Optimistic Search: Exceeded maximum expansions.")
            return None, None, expansions  # Exceeded max expansions

        if current.state == goal_state:
            if current.g < incumbent_solution_cost:
                incumbent_solution_cost = current.g
                best_solution_node = current
            # Do not return immediately; enter cleanup phase
            # Continue searching to verify admissibility
            # No need to add to closed_set yet

        if not allow_reexpansions:
            closed_set.add(current.state)

        # Determine the current search phase based on whether a solution has been found
        if best_solution_node:
            # Cleanup phase: use admissible heuristic
            OptimisticNode.use_f_hat = False
        else:
            # Aggressive phase: continue using h_hat
            OptimisticNode.use_f_hat = True

        # Check termination condition if in cleanup phase and solution exists
        if best_solution_node and not OptimisticNode.use_f_hat:
            if open_list:
                # Find the minimum f in open list
                f_min = open_list[0].f  # Since use_f_hat is False, heap is ordered by f
                if f_min >= incumbent_solution_cost / w:
                    # The current solution is within the desired suboptimality bound
                    # Reconstruct path and return
                    path = []
                    node = best_solution_node
                    while node:
                        path.append(node.state)
                        node = node.parent
                    return path[::-1], incumbent_solution_cost, expansions
            else:
                # Open list is empty; return the best solution found
                path = []
                node = best_solution_node
                while node:
                    path.append(node.state)
                    node = node.parent
                return path[::-1], incumbent_solution_cost, expansions

        # Explore neighbors
        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if not allow_reexpansions and neighbor_state in closed_set:
                continue

            tentative_g = current.g + cost

            # Calculate heuristics based on the current phase
            h = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)
            if OptimisticNode.use_f_hat:
                h_hat = w * h  # Aggressive heuristic
            else:
                h_hat = h  # Admissible heuristic

            # Check if this path is better
            if neighbor_state in closed_set and not allow_reexpansions:
                continue

            # Optionally, track g_scores if necessary
            # For simplicity, we'll skip g_score checks here

            neighbor_node = OptimisticNode(state=neighbor_state, g=tentative_g, h=h, h_hat=h_hat, parent=current)
            heapq.heappush(open_list, neighbor_node)

    # If loop exits without finding a solution
    print("Optimistic Search: No solution found.")
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
        start_state = generate_random_puzzle_state(num_moves=random.randint(250, 300), grid_size=grid_size)
        while not is_solvable(start_state, grid_size):
            start_state = generate_random_puzzle_state(num_moves=random.randint(150, 200), grid_size=grid_size)
        puzzles.append((start_state, goal_state))
    return puzzles

# ---------------------------
# Assessment Function
# ---------------------------

def run_experiments(models, normalization_values, device, grid_size, num_puzzles=5, max_runtime=60, epsilons=[1.5, 2.0, 3.0], output_csv="algorithm_results.csv", min_expansions=5000, max_expansions=50000, max_attempts_per_puzzle=10):
    import math  # Ensure math is imported

    sizes = [grid_size]
    max_runtime = max_runtime  # e.g., 60 seconds

    results = {}
    algorithms = [
        'Traditional A*',
        'Weighted A* w=1.5 (Re-Expansions Allowed)', 
        'Weighted A* w=1.5 (Re-Expansions Not Allowed)',
        'Weighted A* w=2.0 (Re-Expansions Allowed)', 
        'Weighted A* w=2.0 (Re-Expansions Not Allowed)',
        'Weighted A* w=3.0 (Re-Expansions Allowed)', 
        'Weighted A* w=3.0 (Re-Expansions Not Allowed)',
        'IDA*',
        'SMA*',
        'Potential Search (Re-Expansions Allowed)',
        'Potential Search (Re-Expansions Not Allowed)',
        'Optimistic Search w=1.5 (Re-Expansions Allowed)',
        'Optimistic Search w=1.5 (Re-Expansions Not Allowed)',
        'Optimistic Search w=2.0 (Re-Expansions Allowed)',
        'Optimistic Search w=2.0 (Re-Expansions Not Allowed)',
        'Optimistic Search w=3.0 (Re-Expansions Allowed)',
        'Optimistic Search w=3.0 (Re-Expansions Not Allowed)',
        'IMHA* (Re-Expansions Allowed)',          # **Added IMHA* with Re-Expansions Allowed**
        'IMHA* (Re-Expansions Not Allowed)',      # **Added IMHA* with Re-Expansions Not Allowed**
        'SMHA*'
    ]
    # Add model-based algorithms (A*, Focal A*, and Potential Search with Model)
    for model_name in models.keys():
        algorithms.extend([
            f'{model_name} A*',
            f'{model_name} Focal A*',
            f'{model_name} Potential Search (Re-Expansions Allowed)',
            f'{model_name} Potential Search (Re-Expansions Not Allowed)'
        ])

    # Initialize results for all algorithms
    for algo in algorithms:
        results[algo] = {}
        for size in sizes:
            tile_count = size * size - 1
            results[algo][tile_count] = {
                'Puzzles_Solved': 0,
                'Total_Solution_Cost': 0.0,
                'Solution_Costs': [],
                'Total_Expansions': 0.0,
                'Expansions': [],
                'ReExpansions_Percent': []  # Initialize list to store re-expansion percentages
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
            start_state = generate_random_puzzle_state(num_moves=random.randint(120, 180), grid_size=size)
            while not is_solvable(start_state, grid_size):
                start_state = generate_random_puzzle_state(num_moves=random.randint(120, 200), grid_size=size)
            goal_state = tuple(range(1, size * size)) + (0,)

            # Run Traditional A* to check expansions
            print("Running Traditional A* to check expansion criteria")
            path, cost, expansions, expansion_counts = astar_traditional(start_state, goal_state, max_runtime=max_runtime, grid_size=size, max_expansions=max_expansions)

            if path is not None and expansions > min_expansions and expansions < max_expansions:
                # Accepted puzzle
                puzzles_solved += 1
                print(f"Puzzle {puzzles_solved} accepted.")
                re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
                re_expansion_percent = (re_expansions / expansions) * 100 if expansions > 0 else 0.0
                print(f"Expanded {expansions} nodes, cost is {cost}, re-expansion percent is {re_expansion_percent}%")

                # Record Traditional A* results
                results['Traditional A*'][tile_count]['Puzzles_Solved'] += 1
                results['Traditional A*'][tile_count]['Total_Solution_Cost'] += cost
                results['Traditional A*'][tile_count]['Solution_Costs'].append(cost)
                results['Traditional A*'][tile_count]['Total_Expansions'] += expansions
                results['Traditional A*'][tile_count]['Expansions'].append(expansions)
                results['Traditional A*'][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
            else:
                print("Traditional A* did not find a solution within the expansion limit. Regenerating puzzle.")
                continue  # Regenerate a new puzzle

            # Weighted A* with different weights
            for w in epsilons:
                # Weighted A* with Re-Expansions Allowed
                print(f"Running Weighted A* with weight {w} (Re-Expansions Allowed)")
                path, cost, expansions, reopening_count = weighted_a_star_with_reopening_tracking(
                    start_state,
                    goal_state,
                    epsilon=w,
                    max_runtime=max_runtime,
                    grid_size=size,
                    max_expansions=max_expansions,
                    allow_reexpansions=True
                )
                if path is not None:
                    re_expansion_percent = (reopening_count / expansions) * 100 if expansions > 0 else 0.0
                    print(f"Expanded {expansions} nodes, cost is {cost}, re-expansions percent is {re_expansion_percent}%")
                    algo_name = f'Weighted A* w={w} (Re-Expansions Allowed)'
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                    results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)
                else:
                    print(f"Weighted A* with weight {w} (Re-Expansions Allowed) did not find a solution within the expansion limit.")

                # Weighted A* with Re-Expansions Not Allowed
                print(f"Running Weighted A* with weight {w} (Re-Expansions Not Allowed)")
                path, cost, expansions, reopening_count = weighted_a_star_with_reopening_tracking(
                    start_state,
                    goal_state,
                    epsilon=w,
                    max_runtime=max_runtime,
                    grid_size=size,
                    max_expansions=max_expansions,
                    allow_reexpansions=False
                )
                if path is not None:
                    re_expansion_percent = (reopening_count / expansions) * 100 if expansions > 0 else 0.0
                    print(f"Expanded {expansions} nodes, cost is {cost}, re-expansions percent is {re_expansion_percent}%")
                    algo_name = f'Weighted A* w={w} (Re-Expansions Not Allowed)'
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                    results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)
                else:
                    print(f"Weighted A* with weight {w} (Re-Expansions Not Allowed) did not find a solution within the expansion limit.")

            # Potential Search with Re-Expansions Allowed
            print("Running Potential Search (Re-Expansions Allowed)")
            path, cost, expansions, re_expansions, reshuffles = potential_search_with_reopening_tracking(
                start_state,
                goal_state,
                B=3.0,
                max_runtime=max_runtime,
                grid_size=size,
                max_expansions=max_expansions,
                allow_reexpansions=True  # Allow re-expansions
            )

            if path is not None:
                re_expansion_percent = (re_expansions / expansions) * 100 if expansions > 0 else 0.0
                print(f"Expanded {expansions} nodes, cost is {cost}, re-expansions: {re_expansion_percent}%, reshuffles: {reshuffles}")

                algo_name = 'Potential Search (Re-Expansions Allowed)'
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                results[algo_name][tile_count]['Solution_Costs'].append(cost)
                results[algo_name][tile_count]['Total_Expansions'] += expansions
                results[algo_name][tile_count]['Expansions'].append(expansions)
                results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
            else:
                print("Potential Search (Re-Expansions Allowed) did not find a solution within the expansion limit.")

            # Potential Search with Re-Expansions Not Allowed
            print("Running Potential Search (Re-Expansions Not Allowed)")
            path, cost, expansions, re_expansions, reshuffles = potential_search_with_reopening_tracking(
                start_state,
                goal_state,
                B=3.0,
                max_runtime=max_runtime,
                grid_size=size,
                max_expansions=max_expansions,
                allow_reexpansions=False  # Disallow re-expansions
            )

            if path is not None:
                re_expansion_percent = (re_expansions / expansions) * 100 if expansions > 0 else 0.0
                print(f"Expanded {expansions} nodes, cost is {cost}, re-expansions: {re_expansion_percent}%, reshuffles: {reshuffles}")

                algo_name = 'Potential Search (Re-Expansions Not Allowed)'
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                results[algo_name][tile_count]['Solution_Costs'].append(cost)
                results[algo_name][tile_count]['Total_Expansions'] += expansions
                results[algo_name][tile_count]['Expansions'].append(expansions)
                results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
            else:
                print("Potential Search (Re-Expansions Not Allowed) did not find a solution within the expansion limit.")

            # Optimistic Search with different weights
            for w in epsilons:
                # Optimistic Search with Re-Expansions Allowed
                print(f"Running Optimistic Search with weight {w} allowing re-expansions")
                path, cost, expansions = optimistic_search(
                    start_state,
                    goal_state,
                    w=w,
                    max_runtime=max_runtime,
                    grid_size=size,
                    max_expansions=max_expansions,
                    allow_reexpansions=True  # Enable re-expansions
                )
                if path is not None:
                    # Assuming Optimistic Search does not track re-openings separately
                    re_expansion_percent = 0.0  # Placeholder if not tracked
                    print(f"Expanded {expansions} nodes, cost is {cost}")
                    algo_name = f'Optimistic Search w={w} (Re-Expansions Allowed)'
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                    results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
                else:
                    print(f"Optimistic Search w={w} allowing re-expansions did not find a solution within the expansion limit.")

                # Optimistic Search with Re-Expansions Not Allowed
                print(f"Running Optimistic Search with weight {w} without re-expansions")
                path, cost, expansions = optimistic_search(
                    start_state,
                    goal_state,
                    w=w,
                    max_runtime=max_runtime,
                    grid_size=size,
                    max_expansions=max_expansions,
                    allow_reexpansions=False  # Disable re-expansions
                )
                if path is not None:
                    re_expansion_percent = 0.0  # Placeholder if not tracked
                    print(f"Expanded {expansions} nodes, cost is {cost}")
                    algo_name = f'Optimistic Search w={w} (Re-Expansions Not Allowed)'
                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                    results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
                else:
                    print(f"Optimistic Search w={w} without re-expansions did not find a solution within the expansion limit.")

            # IMHA* and SMHA* require w1 and w2
            chosen_epsilon = epsilons[0]  # You can choose a different strategy
            w2 = min(2.0, math.sqrt(chosen_epsilon))
            w1 = chosen_epsilon / w2
            print(f"Using w1={w1} and w2={w2} for IMHA* and SMHA* algorithms.")

            # Generate heuristics
            heuristics = generate_mha_heuristics(goal_state, grid_size=size, num_heuristics=4)

            # IMHA* with Re-Expansions Allowed
            print("Running Independent Multi-Heuristic A* (IMHA*) (Re-Expansions Allowed)")
            imha_path, imha_cost, imha_expansions, imha_reopenings = imha_star_with_reopening_tracking(
                start_state,
                goal_state,
                heuristics,
                w1=w1,
                w2=w2,
                max_runtime=max_runtime,
                grid_size=size,
                max_expansions=max_expansions,
                allow_reexpansions=True  # **Allow re-expansions**
            )
            if imha_path is not None:
                re_expansion_percent = (imha_reopenings / imha_expansions) * 100 if imha_expansions > 0 else 0.0
                print(f"Expanded {imha_expansions} nodes, cost is {imha_cost}, re-expansions percent is {re_expansion_percent}%")
                algo_name = 'IMHA* (Re-Expansions Allowed)'
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += imha_cost
                results[algo_name][tile_count]['Solution_Costs'].append(imha_cost)
                results[algo_name][tile_count]['Total_Expansions'] += imha_expansions
                results[algo_name][tile_count]['Expansions'].append(imha_expansions)
                results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
            else:
                print("IMHA* (Re-Expansions Allowed) did not find a solution within the expansion limit.")

            # IMHA* with Re-Expansions Not Allowed
            print("Running Independent Multi-Heuristic A* (IMHA*) (Re-Expansions Not Allowed)")
            imha_path, imha_cost, imha_expansions, imha_reopenings = imha_star_with_reopening_tracking(
                start_state,
                goal_state,
                heuristics,
                w1=w1,
                w2=w2,
                max_runtime=max_runtime,
                grid_size=size,
                max_expansions=max_expansions,
                allow_reexpansions=False  # **Disallow re-expansions**
            )
            if imha_path is not None:
                re_expansion_percent = (imha_reopenings / imha_expansions) * 100 if imha_expansions > 0 else 0.0
                print(f"Expanded {imha_expansions} nodes, cost is {imha_cost}, re-expansions percent is {re_expansion_percent}%")
                algo_name = 'IMHA* (Re-Expansions Not Allowed)'
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += imha_cost
                results[algo_name][tile_count]['Solution_Costs'].append(imha_cost)
                results[algo_name][tile_count]['Total_Expansions'] += imha_expansions
                results[algo_name][tile_count]['Expansions'].append(imha_expansions)
                results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
            else:
                print("IMHA* (Re-Expansions Not Allowed) did not find a solution within the expansion limit.")

            # SMHA*
            print("Running Shared Multi-Heuristic A* (SMHA*)")
            smha_path, smha_cost, smha_expansions = smha_star(
                start_state,
                goal_state,
                heuristics,
                w1=w1,
                w2=w2,
                max_runtime=max_runtime,
                grid_size=size,
                max_expansions=max_expansions
            )
            if smha_path is not None:
                re_expansion_percent = 0
                print(f"Expanded {smha_expansions} nodes, cost is {smha_cost}, re-expansions percent is {re_expansion_percent}%")
                algo_name = 'SMHA*'
                results[algo_name][tile_count]['Puzzles_Solved'] += 1
                results[algo_name][tile_count]['Total_Solution_Cost'] += smha_cost
                results[algo_name][tile_count]['Solution_Costs'].append(smha_cost)
                results[algo_name][tile_count]['Total_Expansions'] += smha_expansions
                results[algo_name][tile_count]['Expansions'].append(smha_expansions)
                results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
            else:
                print("SMHA* did not find a solution within the expansion limit.")

            # Model-based A* and Focal A*
            for model_name, model_data in models.items():
                model = model_data['model']
                norm_values = normalization_values[model_name]

                # Regular A* with model
                print(f"\nRunning A* with model {model_name}")
                path, cost, expansions, reopening_count = astar_with_model(
                    start_state,
                    goal_state,
                    model,
                    norm_values,
                    device,
                    max_runtime=max_runtime,
                    grid_size=size,
                    max_expansions=max_expansions,
                    allow_reexpansions=True  # Allow re-expansions
                )
                algo_name = f'{model_name} A*'
                if path is not None:
                    re_expansion_percent = (reopening_count / expansions) * 100 if expansions > 0 else 0.0
                    print(f"Expanded {expansions} nodes, cost is {cost}, re-expansions is {re_expansion_percent}%")

                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                    results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
                else:
                    print(f"A* with model {model_name} did not find a solution within the expansion limit.")

                # Focal A* with model
                epsilon = 3.0  
                print(f"Running Focal A* with model {model_name} and epsilon {epsilon}")
                path, cost, expansions, reopening_count = focal_astar_with_model(
                    start_state,
                    goal_state,
                    model,
                    norm_values,
                    device,
                    epsilon=epsilon,
                    max_runtime=max_runtime,
                    grid_size=size,
                    max_expansions=max_expansions,
                    allow_reexpansions=True  # Allow re-expansions
                )
                algo_name = f'{model_name} Focal A*'
                if path is not None:
                    re_expansion_percent = (reopening_count / expansions) * 100 if expansions > 0 else 0.0
                    print(f"Expanded {expansions} nodes, cost is {cost}, re-expansions is {re_expansion_percent}%")

                    results[algo_name][tile_count]['Puzzles_Solved'] += 1
                    results[algo_name][tile_count]['Total_Solution_Cost'] += cost
                    results[algo_name][tile_count]['Solution_Costs'].append(cost)
                    results[algo_name][tile_count]['Total_Expansions'] += expansions
                    results[algo_name][tile_count]['Expansions'].append(expansions)
                    results[algo_name][tile_count]['ReExpansions_Percent'].append(re_expansion_percent)  # Store percentage
                else:
                    print(f"Focal A* with model {model_name} did not find a solution within the expansion limit.")

    # ---------------------------
    # Save the stats to CSV (Modified for Re-Expansion Percentages and Success Rates)
    # ---------------------------

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
            'Expansion_Percent_Difference',
            'Mean_ReExpansions_Percent'  # Changed from ReExpansions fields
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for algo_name, data in results.items():
            for tile_count, metrics in data.items():
                if metrics['Puzzles_Solved'] > 0:
                    avg_solution_cost = metrics['Total_Solution_Cost'] / metrics['Puzzles_Solved']
                    avg_expansions = metrics['Total_Expansions'] / metrics['Puzzles_Solved']
                    if metrics['ReExpansions_Percent']:
                        mean_re_expansions_percent = sum(metrics['ReExpansions_Percent']) / len(metrics['ReExpansions_Percent'])
                    else:
                        mean_re_expansions_percent = 0.0
                else:
                    avg_solution_cost = 0
                    avg_expansions = 0
                    mean_re_expansions_percent = 0.0
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
                    mean_re_expansions_percent_diff = 0.00
                else:
                    if a_star_avg_cost > 0:
                        solution_cost_percent_diff = ((avg_solution_cost - a_star_avg_cost) / a_star_avg_cost) * 100
                    else:
                        solution_cost_percent_diff = 0.00
                    if a_star_avg_expansions > 0:
                        expansion_percent_diff = ((avg_expansions - a_star_avg_expansions) / a_star_avg_expansions) * 100
                    else:
                        expansion_percent_diff = 0.00
                    # For re-expansions, calculate mean percent difference compared to Traditional A* which has 0%
                    mean_re_expansions_percent_diff = mean_re_expansions_percent  # Since Traditional A* has 0 re-expansions
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
                    'Expansion_Percent_Difference': f"{round(expansion_percent_diff, 2)}%",
                    'Mean_ReExpansions_Percent': f"{round(mean_re_expansions_percent_diff, 2)}%"  # Changed Field
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
    parser.add_argument("--min_expansions", type=int, default=300, help="Minimum number of node expansions required for A*")
    parser.add_argument("--max_expansions", type=int, default=10000, help="Maximum number of node expansions allowed for A*")
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