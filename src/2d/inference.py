import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
import matplotlib.pyplot as plt
import heapq
import os
import csv
from models import UNet2DAutoencoder, MLPModel
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import random
import itertools
from dataclasses import dataclass, field

# ---------------------------
# Map Generation and Utility Functions
# ---------------------------

@dataclass
class Room:
    x: int
    y: int
    width: int
    height: int
    children: list = field(default_factory=list)

# def generate_map(
#     width=128,
#     height=128,
#     min_room_size=20,
#     max_room_size=40,
#     max_depth=5,
#     wall_thickness=2,
#     min_openings=1,
#     max_openings=2,
#     opening_size=4,
#     min_obstacles=5,
#     max_obstacles=8,
#     min_obstacle_size=7,
#     max_obstacle_size=10
# ):
#     """
#     Generates a 2D map with randomly sized and positioned rooms separated by walls with random openings.
#     Each room contains a random number of smaller obstacles.
#     """
#     map_grid = np.zeros((height, width), dtype=np.float32)
#     return map_grid
    
#     # Initialize the root room
#     root_room = Room(0, 0, width, height)
    
#     def split_room(room, depth):
#         if depth >= max_depth:
#             return
#         # Check if the room is large enough to split
#         can_split_horizontally = room.height >= 2 * min_room_size + wall_thickness
#         can_split_vertically = room.width >= 2 * min_room_size + wall_thickness
        
#         if not can_split_horizontally and not can_split_vertically:
#             return  # Cannot split further
        
#         # Decide split orientation based on room size and randomness
#         if can_split_horizontally and can_split_vertically:
#             split_horizontally = random.choice([True, False])
#         elif can_split_horizontally:
#             split_horizontally = True
#         else:
#             split_horizontally = False
        
#         if split_horizontally:
#             # Choose a split position
#             split_min = room.y + min_room_size
#             split_max = room.y + room.height - min_room_size - wall_thickness
#             if split_max <= split_min:
#                 return  # Not enough space to split
#             split_pos = random.randint(split_min, split_max)
#             # Create child rooms
#             child1 = Room(room.x, room.y, room.width, split_pos - room.y)
#             child2 = Room(room.x, split_pos + wall_thickness, room.width, room.y + room.height - split_pos - wall_thickness)
#             # Add horizontal wall
#             map_grid[split_pos:split_pos + wall_thickness, room.x:room.x + room.width] = 1
#             # Add openings
#             add_openings((split_pos, room.x), (split_pos, room.x + room.width), orientation='horizontal')
#         else:
#             # Vertical split
#             split_min = room.x + min_room_size
#             split_max = room.x + room.width - min_room_size - wall_thickness
#             if split_max <= split_min:
#                 return  # Not enough space to split
#             split_pos = random.randint(split_min, split_max)
#             # Create child rooms
#             child1 = Room(room.x, room.y, split_pos - room.x, room.height)
#             child2 = Room(split_pos + wall_thickness, room.y, room.x + room.width - split_pos - wall_thickness, room.height)
#             # Add vertical wall
#             map_grid[room.y:room.y + room.height, split_pos:split_pos + wall_thickness] = 1
#             # Add openings
#             add_openings((room.y, split_pos), (room.y + room.height, split_pos), orientation='vertical')
        
#         room.children = [child1, child2]
#         # Recursively split the child rooms
#         split_room(child1, depth + 1)
#         split_room(child2, depth + 1)
    
#     def add_openings(start, end, orientation='horizontal'):
#         """
#         Adds random openings to a wall.
#         """
#         num_openings = random.randint(min_openings, max_openings)
#         if orientation == 'horizontal':
#             wall_length = end[1] - start[1]
#             possible_positions = wall_length - opening_size
#             if possible_positions <= 0:
#                 return
#             for _ in range(num_openings):
#                 opening_start = random.randint(start[1], start[1] + possible_positions)
#                 map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
#         else:
#             wall_length = end[0] - start[0]
#             possible_positions = wall_length - opening_size
#             if possible_positions <= 0:
#                 return
#             for _ in range(num_openings):
#                 opening_start = random.randint(start[0], start[0] + possible_positions)
#                 map_grid[opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness] = 0
    
#     # Start splitting from the root room
#     split_room(root_room, 0)
    
#     # Collect all leaf rooms
#     leaf_rooms = []
#     def collect_leaf_rooms(room):
#         if not room.children:
#             leaf_rooms.append(room)
#         else:
#             for child in room.children:
#                 collect_leaf_rooms(child)
    
#     collect_leaf_rooms(root_room)
    
#     # Add obstacles to each leaf room
#     for room in leaf_rooms:
#         num_obstacles = random.randint(min_obstacles, max_obstacles)
#         for _ in range(num_obstacles):
#             obstacle_w = random.randint(min_obstacle_size, max_obstacle_size)
#             obstacle_h = random.randint(min_obstacle_size, max_obstacle_size)
#             # Ensure obstacle fits within the room with some padding
#             if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
#                 continue  # Skip if obstacle is too big for the room
#             obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
#             obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
#             # Avoid placing obstacles on walls
#             map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 1
    
#     # Optionally, add outer boundary walls
#     # Top and bottom
#     map_grid[0:wall_thickness, :] = 1
#     map_grid[-wall_thickness:, :] = 1
#     # Left and right
#     map_grid[:, 0:wall_thickness] = 1
#     map_grid[:, -wall_thickness:] = 1
    
#     return map_grid

def generate_map(width=128, height=128, block_size=25, num_blocks=6):
    """
    Generates a 2D map with a random scattering of num_blocks 25x25 obstacle blocks.
    """
    map_grid = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_blocks):
        # Randomly select the top-left corner for the block
        block_x = np.random.randint(0, width - block_size)
        block_y = np.random.randint(0, height - block_size)

        # Fill the area of block_size x block_size with obstacles (1s)
        map_grid[block_y:block_y + block_size, block_x:block_x + block_size] = 1

    return map_grid


def is_valid(pos, map_grid):
    return 0 <= pos[0] < map_grid.shape[0] and 0 <= pos[1] < map_grid.shape[1] and map_grid[pos] == 0

def generate_start_goal(map_grid, min_distance_ratio=0):
    """
    Generates start and goal positions on the map that are valid and far away from each other.
    :param map_grid: 2D map grid (numpy array)
    :param min_distance_ratio: Minimum distance ratio as a fraction of the map's diagonal
    :return: Tuple (start, goal)
    """
    diagonal_distance = np.sqrt(map_grid.shape[0]**2 + map_grid.shape[1]**2)
    min_distance = diagonal_distance * min_distance_ratio

    while True:
        start = (np.random.randint(0, map_grid.shape[0]), np.random.randint(0, map_grid.shape[1]))
        goal = (np.random.randint(0, map_grid.shape[0]), np.random.randint(0, map_grid.shape[1]))
        distance = euclidean_distance(start, goal)
        
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal and distance >= min_distance:
            return start, goal


def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(pos):
    x, y = pos
    neighbors = [
        ((x-1, y), 1),   # Up
        ((x+1, y), 1),   # Down
        ((x, y-1), 1),   # Left
        ((x, y+1), 1),   # Right
        ((x-1, y-1), np.sqrt(2)),  # Up-Left
        ((x-1, y+1), np.sqrt(2)),  # Up-Right
        ((x+1, y-1), np.sqrt(2)),  # Down-Left
        ((x+1, y+1), np.sqrt(2))   # Down-Right
    ]
    return neighbors

# ---------------------------
# Visualization Function
# ---------------------------

def visualize_comparison(map_grid, start, goal, f_star_map1, f_star_map2, f_map, model1_path, model2_path, traditional_path, output_dir="output", run=0):
    plt.figure(figsize=(18, 6))

    # Create custom colormap for the map_grid
    cmap = ListedColormap(['white', 'black'])

    # Model 1-based A* plot
    plt.subplot(1, 3, 1)
    plt.title("Model 1 Neural Priority Function")
    plt.imshow(map_grid, cmap=cmap, interpolation='nearest')
    valid_positions = np.argwhere(np.isfinite(f_star_map1))
    scatter = plt.scatter(valid_positions[:, 1], valid_positions[:, 0],
                          c=f_star_map1[valid_positions[:, 0], valid_positions[:, 1]],
                          cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(scatter, label="f* values (Model 1)")
    plt.plot(start[1], start[0], 'go', markersize=10)  # Removed label='Start'
    plt.plot(goal[1], goal[0], 'ro', markersize=10)    # Removed label='Goal'
    plt.plot([p[1] for p in model1_path], [p[0] for p in model1_path], 'b-', linewidth=2)
    # Removed plt.legend()

    # Model 2-based A* plot
    plt.subplot(1, 3, 2)
    plt.title("Model 2 Neural Priority Function")
    plt.imshow(map_grid, cmap=cmap, interpolation='nearest')
    valid_positions = np.argwhere(np.isfinite(f_star_map2))
    scatter = plt.scatter(valid_positions[:, 1], valid_positions[:, 0],
                          c=f_star_map2[valid_positions[:, 0], valid_positions[:, 1]],
                          cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(scatter, label="f* values (Model 2)")
    plt.plot(start[1], start[0], 'go', markersize=10)  # Removed label='Start'
    plt.plot(goal[1], goal[0], 'ro', markersize=10)    # Removed label='Goal'
    plt.plot([p[1] for p in model2_path], [p[0] for p in model2_path], 'b-', linewidth=2)
    # Removed plt.legend()

    # Traditional A* plot
    plt.subplot(1, 3, 3)
    plt.title("Traditional A* f = g + h")
    plt.imshow(map_grid, cmap=cmap, interpolation='nearest')
    valid_positions = np.argwhere(np.isfinite(f_map))
    scatter = plt.scatter(valid_positions[:, 1], valid_positions[:, 0],
                          c=f_map[valid_positions[:, 0], valid_positions[:, 1]],
                          cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(scatter, label="f values (Traditional A*)")
    plt.plot(start[1], start[0], 'go', markersize=10)  # Removed label='Start'
    plt.plot(goal[1], goal[0], 'ro', markersize=10)    # Removed label='Goal'
    plt.plot([p[1] for p in traditional_path], [p[0] for p in traditional_path], 'b-', linewidth=2)
    # Removed plt.legend()

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'comparison_run_{run}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Normalization Functions
# ---------------------------

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# ---------------------------
# Inference Function
# ---------------------------

def run_inference(map_grid, start, goal, current, g, encoder, model, normalization_values, device):
    f_star_min = normalization_values['f_star_min']
    f_star_max = normalization_values['f_star_max']
    g_min = normalization_values['g_min']
    g_max = normalization_values['g_max']
    h_min = normalization_values['h_min']
    h_max = normalization_values['h_max']

    with torch.no_grad():
        map_tensor = torch.from_numpy(map_grid).float().unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]
        encoded_map = encoder.encode(map_tensor).cpu().numpy().flatten()  # Should be 512-dimensional

    h = euclidean_distance(current, goal)

    g_normalized = normalize(g, g_min, g_max)
    h_normalized = normalize(h, h_min, h_max)

    start_normalized = np.array(start) / 127.0
    goal_normalized = np.array(goal) / 127.0
    current_normalized = np.array(current) / 127.0
    input_tensor = np.concatenate([
        start_normalized,             # 2 values
        goal_normalized,              # 2 values
        current_normalized,           # 2 values
        [g_normalized, h_normalized], # 2 values
        encoded_map                   # 512 values
    ])  # Total: 520

    input_tensor = torch.from_numpy(input_tensor).float().to(device)  # Shape: [520]
    
    # Ensure input_tensor has shape [1, 520]
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, 520]

    model.eval()
    with torch.no_grad():
        f_star_predicted, _ = model(input_tensor)
    f_star_denormalized = denormalize(f_star_predicted.item(), f_star_min, f_star_max)

    return f_star_denormalized

# ---------------------------
# ModelNode Class for Model-based A*
# ---------------------------

class ModelNode:
    def __init__(self, pos, f_star, g, parent=None, h=None):
        self.pos = pos
        self.f_star = f_star
        self.g = g
        self.parent = parent
        self.h = h 

    def __lt__(self, other):
        return self.f_star < other.f_star


# ---------------------------
# A* Search with Neural Model's f* Values
# ---------------------------

def astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, max_expansions=100000):
    open_list = []
    closed_set = set()
    g_score = {start: 0}
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_node = ModelNode(start, f_star=start_f_star, g=0)
    heapq.heappush(open_list, start_node)
    expansions = 0

    # Initialize f_star_map with NaNs for visualization
    f_star_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_star_map

        if current.pos == goal:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g  # Total cost from start to goal
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_star_map

        closed_set.add(current.pos)

        # Record f_star value for visualization
        f_star_map[current.pos[0], current.pos[1]] = current.f_star

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue  # Not a better path

            g_score[next_pos] = tentative_g
            f_star = run_inference(map_grid, start, goal, next_pos, tentative_g, encoder, model, normalization_values, device)
            next_node = ModelNode(next_pos, f_star=f_star, g=tentative_g, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions, f_star_map  # Return None values if no path found

def focal_astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, epsilon=1.5, max_expansions=100000):
    import bisect

    class OpenList:
        def __init__(self):
            self.elements = []
            self.entry_finder = {}
            self.counter = itertools.count()

        def add_node(self, node):
            f = node.g + node.h
            if node.pos in self.entry_finder:
                existing_count, existing_node = self.entry_finder[node.pos]
                if node.g < existing_node.g:
                    self.remove_node(existing_node)
                else:
                    return
            count_value = next(self.counter)
            bisect.insort_left(self.elements, (f, count_value, node))
            self.entry_finder[node.pos] = (count_value, node)

        def remove_node(self, node):
            if node.pos in self.entry_finder:
                count_value, _ = self.entry_finder[node.pos]
                f = node.g + node.h
                idx = bisect.bisect_left(self.elements, (f, count_value, node))
                if idx < len(self.elements) and self.elements[idx][2].pos == node.pos:
                    self.elements.pop(idx)
                    del self.entry_finder[node.pos]

        def get_f_min(self):
            if self.elements:
                return self.elements[0][0]
            else:
                return float('inf')

        def get_nodes_within_epsilon(self, f_min, epsilon):
            idx = bisect.bisect_right(self.elements, (f_min * epsilon, float('inf'), None))
            return [node for f, count_value, node in self.elements[:idx]]

        def is_empty(self):
            return not self.elements

    open_list = OpenList()
    closed_set = set()
    g_score = {start: 0}
    h_start = euclidean_distance(start, goal)
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_node = ModelNode(start, f_star=start_f_star, g=0, h=h_start)
    open_list.add_node(start_node)
    expansions = 0

    f_star_map = np.full(map_grid.shape, np.nan)

    while not open_list.is_empty():
        current_f_min = open_list.get_f_min()
        focal_list = open_list.get_nodes_within_epsilon(current_f_min, epsilon)

        if not focal_list:
            return None, None, expansions, f_star_map

        current = min(focal_list, key=lambda node: node.f_star)
        open_list.remove_node(current)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_star_map

        if current.pos == goal:
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_star_map

        closed_set.add(current.pos)
        f_star_map[current.pos[0], current.pos[1]] = current.f_star

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue

            g_score[next_pos] = tentative_g
            h = euclidean_distance(next_pos, goal)
            f_star = run_inference(map_grid, start, goal, next_pos, tentative_g, encoder, model, normalization_values, device)
            next_node = ModelNode(next_pos, f_star=f_star, g=tentative_g, h=h, parent=current)
            open_list.add_node(next_node)

    return None, None, expansions, f_star_map

def generate_mha_heuristics():
    return [
        lambda pos, goal: euclidean_distance(pos, goal),
        lambda pos, goal: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]),
        lambda pos, goal: max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1])),
        lambda pos, goal: 0.5 * euclidean_distance(pos, goal),  # Scaled heuristic
    ]

class MHAStarNode:
    def __init__(self, pos, g, h_values, parent=None):
        self.pos = pos
        self.g = g
        self.h_values = h_values
        self.parent = parent
        self.f = min([g + h for h in h_values])

    def __lt__(self, other):
        return self.f < other.f

def multi_heuristic_astar(start, goal, map_grid, heuristic_functions, max_expansions=100000):
    open_list = []
    closed_set = set()
    g_score = {start: 0}
    h_values_start = [h(start, goal) for h in heuristic_functions]
    start_node = MHAStarNode(start, g=0, h_values=h_values_start)
    heapq.heappush(open_list, start_node)
    expansions = 0

    f_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_map

        if current.pos == goal:
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_map

        closed_set.add(current.pos)
        f_map[current.pos[0], current.pos[1]] = current.f

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue

            g_score[next_pos] = tentative_g
            h_values = [h(next_pos, goal) for h in heuristic_functions]
            next_node = MHAStarNode(next_pos, g=tentative_g, h_values=h_values, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions, f_map

def potential_search(start, goal, map_grid, C, max_expansions=100000):
    class PotentialNode:
        def __init__(self, pos, g, h, flnr, parent=None):
            self.pos = pos
            self.g = g
            self.h = h
            self.flnr = flnr
            self.parent = parent

        def __lt__(self, other):
            return self.flnr < other.flnr

    open_list = []
    closed_set = set()
    g_score = {}
    h_start = euclidean_distance(start, goal)
    g_start = 0

    if g_start > C:
        return None, None, 0, None

    flnr_start = h_start / max(C - g_start, 1e-8)
    start_node = PotentialNode(pos=start, g=g_start, h=h_start, flnr=flnr_start)
    heapq.heappush(open_list, start_node)
    g_score[start] = g_start
    expansions = 0

    flnr_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, flnr_map

        if current.pos == goal and current.g <= C:
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, flnr_map

        closed_set.add(current.pos)
        flnr_map[current.pos[0], current.pos[1]] = current.flnr

        for next_pos, cost in get_neighbors(current.pos):
            tentative_g = current.g + cost

            if tentative_g > C or not is_valid(next_pos, map_grid):
                continue

            if next_pos in closed_set and tentative_g >= g_score.get(next_pos, float('inf')):
                continue

            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                h = euclidean_distance(next_pos, goal)
                if C - tentative_g > 0:
                    flnr = h / (C - tentative_g)
                else:
                    flnr = float('inf')  # Avoid division by zero or negative numbers
                next_node = PotentialNode(pos=next_pos, g=tentative_g, h=h, flnr=flnr, parent=current)
                heapq.heappush(open_list, next_node)

    return None, None, expansions, flnr_map


def astar_weighted(start, goal, map_grid, epsilon=1.5, max_expansions=100000):
    class WeightedAStarNode:
        def __init__(self, pos, g, h, parent=None):
            self.pos = pos
            self.g = g
            self.h = h
            self.f = g + epsilon * h
            self.parent = parent

        def __lt__(self, other):
            return self.f < other.f

    open_list = []
    closed_set = set()
    g_score = {start: 0}
    h_start = euclidean_distance(start, goal)
    start_node = WeightedAStarNode(start, g=0, h=h_start, parent=None)
    heapq.heappush(open_list, start_node)
    expansions = 0

    f_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_map

        if current.pos == goal:
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_map

        closed_set.add(current.pos)
        f_map[current.pos[0], current.pos[1]] = current.f

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue

            g_score[next_pos] = tentative_g
            h = euclidean_distance(next_pos, goal)
            next_node = WeightedAStarNode(next_pos, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions, f_map


# ---------------------------
# A* Search with Traditional f = g + h
# ---------------------------

class AStarNode:
    def __init__(self, pos, g, h, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def astar_traditional(start, goal, map_grid, max_expansions=100000):
    open_list = []
    closed_set = set()
    g_score = {start: 0}
    start_node = AStarNode(start, g=0, h=euclidean_distance(start, goal))
    heapq.heappush(open_list, start_node)
    expansions = 0

    # Initialize f_map with NaNs for visualization
    f_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_map  # Return None values if max expansions exceeded

        if current.pos == goal:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g  # Total cost from start to goal
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_map

        closed_set.add(current.pos)

        # Record f-value for visualization
        f_map[current.pos[0], current.pos[1]] = current.f

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue  # Not a better path

            g_score[next_pos] = tentative_g
            h = euclidean_distance(next_pos, goal)
            next_node = AStarNode(next_pos, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions, f_map  # Return None values if no path found

# ---------------------------
# Assessment Function
# ---------------------------

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap

def run_assessment(encoder, models, normalization_values, device, num_maps=1, num_queries_per_map=10, output_csv="output.csv", output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    algorithms = ['Traditional A*', 'Weighted A* ε=1.5', 'Weighted A* ε=3.0', 'Potential Search', 'Multi-Heuristic A*']
    # Add model-based algorithms
    for model_name in models:
        algorithms.append(f'{model_name} A*')
        algorithms.append(f'{model_name} Focal A* ε=1.5')

    stats = {algo: {
        'total_expansions_reduction': 0,
        'total_path_cost_increase': 0,
        'path_cost_increases': [],
        'optimal_paths': 0,
        'total_queries': 0
    } for algo in algorithms}

    query_counter = 0
    for map_idx in range(num_maps):
        map_data = generate_map()
        print(f"\nRunning assessment on generated map {map_idx + 1}\n")

        for query in range(num_queries_per_map):
            start, goal = generate_start_goal(map_data)

            # Traditional A*
            print(f"\n\n\nRunning Traditional A* on query {query + 1}...")
            traditional_path, traditional_path_cost, traditional_expanded, f_map = astar_traditional(start, goal, map_data)

            if traditional_path_cost is None:
                print(f"No path found for Traditional A* on query {query + 1}. Skipping...\n")
                continue

            print(f"Traditional A* expansions: {traditional_expanded}, cost: {traditional_path_cost}")

            # # Save visualization
            # plt.figure(figsize=(10, 10))
            # plt.imshow(map_data, cmap=ListedColormap(['white', 'black']))
            # plt.title("A* (f = g + h)")
            # valid_positions = np.argwhere(np.isfinite(f_map))
            # scatter = plt.scatter(valid_positions[:, 1], valid_positions[:, 0],
            #                       c=f_map[valid_positions[:, 0], valid_positions[:, 1]],
            #                       cmap='viridis', s=20, alpha=0.8)
            # plt.colorbar(scatter, label="f values")
            # plt.plot([p[1] for p in traditional_path], [p[0] for p in traditional_path], 'b-', linewidth=2)
            # plt.savefig(os.path.join(output_dir, f'traditional_astar_map_{map_idx}_query_{query_counter}.png'), dpi=300, bbox_inches='tight')
            # plt.close()

            # Weighted A* ε=1.5 and ε=3.0
            for epsilon in [1.5, 3.0]:
                algo_name = f'Weighted A* ε={epsilon}'
                print(f"Running {algo_name} on query {query + 1}...")
                path, path_cost, expanded, f_map_weighted = astar_weighted(start, goal, map_data, epsilon=epsilon)

                if path_cost is None:
                    print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                

                print(f"{algo_name} expansions: {expanded}, cost: {path_cost}")
                expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

            # Potential Search
            print(f"Running Potential Search on query {query + 1}...")
            C = traditional_path_cost*1.5
            algo_name = 'Potential Search'
            path, path_cost, expanded, flnr_map = potential_search(start, goal, map_data, C=C)

            if path_cost is None:
                print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
            else:

                print(f"Potential Search expansions: {expanded}, cost: {path_cost}")
                expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(path_cost - traditional_path_cost) < 1:
                    stats[algo_name]['optimal_paths'] += 1
            stats[algo_name]['total_queries'] += 1

            # Multi-Heuristic A*
            print(f"Running Multi-Heuristic A* on query {query + 1}...")
            heuristic_functions = generate_mha_heuristics()
            algo_name = 'Multi-Heuristic A*'
            path, path_cost, expanded, f_map_mha = multi_heuristic_astar(start, goal, map_data, heuristic_functions)

            if path_cost is None:
                print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                

            print(f"Multi-Heuristic A* expansions: {expanded}, cost: {path_cost}")
            expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
            cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

            stats[algo_name]['total_expansions_reduction'] += expansions_diff
            stats[algo_name]['total_path_cost_increase'] += cost_diff
            stats[algo_name]['path_cost_increases'].append(cost_diff)
            if abs(path_cost - traditional_path_cost) < 1e-6:
                stats[algo_name]['optimal_paths'] += 1
            stats[algo_name]['total_queries'] += 1

            # Model-based A* and Focal A*
            for model_name, model in models.items():
                # Model-based A*
                algo_name = f'{model_name} A*'
                print(f"Running {algo_name} on query {query + 1}...")
                model_path, model_path_cost, model_expanded, f_star_map = astar_with_model(start, goal, map_data, encoder, model, normalization_values, device)
                
                if model_path_cost is None:
                    print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                    continue

                print(f"{algo_name} expansions: {model_expanded}, cost: {model_path_cost}")
                expansions_diff = 100 * (traditional_expanded - model_expanded) / traditional_expanded
                cost_diff = 100 * (model_path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(model_path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

                # Focal A* with ε=1.5
                algo_name = f'{model_name} Focal A* ε=1.5'
                print(f"Running {algo_name} on query {query + 1}...")
                focal_path, focal_path_cost, focal_expanded, f_star_map_focal = focal_astar_with_model(start, goal, map_data, encoder, model, normalization_values, device, epsilon=1.5)
                
                if focal_path_cost is None:
                    print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                    continue

                print(f"{algo_name} expansions: {focal_expanded}, cost: {focal_path_cost}")
                expansions_diff = 100 * (traditional_expanded - focal_expanded) / traditional_expanded
                cost_diff = 100 * (focal_path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(focal_path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

            query_counter += 1

    # Save stats to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Avg_Expansions_Reduction', 'Avg_Path_Cost_Increase', 'Std_Path_Cost_Increase', 'Optimal_Paths', 'Total_Queries']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for algo_name, data in stats.items():
            if data['total_queries'] > 0:
                avg_exp_reduction = data['total_expansions_reduction'] / data['total_queries']
                avg_cost_increase = data['total_path_cost_increase'] / data['total_queries']
                std_cost_increase = np.std(data['path_cost_increases'])
                writer.writerow({
                    'Algorithm': algo_name,
                    'Avg_Expansions_Reduction': avg_exp_reduction,
                    'Avg_Path_Cost_Increase': avg_cost_increase,
                    'Std_Path_Cost_Increase': std_cost_increase,
                    'Optimal_Paths': data['optimal_paths'],
                    'Total_Queries': data['total_queries']
                })
    print(f"Assessment results saved to {output_csv}")


# ---------------------------
# Main Execution
# ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run A* with multiple neural network models and traditional A* for comparison.")
    parser.add_argument("encoder_path", type=str, help="Path to the pre-trained encoder model")
    parser.add_argument("model_paths_or_dir", type=str, nargs='+', help="Paths to the pre-trained MLPModels or a directory containing them")
    parser.add_argument("normalization_values_path", type=str, help="Path to the normalization values pickle file")
    parser.add_argument("--num_maps", type=int, default=10, help="Number of maps to generate")
    parser.add_argument("--num_queries_per_map", type=int, default=2, help="Number of queries to run for each map")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="Path to the output CSV file")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Encoder with latent_dim=512
    encoder = UNet2DAutoencoder(input_channels=1, latent_dim=512).to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    encoder.eval()

    # Initialize Models with appropriate input_size
    models = {}
    input_size = (3 * 2) + 512 + 2  # start, goal, current (3 positions * 2 coordinates) + latent_dim=512 + 2 heuristic values
    model_paths = []

    for path in args.model_paths_or_dir:
        if os.path.isdir(path):
            # List all files in the directory that match model files
            for file_name in os.listdir(path):
                if file_name.endswith('.pth') or file_name.endswith('.pt'):
                    model_paths.append(os.path.join(path, file_name))
        else:
            # Assume it's a model file
            model_paths.append(path)

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        model = MLPModel(input_size=input_size, output_size=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[model_name] = model

    # Load Normalization Values
    with open(args.normalization_values_path, 'rb') as f:
        normalization_values = pickle.load(f)

    # Run Assessment
    run_assessment(encoder, models, normalization_values, device,
                   num_maps=args.num_maps, num_queries_per_map=args.num_queries_per_map,
                   output_csv=args.output_csv, output_dir=args.output_dir)
