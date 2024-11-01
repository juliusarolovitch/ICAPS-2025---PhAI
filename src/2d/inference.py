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
from scipy.ndimage import binary_dilation

# ---------------------------
# Neural Network Models
# ---------------------------

# Assuming UNet2DAutoencoder and MLPModel are defined in 'models.py'

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

def generate_map(
    width=512,  # Reduced from 1600 to 512
    height=512,  # Reduced from 1600 to 512
    min_room_size=60,  # Adjusted for smaller maps
    max_room_size=120,  # Adjusted for smaller maps
    max_depth=5,  # Reduced depth for smaller maps
    wall_thickness=5,
    min_openings=1,
    max_openings=2,
    min_opening_size=10,  # Adjusted for smaller maps
    max_opening_size=20,  # Adjusted for smaller maps
    min_obstacles=4,
    max_obstacles=20,
    min_obstacle_size=10,
    max_obstacle_size=30,
    obstacle_attempts=10,
    trap_probability=0.4
):
    """
    Generates a 2D map with rooms and walls with openings.
    Adds rectangular obstacles and concave traps without overlapping.

    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        min_room_size (int): Minimum size of a room.
        max_room_size (int): Maximum size of a room.
        max_depth (int): Maximum recursion depth for splitting rooms.
        wall_thickness (int): Thickness of the walls between rooms.
        min_openings (int): Minimum number of openings per wall.
        max_openings (int): Maximum number of openings per wall.
        min_opening_size (int): Minimum size of each opening in pixels.
        max_opening_size (int): Maximum size of each opening in pixels.
        min_obstacles (int): Minimum number of obstacles per room.
        max_obstacles (int): Maximum number of obstacles per room.
        min_obstacle_size (int): Minimum size (width/height) of each rectangular obstacle.
        max_obstacle_size (int): Maximum size (width/height) of each rectangular obstacle.
        obstacle_attempts (int): Number of attempts to place an obstacle without overlap.
        trap_probability (float): Probability of placing a concave trap instead of a regular obstacle.

    Returns:
        np.ndarray: 2D occupancy map of shape [height, width].
    """
    map_grid = np.zeros((height, width), dtype=np.float32)
    
    # Initialize the root room
    root_room = Room(0, 0, width, height)
    
    def split_room(room, depth):
        if depth >= max_depth:
            return
        # Check if the room is large enough to split
        can_split_horizontally = room.height >= 2 * min_room_size + wall_thickness
        can_split_vertically = room.width >= 2 * min_room_size + wall_thickness
        
        if not can_split_horizontally and not can_split_vertically:
            return  # Cannot split further
        
        # Decide split orientation based on room size and randomness
        if can_split_horizontally and can_split_vertically:
            split_horizontally = random.choice([True, False])
        elif can_split_horizontally:
            split_horizontally = True
        else:
            split_horizontally = False
        
        if split_horizontally:
            # Choose a split position
            split_min = room.y + min_room_size
            split_max = room.y + room.height - min_room_size - wall_thickness
            if split_max <= split_min:
                return  # Not enough space to split
            split_pos = random.randint(split_min, split_max)
            # Create child rooms
            child1 = Room(room.x, room.y, room.width, split_pos - room.y)
            child2 = Room(room.x, split_pos + wall_thickness, room.width, room.y + room.height - split_pos - wall_thickness)
            # Add horizontal wall
            map_grid[split_pos:split_pos + wall_thickness, room.x:room.x + room.width] = 1
            # Add openings
            add_openings((split_pos, room.x), (split_pos, room.x + room.width), orientation='horizontal')
        else:
            # Vertical split
            split_min = room.x + min_room_size
            split_max = room.x + room.width - min_room_size - wall_thickness
            if split_max <= split_min:
                return  # Not enough space to split
            split_pos = random.randint(split_min, split_max)
            # Create child rooms
            child1 = Room(room.x, room.y, split_pos - room.x, room.height)
            child2 = Room(split_pos + wall_thickness, room.y, room.x + room.width - split_pos - wall_thickness, room.height)
            # Add vertical wall
            map_grid[room.y:room.y + room.height, split_pos:split_pos + wall_thickness] = 1
            # Add openings
            add_openings((room.y, split_pos), (room.y + room.height, split_pos), orientation='vertical')
        
        room.children = [child1, child2]
        # Recursively split the child rooms
        split_room(child1, depth + 1)
        split_room(child2, depth + 1)
    
    def add_openings(start, end, orientation='horizontal'):
        """
        Adds random openings to a wall.

        Args:
            start (tuple): Starting coordinate (y, x).
            end (tuple): Ending coordinate (y, x).
            orientation (str): 'horizontal' or 'vertical'.
        """
        num_openings = random.randint(min_openings, max_openings)
        if orientation == 'horizontal':
            wall_length = end[1] - start[1]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)  # Ensure opening size doesn't exceed wall
                opening_start = random.randint(start[1], end[1] - opening_size)
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        else:
            wall_length = end[0] - start[0]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)  # Ensure opening size doesn't exceed wall
                opening_start = random.randint(start[0], end[0] - opening_size)
                map_grid[opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness] = 0
    
    def place_concave_trap(room):
        """
        Places a concave trap within the given room. The trap can be L-shaped or triangular.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        trap_type = random.choice(['L', 'triangle'])
        if trap_type == 'L':
            return place_L_shaped_trap(room)
        else:
            return place_triangular_trap(room)
    
    def place_L_shaped_trap(room):
        """
        Places an L-shaped concave trap within the given room.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        trap_size = random.randint(min_obstacle_size, max_obstacle_size)
        trap_thickness = wall_thickness  # Thickness of the trap arms

        # Ensure the trap fits within the room
        if (trap_size * 2 + wall_thickness) > room.width or (trap_size * 2 + wall_thickness) > room.height:
            return False  # Trap too big for the room

        # Choose a position for the corner of the L-shape
        corner_x = random.randint(room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
        corner_y = random.randint(room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)

        # Randomly decide the orientation of the L-shape
        orientation = random.choice(['left', 'right', 'up', 'down'])

        if orientation == 'left':
            # Horizontal arm to the left, vertical arm upwards
            arm1 = ((corner_y, corner_x - trap_size), (trap_size, trap_thickness))
            arm2 = ((corner_y - trap_size, corner_x), (trap_thickness, trap_size))
        elif orientation == 'right':
            # Horizontal arm to the right, vertical arm upwards
            arm1 = ((corner_y, corner_x), (trap_size, trap_thickness))
            arm2 = ((corner_y - trap_size, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))
        elif orientation == 'up':
            # Vertical arm upwards, horizontal arm to the left
            arm1 = ((corner_y - trap_size, corner_x), (trap_thickness, trap_size))
            arm2 = ((corner_y - trap_size, corner_x - trap_size), (trap_size, trap_thickness))
        else:  # 'down'
            # Vertical arm downwards, horizontal arm to the right
            arm1 = ((corner_y, corner_x), (trap_thickness, trap_size))
            arm2 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_size, trap_thickness))
        
        # Check if arms are within bounds
        (y1, x1), (h1, w1) = arm1
        (y2, x2), (h2, w2) = arm2

        if (x1 < 0 or y1 < 0 or x1 + w1 > width or y1 + h1 > height or
            x2 < 0 or y2 < 0 or x2 + w2 > width or y2 + h2 > height):
            return False  # Out of bounds

        # Check for overlap with existing obstacles
        if (np.any(map_grid[y1:y1 + h1, x1:x1 + w1] == 1) or
            np.any(map_grid[y2:y2 + h2, x2:x2 + w2] == 1)):
            return False  # Overlaps with existing obstacle

        # Place the L-shaped trap
        map_grid[y1:y1 + h1, x1:x1 + w1] = 1
        map_grid[y2:y2 + h2, x2:x2 + w2] = 1

        return True  # Successfully placed

    def place_triangular_trap(room):
        """
        Places a triangular concave trap within the given room.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        trap_size = random.randint(min_obstacle_size, max_obstacle_size)
        trap_thickness = wall_thickness  # Thickness of the trap lines

        # Define the three lines of the triangle (right-angled for simplicity)
        # Choose a position for the right angle
        corner_x = random.randint(room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
        corner_y = random.randint(room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)

        # Randomly decide the orientation of the triangle
        orientation = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])

        if orientation == 'top-left':
            arm1 = ((corner_y, corner_x), (trap_thickness, trap_size))          # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))          # Horizontal
            arm3 = ((corner_y, corner_x), (trap_thickness, trap_thickness))    # Diagonal
        elif orientation == 'top-right':
            arm1 = ((corner_y, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))  # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))                              # Horizontal
            arm3 = ((corner_y, corner_x + trap_size - trap_thickness), (trap_thickness, trap_thickness))  # Diagonal
        elif orientation == 'bottom-left':
            arm1 = ((corner_y + trap_size - trap_thickness, corner_x), (trap_thickness, trap_size))  # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))                              # Horizontal
            arm3 = ((corner_y + trap_size - trap_thickness, corner_x), (trap_thickness, trap_thickness))  # Diagonal
        else:  # 'bottom-right'
            arm1 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))  # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))                              # Horizontal
            arm3 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_thickness, trap_thickness))  # Diagonal

        # Check if arms are within bounds
        arms = [arm1, arm2, arm3]
        for (y, x), (h, w) in arms:
            if (x < 0 or y < 0 or x + w > width or y + h > height):
                return False  # Out of bounds

        # Check for overlap with existing obstacles
        for (y, x), (h, w) in arms:
            if np.any(map_grid[y:y + h, x:x + w] == 1):
                return False  # Overlaps with existing obstacle

        # Place the triangular trap
        for (y, x), (h, w) in arms:
            map_grid[y:y + h, x:x + w] = 1

        return True  # Successfully placed

    # Start splitting from the root room
    split_room = locals()['split_room']  # To ensure recursive references work
    split_room(root_room, 0)
    
    # Collect all leaf rooms
    leaf_rooms = []
    def collect_leaf_rooms(room):
        if not room.children:
            leaf_rooms.append(room)
        else:
            for child in room.children:
                collect_leaf_rooms(child)

    collect_leaf_rooms(root_room)
    
    # Add obstacles to each leaf room
    for room in leaf_rooms:
        num_obstacles = random.randint(min_obstacles, max_obstacles)
        for _ in range(num_obstacles):
            if random.random() < trap_probability:
                # Attempt to place a concave trap
                placed = False
                for attempt in range(obstacle_attempts):
                    if place_concave_trap(room):
                        placed = True
                        break  # Successfully placed
                # if not placed:
                    # print(f"Could not place a concave trap in room at ({room.x}, {room.y}) after {obstacle_attempts} attempts.")
                continue  # Move to next obstacle
            else:
                # Place a regular rectangular obstacle
                placed = False
                for attempt in range(obstacle_attempts):
                    obstacle_w = random.randint(min_obstacle_size, max_obstacle_size)
                    obstacle_h = random.randint(min_obstacle_size, max_obstacle_size)
                    # Ensure obstacle fits within the room with some padding
                    if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
                        continue  # Skip if obstacle is too big for the room
                    obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
                    obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
                    # Check for overlap
                    if np.any(map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] == 1):
                        continue  # Overlaps with existing obstacle
                    # Place the rectangular obstacle
                    map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 1
                    placed = True
                    break  # Successfully placed
                if not placed:
                    # print(f"Could not place a rectangular obstacle in room at ({room.x}, {room.y}) after {obstacle_attempts} attempts.")
                    continue  # Skip if unable to place after attempts

    # Add outer boundary walls
    # Top and bottom
    map_grid[0:wall_thickness, :] = 1
    map_grid[-wall_thickness:, :] = 1
    # Left and right
    map_grid[:, 0:wall_thickness] = 1
    map_grid[:, -wall_thickness:] = 1
    
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

def visualize_comparison(map_grid, start, goal, paths, f_maps, output_dir="output", run=0):
    """
    Generates and saves a comprehensive visualization comparing multiple A* algorithms.
    
    :param map_grid: 2D numpy array representing the map.
    :param start: Tuple representing the start position.
    :param goal: Tuple representing the goal position.
    :param paths: Dictionary where keys are algorithm names and values are paths (list of positions).
    :param f_maps: Dictionary where keys are algorithm names and values are f-value maps.
    :param output_dir: Directory to save the visualization.
    :param run: Integer representing the current run/query number.
    """
    num_algorithms = len(paths)
    plt.figure(figsize=(6 * num_algorithms, 6))
    
    # Create custom colormap for the map_grid
    cmap = ListedColormap(['white', 'black'])

    for idx, (algo_name, path) in enumerate(paths.items(), 1):
        plt.subplot(1, num_algorithms, idx)
        plt.title(algo_name)
        plt.imshow(map_grid, cmap=cmap, interpolation='nearest')
        
        f_map = f_maps.get(algo_name, np.nan)
        if not np.all(np.isnan(f_map)):
            valid_positions = np.argwhere(np.isfinite(f_map))
            scatter = plt.scatter(valid_positions[:, 1], valid_positions[:, 0],
                                  c=f_map[valid_positions[:, 0], valid_positions[:, 1]],
                                  cmap='viridis', s=.2, alpha=0.8)
            plt.colorbar(scatter, label="f values")

        if path:
            # Plot the path
            plt.plot([p[1] for p in path], [p[0] for p in path], 'b-', linewidth=2)  # Path only

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
# Heuristic Generation Functions
# ---------------------------

# ---------------------------
# Heuristic Generation Functions
# ---------------------------

def inflate_obstacles(map_grid, radius):
    """
    Inflates obstacles in the map using binary dilation.

    Args:
        map_grid (np.ndarray): Original map grid.
        radius (int): Radius for inflation.

    Returns:
        np.ndarray: Inflated map grid.
    """
    structure = np.ones((2 * radius + 1, 2 * radius + 1))
    inflated_map = binary_dilation(map_grid, structure=structure)
    return inflated_map.astype(np.float32)

def compute_heuristic_map(map_grid, goal):
    """
    Computes the heuristic map using Dijkstra's algorithm from the goal.

    Args:
        map_grid (np.ndarray): Map grid where obstacles are marked as 1.
        goal (tuple): Goal position (y, x).

    Returns:
        np.ndarray: Heuristic map with distances to the goal.
    """
    h_map = np.full(map_grid.shape, np.inf)
    visited = np.zeros(map_grid.shape, dtype=bool)
    h_map[goal] = 0
    heap = [(0, goal)]

    while heap:
        cost, current = heapq.heappop(heap)
        if visited[current]:
            continue
        visited[current] = True

        for neighbor, move_cost in get_neighbors(current):
            if not is_valid(neighbor, map_grid) or visited[neighbor]:
                continue
            tentative_cost = cost + move_cost
            if tentative_cost < h_map[neighbor]:
                h_map[neighbor] = tentative_cost
                heapq.heappush(heap, (tentative_cost, neighbor))

    return h_map

def extract_path_from_heuristic(h_map, start, goal):
    """
    Extracts a path from start to goal using the heuristic map.

    Args:
        h_map (np.ndarray): Heuristic map with distances to the goal.
        start (tuple): Start position (y, x).
        goal (tuple): Goal position (y, x).

    Returns:
        list: Path from start to goal as a list of positions.
    """
    path = []
    current = start
    while current != goal:
        path.append(current)
        min_cost = np.inf
        next_step = None
        for neighbor, _ in get_neighbors(current):
            if h_map[neighbor] < min_cost:
                min_cost = h_map[neighbor]
                next_step = neighbor
        if next_step is None:
            break  # No path found
        current = next_step
    path.append(goal)
    return path

def identify_bottlenecks(path, map_grid):
    """
    Identifies bottleneck positions in the path where the number of free neighbors is <= 2.

    Args:
        path (list): Path as a list of positions.
        map_grid (np.ndarray): Original map grid.

    Returns:
        list: List of bottleneck positions.
    """
    bottlenecks = []
    for pos in path:
        free_neighbors = sum(is_valid(neighbor, map_grid) for neighbor, _ in get_neighbors(pos))
        if free_neighbors <= 2:
            bottlenecks.append(pos)
    return bottlenecks

def block_bottlenecks(map_grid, bottlenecks):
    """
    Blocks bottleneck positions in the map by setting them as obstacles.

    Args:
        map_grid (np.ndarray): Original map grid.
        bottlenecks (list): List of bottleneck positions.

    Returns:
        np.ndarray: Modified map grid with bottlenecks blocked.
    """
    map_with_blocked_bottlenecks = map_grid.copy()
    for pos in bottlenecks:
        map_with_blocked_bottlenecks[pos] = 1  # Mark as obstacle
    return map_with_blocked_bottlenecks

def generate_dual_heuristics(map_grid, goal, inflation_radius=5):
    """
    Generates dual heuristics by inflating obstacles and computing separate heuristic maps.

    Args:
        map_grid (np.ndarray): Original map grid.
        goal (tuple): Goal position.
        inflation_radius (int): Radius for obstacle inflation.

    Returns:
        list: List of heuristic functions.
    """
    # Original heuristic map (h0)
    h0_map = compute_heuristic_map(map_grid, goal)

    # Dual heuristic (h1) by inflating obstacles
    inflated_map = inflate_obstacles(map_grid, radius=inflation_radius)
    h1_map = compute_heuristic_map(inflated_map, goal)

    # Heuristic functions using precomputed heuristic maps
    h_functions = [
        lambda pos, h_map=h0_map: h_map[pos],
        lambda pos, h_map=h1_map: h_map[pos],
    ]

    return h_functions

def generate_progressive_heuristics(map_grid, goal, start, max_iterations=10):
    """
    Generates progressive heuristics by identifying bottlenecks and computing additional heuristic maps.

    Args:
        map_grid (np.ndarray): Original map grid.
        goal (tuple): Goal position.
        start (tuple): Start position.
        max_iterations (int): Maximum number of progressive heuristics to generate.

    Returns:
        list: List of heuristic functions.
    """
    h_functions = []
    map_copy = map_grid.copy()
    iteration = 0

    while iteration < max_iterations:
        h_map = compute_heuristic_map(map_copy, goal)
        if np.isinf(h_map[start]):
            break  # No path exists

        h_functions.append(lambda pos, h_map=h_map: h_map[pos])

        path = extract_path_from_heuristic(h_map, start, goal)
        bottlenecks = identify_bottlenecks(path, map_copy)
        if not bottlenecks:
            break  # No more bottlenecks to block

        map_copy = block_bottlenecks(map_copy, bottlenecks)
        iteration += 1

    return h_functions

def generate_mha_heuristics(map_grid, goal, start):
    """
    Generates multiple heuristics for MHA* algorithms, including dual and progressive heuristics.

    Args:
        map_grid (np.ndarray): Original map grid.
        goal (tuple): Goal position.
        start (tuple): Start position.

    Returns:
        list: List of heuristic functions.
    """
    # Dual Heuristics
    dual_heuristics = generate_dual_heuristics(map_grid, goal, inflation_radius=5)

    # Progressive Heuristics
    progressive_heuristics = generate_progressive_heuristics(map_grid, goal, start, max_iterations=5)

    # Combine all heuristics
    h_functions = dual_heuristics + progressive_heuristics

    return h_functions


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
        encoded_map = encoder.encode(map_tensor).cpu().numpy().flatten()  # Should be 1024-dimensional

    h = euclidean_distance(current, goal)

    g_normalized = normalize(g, g_min, g_max)
    h_normalized = normalize(h, h_min, h_max)

    start_normalized = np.array(start) / 127.0
    goal_normalized = np.array(goal) / 127.0
    current_normalized = np.array(current) / 127.0
    input_tensor = np.concatenate([
        start_normalized,
        goal_normalized,              # 2 values
        current_normalized,           # 2 values
        encoded_map,                   # 1024 values
        [g_normalized, h_normalized] # 2 values
    ])  # Total: 1032

    input_tensor = torch.from_numpy(input_tensor).float().to(device)  # Shape: [1032]
    
    # Ensure input_tensor has shape [1, 1032]
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, 1032]

    model.eval()
    with torch.no_grad():
        f_star_predicted = model(input_tensor)
        # print(f"f_star_predicted type: {type(f_star_predicted)}")
        # print(f"f_star_predicted contents: {f_star_predicted}")
        if isinstance(f_star_predicted, tuple) or isinstance(f_star_predicted, list):
            f_star_predicted = f_star_predicted[0]
            # print(f"Extracted f_star_predicted from tuple: {f_star_predicted}")
        # print(f"f_star_predicted shape: {f_star_predicted.shape if isinstance(f_star_predicted, torch.Tensor) else 'N/A'}")
    
        # Attempt to get the scalar value
        if isinstance(f_star_predicted, torch.Tensor):
            if f_star_predicted.numel() == 1:
                f_star_denormalized = denormalize(f_star_predicted.item(), f_star_min, f_star_max)
            else:
                # If tensor has more than one element, handle accordingly
                f_star_denormalized = denormalize(f_star_predicted.mean().item(), f_star_min, f_star_max)
                print("Warning: f_star_predicted has multiple elements. Using mean value for denormalization.")
        else:
            raise TypeError(f"Expected f_star_predicted to be a torch.Tensor, but got {type(f_star_predicted)}")
    
    return f_star_denormalized


# ---------------------------
# Search Algorithm Classes
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
# Heuristic-Based Search Algorithms
# ---------------------------

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
    start_node = WeightedAStarNode(start, g=0, h=h_start)
    heapq.heappush(open_list, start_node)
    expansions = 0

    f_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)

        # **Add the check here**
        if current.pos in closed_set:
            continue  # Skip if already expanded

        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_map

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
            next_node = WeightedAStarNode(next_pos, tentative_g, h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, None, expansions, f_map  # Return None values if no path found

def potential_search(start, goal, map_grid, B=3.0, max_expansions=100000):
    class PotentialNode:
        def __init__(self, pos, g, h, parent=None):
            self.pos = pos
            self.g = g
            self.h = h
            self.f = g + h  # f(n) = g(n) + h(n)
            self.parent = parent
            self.flnr = h / max(B * self.f - g, 1e-8)  # Initial flnr computation

        def __lt__(self, other):
            return self.flnr < other.flnr

    open_list = []
    closed_set = set()
    g_score = {start: 0}
    re_expansions = 0
    h_start = euclidean_distance(start, goal)
    f_min = h_start  # Since g=0 at the start, f = h
    C = B * f_min

    start_node = PotentialNode(start, g=0, h=h_start)
    heapq.heappush(open_list, (start_node.flnr, start_node))
    expansions = 0

    flnr_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current_flnr, current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            print("Dynamic Potential Search: Max expansions reached.")
            return None, None, expansions, re_expansions, flnr_map

        # Skip processing if we have already found a better path
        if current.g > g_score.get(current.pos, float('inf')):
            continue

        # Check for re-expansion
        if current.pos in closed_set:
            re_expansions += 1
            # Only proceed if we have a better g value
            if current.g >= g_score[current.pos]:
                continue
        else:
            closed_set.add(current.pos)

        g_score[current.pos] = current.g
        flnr_map[current.pos[0], current.pos[1]] = current.flnr

        # Check if goal is reached within cost bound
        if current.pos == goal and current.g <= C:
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            print(f"DPS: Path found with cost {total_cost} in {expansions} expansions and {re_expansions} re-expansions.")
            return path[::-1], total_cost, expansions, re_expansions, flnr_map

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost
            h = euclidean_distance(next_pos, goal)
            f = tentative_g + h

            # Prune paths that exceed the cost bound
            if tentative_g > C:
                continue

            # Update f_min and cost bound C if a better f is found
            if f < f_min:
                f_min = f
                C = B * f_min
                # Recompute flnr for all nodes in open_list
                new_open_list = []
                while open_list:
                    _, node = heapq.heappop(open_list)
                    node.flnr = node.h / max(C - node.g, 1e-8) if (C - node.g) > 0 else float('inf')
                    heapq.heappush(new_open_list, (node.flnr, node))
                open_list = new_open_list

            # Skip if not a better path
            if tentative_g >= g_score.get(next_pos, float('inf')):
                continue

            # Update g_score and add neighbor to open_list
            g_score[next_pos] = tentative_g
            flnr = h / max(C - tentative_g, 1e-8) if (C - tentative_g) > 0 else float('inf')
            neighbor_node = PotentialNode(next_pos, tentative_g, h, parent=current)
            neighbor_node.flnr = flnr
            heapq.heappush(open_list, (neighbor_node.flnr, neighbor_node))

    print("Dynamic Potential Search: No solution found.")
    return None, None, expansions, re_expansions, flnr_map

def imha_star(start, goal, map_grid, heuristic_functions, w=1.0, max_expansions=100000):
    """
    Independent Multi-Heuristic A* (IMHA*)
    Each heuristic has its own open list and operates independently.
    """
    num_heuristics = len(heuristic_functions)
    open_lists = [[] for _ in range(num_heuristics)]
    closed_set = set()
    g_score = {}
    expansions = 0

    # Initialize nodes for each heuristic
    for i, h_func in enumerate(heuristic_functions):
        h_value = h_func(start)
        node = AStarNode(start, g=0, h=h_value)
        heapq.heappush(open_lists[i], (node.g + w * node.h, node))
        g_score[(start, i)] = 0  # Separate g-scores per heuristic

    f_map = np.full(map_grid.shape, np.nan)

    while any(open_lists):
        for i, open_list in enumerate(open_lists):
            if not open_list:
                continue
            f_current, current = heapq.heappop(open_list)
            expansions += 1

            if expansions > max_expansions:
                return None, None, expansions, f_map

            if current.pos == goal:
                # Reconstruct path
                path = []
                total_cost = current.g
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1], total_cost, expansions, f_map

            if current.pos in closed_set:
                continue

            closed_set.add(current.pos)
            f_map[current.pos[0], current.pos[1]] = f_current

            for next_pos, cost in get_neighbors(current.pos):
                if not is_valid(next_pos, map_grid):
                    continue

                tentative_g = current.g + cost

                if tentative_g >= g_score.get((next_pos, i), float('inf')):
                    continue

                g_score[(next_pos, i)] = tentative_g
                h_value = heuristic_functions[i](next_pos)
                next_node = AStarNode(next_pos, g=tentative_g, h=h_value, parent=current)
                heapq.heappush(open_lists[i], (next_node.g + w * next_node.h, next_node))

    return None, None, expansions, f_map

def smha_star(start, goal, map_grid, heuristic_functions, w1=1.0, w2=1.0, max_expansions=100000):
    """
    Shared Multi-Heuristic A* (SMHA*)
    All heuristics share a single open list.
    """
    class SMHANode:
        def __init__(self, pos, g, h_values, parent=None):
            self.pos = pos
            self.g = g
            self.h_values = h_values  # List of heuristic values
            self.parent = parent
            self.f_anchor = g + h_values[0]  # Anchor heuristic
            self.f_inadmissible = [g + h for h in h_values[1:]]
    
        def __lt__(self, other):
            return self.f_anchor < other.f_anchor

    open_list = []
    closed_set = set()
    g_score = {start: 0}
    expansions = 0

    h_values_start = [h(start) for h in heuristic_functions]
    start_node = SMHANode(start, g=0, h_values=h_values_start)
    heapq.heappush(open_list, (start_node.f_anchor, start_node))

    f_map = np.full(map_grid.shape, np.nan)

    while open_list:
        f_current, current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_map

        if current.pos == goal:
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_map

        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)
        f_map[current.pos[0], current.pos[1]] = f_current

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost

            if tentative_g >= g_score.get(next_pos, float('inf')):
                continue

            g_score[next_pos] = tentative_g
            h_values = [h(next_pos) for h in heuristic_functions]
            next_node = SMHANode(next_pos, g=tentative_g, h_values=h_values, parent=current)
            f_anchor = tentative_g + w1 * h_values[0]
            f_inadmissible = [tentative_g + w2 * h for h in h_values[1:]]
            f_min = min([f_anchor] + f_inadmissible)
            heapq.heappush(open_list, (f_min, next_node))

    return None, None, expansions, f_map

# ---------------------------
# Model-Based Search Algorithms
# ---------------------------

def astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, max_expansions=100000):
    """
    A* Search Algorithm with Neural Network-based Heuristic.

    Args:
        start (tuple): Starting position (y, x).
        goal (tuple): Goal position (y, x).
        map_grid (np.ndarray): 2D occupancy map where 0 is free and 1 is obstacle.
        encoder (UNet2DAutoencoder): Pre-trained encoder model.
        model (MLPModel): Pre-trained MLP model for heuristic prediction.
        normalization_values (dict): Normalization parameters for inputs and outputs.
        device (torch.device): Device to run the models on.
        max_expansions (int): Maximum number of node expansions allowed.

    Returns:
        tuple:
            - path (list): List of positions from start to goal. None if no path found.
            - total_cost (float): Total cost of the path. None if no path found.
            - expansions (int): Total number of node expansions performed.
            - expansion_counts (dict): Dictionary mapping states to their expansion counts.
            - f_star_map (np.ndarray): Map of f_star values for visualization.
    """
    open_list = []
    g_score = {start: 0}
    expansion_counts = {}  # To track how many times each state is expanded
    closed_set = set()

    # Compute initial f_star for the start node using the neural network model
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_node = ModelNode(pos=start, f_star=start_f_star, g=0, parent=None)
    heapq.heappush(open_list, start_node)
    expansions = 0

    # Initialize f_star_map with NaNs for visualization
    f_star_map = np.full(map_grid.shape, np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        if current.pos in closed_set:
            continue
        expansions += 1
        expansion_counts[current.pos] = expansion_counts.get(current.pos, 0) + 1

        if expansions > max_expansions:
            return None, None, expansions, expansion_counts, f_star_map

        if current.pos == goal:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g  # Total cost from start to goal
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, expansion_counts, f_star_map

        closed_set.add(current.pos)

        # Record f_star value for visualization
        f_star_map[current.pos[0], current.pos[1]] = current.f_star

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost

            if next_pos in closed_set:
                if tentative_g < g_score.get(next_pos, float('inf')):
                    # Found a better path to a node in closed set, re-open it
                    closed_set.remove(next_pos)
                else:
                    continue  # Not a better path, skip

            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                f_star = run_inference(map_grid, start, goal, next_pos, tentative_g, encoder, model, normalization_values, device)
                next_node = ModelNode(pos=next_pos, f_star=f_star, g=tentative_g, parent=current)
                heapq.heappush(open_list, next_node)

    return None, None, expansions, expansion_counts, f_star_map  # Return None values if no path found

def focal_astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, epsilon=3.0, max_expansions=100000):
    """
    Focal A* Search Algorithm with Neural Network-based Heuristic.

    Args:
        start (tuple): Starting position (y, x).
        goal (tuple): Goal position (y, x).
        map_grid (np.ndarray): 2D occupancy map where 0 is free and 1 is obstacle.
        encoder (UNet2DAutoencoder): Pre-trained encoder model.
        model (MLPModel): Pre-trained MLP model for heuristic prediction.
        normalization_values (dict): Normalization parameters for inputs and outputs.
        device (torch.device): Device to run the models on.
        epsilon (float): Suboptimality factor.
        max_expansions (int): Maximum number of node expansions allowed.

    Returns:
        tuple:
            - path (list): List of positions from start to goal. None if no path found.
            - total_cost (float): Total cost of the path. None if no path found.
            - expansions (int): Total number of node expansions performed.
            - expansion_counts (dict): Dictionary mapping states to their expansion counts.
            - f_star_map (np.ndarray): Map of f_star values for visualization.
    """
    import bisect

    class OpenList:
        def __init__(self):
            self.elements = []
            self.entry_finder = {}
            self.counter = itertools.count()

        def add_node(self, node):
            # Use node.f_star instead of node.g + node.h
            f = node.f_star
            if node.pos in self.entry_finder:
                existing_count, existing_node = self.entry_finder[node.pos]
                if node.f_star < existing_node.f_star:
                    self.remove_node(existing_node)
                else:
                    return  # Existing node has a better or equal path
            count_value = next(self.counter)
            bisect.insort_left(self.elements, (f, count_value, node))
            self.entry_finder[node.pos] = (count_value, node)

        def remove_node(self, node):
            if node.pos in self.entry_finder:
                count_value, existing_node = self.entry_finder[node.pos]
                f = existing_node.f_star
                idx = bisect.bisect_left(self.elements, (f, count_value, existing_node))
                while idx < len(self.elements):
                    if self.elements[idx][2].pos == node.pos:
                        self.elements.pop(idx)
                        break
                    idx += 1
                del self.entry_finder[node.pos]

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

    open_list = OpenList()
    g_score = {start: 0}
    expansion_counts = {}  # To track how many times each state is expanded
    expansions = 0
    closed_set = set()

    # Compute initial f_star for the start node using the neural network model
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_node = ModelNode(pos=start, f_star=start_f_star, g=0, parent=None)
    open_list.add_node(start_node)

    # Initialize f_star_map with NaNs for visualization
    f_star_map = np.full(map_grid.shape, np.nan)

    while not open_list.is_empty():
        current_f_min = open_list.get_f_min()
        focal_nodes = open_list.get_focal_nodes(current_f_min, epsilon)

        if not focal_nodes:
            return None, None, expansions, expansion_counts, f_star_map

        # Select the node with the smallest f_star from the focal list
        current = min(focal_nodes, key=lambda node: node.f_star)
        open_list.remove_node(current)
        if current.pos in closed_set:
            continue
        expansions += 1
        expansion_counts[current.pos] = expansion_counts.get(current.pos, 0) + 1

        if expansions > max_expansions:
            return None, None, expansions, expansion_counts, f_star_map

        if current.pos == goal:
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g  # Total cost from start to goal
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, expansion_counts, f_star_map

        closed_set.add(current.pos)

        # Record f_star value for visualization
        f_star_map[current.pos[0], current.pos[1]] = current.f_star

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost

            if next_pos in closed_set:
                if tentative_g < g_score.get(next_pos, float('inf')):
                    # Found a better path to a node in closed set, re-open it
                    closed_set.remove(next_pos)
                else:
                    continue  # Not a better path, skip

            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                f_star = run_inference(map_grid, start, goal, next_pos, tentative_g, encoder, model, normalization_values, device)
                next_node = ModelNode(pos=next_pos, f_star=f_star, g=tentative_g, parent=current)
                open_list.add_node(next_node)

    return None, None, expansions, expansion_counts, f_star_map  # Return None values if no path found

# ---------------------------
# IMHA* and SMHA* Correct Implementations with Enhanced Heuristics
# ---------------------------

def imha_star(start, goal, map_grid, heuristic_functions, w=1.0, max_expansions=100000):
    """
    Independent Multi-Heuristic A* (IMHA*)
    Each heuristic has its own open list and operates independently.
    """
    num_heuristics = len(heuristic_functions)
    open_lists = [[] for _ in range(num_heuristics)]
    closed_set = set()
    g_score = {}
    expansions = 0

    # Initialize nodes for each heuristic
    for i, h_func in enumerate(heuristic_functions):
        h_value = h_func(start)
        node = AStarNode(start, g=0, h=h_value)
        heapq.heappush(open_lists[i], (node.g + w * node.h, node))
        g_score[(start, i)] = 0  # Separate g-scores per heuristic

    f_map = np.full(map_grid.shape, np.nan)

    while any(open_lists):
        for i, open_list in enumerate(open_lists):
            if not open_list:
                continue
            f_current, current = heapq.heappop(open_list)
            expansions += 1

            if expansions > max_expansions:
                return None, None, expansions, f_map

            if current.pos == goal:
                # Reconstruct path
                path = []
                total_cost = current.g
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1], total_cost, expansions, f_map

            if current.pos in closed_set:
                continue

            closed_set.add(current.pos)
            f_map[current.pos[0], current.pos[1]] = f_current

            for next_pos, cost in get_neighbors(current.pos):
                if not is_valid(next_pos, map_grid):
                    continue

                tentative_g = current.g + cost

                if tentative_g >= g_score.get((next_pos, i), float('inf')):
                    continue

                g_score[(next_pos, i)] = tentative_g
                h_value = heuristic_functions[i](next_pos)
                next_node = AStarNode(next_pos, g=tentative_g, h=h_value, parent=current)
                heapq.heappush(open_lists[i], (next_node.g + w * next_node.h, next_node))

    return None, None, expansions, f_map

def smha_star(start, goal, map_grid, heuristic_functions, w1=1.0, w2=1.0, max_expansions=100000):
    """
    Shared Multi-Heuristic A* (SMHA*)
    All heuristics share a single open list.
    """
    class SMHANode:
        def __init__(self, pos, g, h_values, parent=None):
            self.pos = pos
            self.g = g
            self.h_values = h_values  # List of heuristic values
            self.parent = parent
            self.f_anchor = g + h_values[0]  # Anchor heuristic
            self.f_inadmissible = [g + h for h in h_values[1:]]
    
        def __lt__(self, other):
            return self.f_anchor < other.f_anchor

    open_list = []
    closed_set = set()
    g_score = {start: 0}
    expansions = 0

    h_values_start = [h(start) for h in heuristic_functions]
    start_node = SMHANode(start, g=0, h_values=h_values_start)
    heapq.heappush(open_list, (start_node.f_anchor, start_node))

    f_map = np.full(map_grid.shape, np.nan)

    while open_list:
        f_current, current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            return None, None, expansions, f_map

        if current.pos == goal:
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], total_cost, expansions, f_map

        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)
        f_map[current.pos[0], current.pos[1]] = f_current

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost

            if tentative_g >= g_score.get(next_pos, float('inf')):
                continue

            g_score[next_pos] = tentative_g
            h_values = [h(next_pos) for h in heuristic_functions]

            next_node = SMHANode(next_pos, g=tentative_g, h_values=h_values, parent=current)
            f_anchor = tentative_g + w1 * h_values[0]
            f_inadmissible = [tentative_g + w2 * h for h in h_values[1:]]
            f_min = min([f_anchor] + f_inadmissible)
            heapq.heappush(open_list, (f_min, next_node))

    return None, None, expansions, f_map

# ---------------------------
# Heuristic-Based MHA* Enhanced with Dual and Progressive Heuristics
# ---------------------------

def generate_mha_heuristics(map_grid, goal, start):
    """
    Generates multiple heuristics for MHA* algorithms, including dual and progressive heuristics.

    Args:
        map_grid (np.ndarray): Original map grid.
        goal (tuple): Goal position.
        start (tuple): Start position.

    Returns:
        list: List of heuristic functions.
    """
    # Dual Heuristics
    dual_heuristics = generate_dual_heuristics(map_grid, goal, inflation_radius=5)

    # Progressive Heuristics
    progressive_heuristics = generate_progressive_heuristics(map_grid, goal, start, max_iterations=5)

    # Combine all heuristics
    h_functions = dual_heuristics + progressive_heuristics

    return h_functions

# ---------------------------
# Assessment Function
# ---------------------------

def run_assessment(encoder, models, normalization_values, device, num_maps=1, num_queries_per_map=10, output_csv="output.csv", output_dir="visualizations"):
    """
    Runs assessment of various search algorithms on generated maps and records their performance.

    Args:
        encoder (UNet2DAutoencoder): Pre-trained encoder model.
        models (dict): Dictionary of pre-trained MLP models.
        normalization_values (dict): Normalization parameters for models.
        device (torch.device): Device to run the models on.
        num_maps (int): Number of maps to generate for assessment.
        num_queries_per_map (int): Number of queries per map.
        output_csv (str): Path to the output CSV file.
        output_dir (str): Directory to save visualizations.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define all algorithms including model-based ones
    algorithms = ['Traditional A*', 'Weighted A* ε=3.0', 'Potential Search', 'IMHA*', 'SMHA*']
    for model_name in models:
        algorithms.append(f'{model_name} A*')
        algorithms.append(f'{model_name} Focal A* ε=3.0')

    # Initialize statistics dictionary with re-expansions tracking
    stats = {algo: {
        'total_expansions_reduction': 0,
        'total_path_cost_increase': 0,
        'path_cost_increases': [],
        'optimal_paths': 0,
        'total_queries': 0,
        'total_re_expansions': 0,          # Total re-expansions across all queries
        're_expansions_per_query': [],      # List of re-expansions per query
        'total_expansions': 0               # Total expansions across all queries
    } for algo in algorithms}

    query_counter = 0
    for map_idx in range(num_maps):
        map_data = generate_map()
        print(f"\nRunning assessment on generated map {map_idx + 1}\n")

        for query in range(num_queries_per_map):
            print(f"\nRunning assessment on query {query + 1} for map {map_idx + 1}...\n")

            # Traditional A*
            print(f"Running Traditional A* on query {query + 1}...")
            while True:
                start, goal = generate_start_goal(map_data, min_distance_ratio=0.2)  # Ensuring a minimum distance
                traditional_path, traditional_path_cost, traditional_expanded, f_map = astar_traditional(start, goal, map_data, max_expansions=20000)
                if traditional_path is not None:
                    break

            # Store results for visualization
            visualization_data = {
                'start': start,
                'goal': goal,
                'paths': {},
                'f_maps': {}
            }

            print(f"Traditional A* expansions: {traditional_expanded}, cost: {traditional_path_cost}")

            # Save Traditional A* results for visualization
            visualization_data['paths']['Traditional A*'] = traditional_path
            visualization_data['f_maps']['Traditional A*'] = f_map

            # Initialize re-expansions dictionary
            expansion_counts_dict = {}

            # Weighted A* ε=3.0
            print(f"Running Weighted A* ε=3.0 on query {query + 1}...")
            epsilon = 3.0
            algo_name = f'Weighted A* ε={epsilon}'
            path, path_cost, expanded, f_map_weighted = astar_weighted(start, goal, map_data, epsilon=epsilon, max_expansions=100000)

            if path is None:
                print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
            else:
                print(f"{algo_name} expansions: {expanded}, cost: {path_cost}")
                expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

                # Since Weighted A* does not track re-expansions, set to 0
                re_expansions = 0
                stats[algo_name]['total_re_expansions'] += re_expansions
                stats[algo_name]['re_expansions_per_query'].append(re_expansions)

                # **Add total expansions tracking**
                stats[algo_name]['total_expansions'] += expanded

                # Save Weighted A* results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = f_map_weighted

            # Potential Search
            print(f"Running Potential Search on query {query + 1}...")
            B = 3.0  # Suboptimality bound
            C = traditional_path_cost * B  # Set to B times the optimal cost
            algo_name = 'Potential Search'
            path, path_cost, expanded, re_expansions_ps, flnr_map = potential_search(start, goal, map_data, B=3, max_expansions=100000)

            if path is None:
                print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
            else:
                print(f"Potential Search expansions: {expanded}, cost: {path_cost}, re-expansions: {re_expansions_ps}")
                expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(path_cost - traditional_path_cost) < 1:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

                # **Update re-expansions**
                stats[algo_name]['total_re_expansions'] += re_expansions_ps
                stats[algo_name]['re_expansions_per_query'].append(re_expansions_ps)

                # **Add total expansions tracking**
                stats[algo_name]['total_expansions'] += expanded

                # Save Potential Search results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = flnr_map

            # Generate Heuristics for MHA* based algorithms
            heuristic_functions = generate_mha_heuristics(map_data, goal, start)

            # IMHA*
            print(f"Running IMHA* on query {query + 1}...")
            algo_name = 'IMHA*'
            path, path_cost, expanded, f_map_imha = imha_star(start, goal, map_grid=map_data, heuristic_functions=heuristic_functions, w=1.0, max_expansions=100000)

            if path is None:
                print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
            else:
                print(f"{algo_name} expansions: {expanded}, cost: {path_cost}")
                expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

                # Since IMHA* does not track re-expansions, set to 0
                re_expansions = 0
                stats[algo_name]['total_re_expansions'] += re_expansions
                stats[algo_name]['re_expansions_per_query'].append(re_expansions)

                # **Add total expansions tracking**
                stats[algo_name]['total_expansions'] += expanded

                # Save IMHA* results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = f_map_imha

            # SMHA*
            print(f"Running SMHA* on query {query + 1}...")
            algo_name = 'SMHA*'
            path, path_cost, expanded, f_map_smha = smha_star(start, goal, map_grid=map_data, heuristic_functions=heuristic_functions, w1=1.0, w2=1.0, max_expansions=100000)

            if path is None:
                print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
            else:
                print(f"{algo_name} expansions: {expanded}, cost: {path_cost}")
                expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

                # Since SMHA* does not track re-expansions, set to 0
                re_expansions = 0
                stats[algo_name]['total_re_expansions'] += re_expansions
                stats[algo_name]['re_expansions_per_query'].append(re_expansions)

                # **Add total expansions tracking**
                stats[algo_name]['total_expansions'] += expanded

                # Save SMHA* results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = f_map_smha

            # Model-based A* and Focal A*
            for model_name, model in models.items():
                # Model-based A*
                algo_name = f'{model_name} A*'
                print(f"Running {algo_name} on query {query + 1}...")
                model_path, model_path_cost, model_expanded, expansion_counts, f_star_map = astar_with_model(
                    start, goal, map_grid=map_data, encoder=encoder, model=model,
                    normalization_values=normalization_values, device=device, max_expansions=50000
                )

                if model_path is None:
                    print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                    continue

                expansions_diff = 100 * (traditional_expanded - model_expanded) / traditional_expanded
                cost_diff = 100 * (model_path_cost - traditional_path_cost) / traditional_path_cost

                stats[algo_name]['total_expansions_reduction'] += expansions_diff
                stats[algo_name]['total_path_cost_increase'] += cost_diff
                stats[algo_name]['path_cost_increases'].append(cost_diff)
                if abs(model_path_cost - traditional_path_cost) < 1e-6:
                    stats[algo_name]['optimal_paths'] += 1
                stats[algo_name]['total_queries'] += 1

                # **Calculate re-expansions**
                re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
                print(f"{algo_name} expansions: {model_expanded}, cost: {model_path_cost}, re-expansions: {re_expansions}")
                stats[algo_name]['total_re_expansions'] += re_expansions
                stats[algo_name]['re_expansions_per_query'].append(re_expansions)

                # **Add total expansions tracking**
                stats[algo_name]['total_expansions'] += model_expanded

                # Save Model-based A* results for visualization
                visualization_data['paths'][algo_name] = model_path
                visualization_data['f_maps'][algo_name] = f_star_map

                # Focal A* with ε=3.0
                algo_name = f'{model_name} Focal A* ε=3.0'
                print(f"Running {algo_name} on query {query + 1}...")
                focal_path, focal_path_cost, focal_expanded, focal_expansion_counts, f_star_map_focal = focal_astar_with_model(
                    start, goal, map_grid=map_data, encoder=encoder, model=model,
                    normalization_values=normalization_values, device=device, epsilon=3.0, max_expansions=50000
                )

                if focal_path is None:
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

                # **Calculate re-expansions**
                re_expansions = sum(1 for count in focal_expansion_counts.values() if count > 1)
                stats[algo_name]['total_re_expansions'] += re_expansions
                stats[algo_name]['re_expansions_per_query'].append(re_expansions)

                # **Add total expansions tracking**
                stats[algo_name]['total_expansions'] += focal_expanded

                # Save Focal A* results for visualization
                visualization_data['paths'][algo_name] = focal_path
                visualization_data['f_maps'][algo_name] = f_star_map_focal

            # Save comprehensive visualization for the current query
            visualize_comparison(
                map_grid=map_data,
                start=start,
                goal=goal,
                paths=visualization_data['paths'],
                f_maps=visualization_data['f_maps'],
                output_dir=output_dir,
                run=query_counter
            )

            query_counter += 1

            # Print cumulative results after each query
            print(f"\nCumulative Results after {query_counter} queries:")
            header = f"{'Algorithm':<30} {'Avg Exp Reduction (%)':<25} {'Path Cost Percent Increase (%)':<30} {'Avg ReExpansions':<20} {'Mean ReExpansions (%)':<25} {'Optimal Paths':<15} {'Total Queries':<15}"
            print(header)
            print("-" * len(header))
            for algo_name, data in stats.items():
                if data['total_queries'] > 0:
                    avg_exp_reduction = data['total_expansions_reduction'] / data['total_queries']
                    avg_cost_increase = data['total_path_cost_increase'] / data['total_queries']
                    std_cost_increase = np.std(data['path_cost_increases'])
                    avg_re_expansions = data['total_re_expansions'] / data['total_queries']
                    mean_reexp_percent = (data['total_re_expansions'] / data['total_expansions'] * 100) if data['total_expansions'] > 0 else 0
                    optimal_paths = data['optimal_paths']
                    total_queries = data['total_queries']
                else:
                    avg_exp_reduction = 0
                    avg_cost_increase = 0
                    std_cost_increase = 0
                    avg_re_expansions = 0
                    mean_reexp_percent = 0
                    optimal_paths = 0
                    total_queries = 0
                print(f"{algo_name:<30} {avg_exp_reduction:<25.2f} {avg_cost_increase:<30.2f} {avg_re_expansions:<20.2f} {mean_reexp_percent:<25.2f} {optimal_paths:<15} {total_queries:<15}")
            print("\n" + "-"*160 + "\n")

    # ---------------------------
    # CSV Writing Enhancement
    # ---------------------------

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'Algorithm',
            'Avg_Expansions_Reduction',
            'Path_Cost_Percent_Increase',
            'Path_Cost_Percent_STD',
            'Avg_ReExpansions',
            'Mean_ReExpansions_Percent',
            'Optimal_Paths',
            'Total_Queries'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for algo_name, data in stats.items():
            if data['total_queries'] > 0:
                avg_exp_reduction = data['total_expansions_reduction'] / data['total_queries']
                avg_cost_increase = data['total_path_cost_increase'] / data['total_queries']
                std_cost_increase = np.std(data['path_cost_increases'])
                avg_re_expansions = data['total_re_expansions'] / data['total_queries']
                mean_reexp_percent = (data['total_re_expansions'] / data['total_expansions'] * 100) if data['total_expansions'] > 0 else 0
                optimal_paths = data['optimal_paths']
                total_queries = data['total_queries']
            else:
                avg_exp_reduction = 0
                avg_cost_increase = 0
                std_cost_increase = 0
                avg_re_expansions = 0
                mean_reexp_percent = 0
                optimal_paths = 0
                total_queries = 0
            writer.writerow({
                'Algorithm': algo_name,
                'Avg_Expansions_Reduction': round(avg_exp_reduction, 2),
                'Path_Cost_Percent_Increase': round(avg_cost_increase, 2),
                'Path_Cost_Percent_STD': round(std_cost_increase, 2),
                'Avg_ReExpansions': round(avg_re_expansions, 2),
                'Mean_ReExpansions_Percent': round(mean_reexp_percent, 2),
                'Optimal_Paths': optimal_paths,
                'Total_Queries': total_queries
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
    parser.add_argument("--output_dir", type=str, default="visualizations_comp", help="Directory to save visualizations")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Encoder with latent_dim=1024
    encoder = UNet2DAutoencoder(input_channels=1, latent_dim=1024).to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    encoder.eval()

    # Initialize Models with appropriate input_size
    models = {}
    input_size = (2 * 3) + 1024 + 2  # start, goal, current (3 positions * 2 coordinates) + latent_dim=1024 + 2 heuristic values
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
