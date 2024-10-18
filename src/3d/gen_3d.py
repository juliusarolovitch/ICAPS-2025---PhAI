import numpy as np
import torch
from torch.utils.data import Dataset
import random
import heapq
from tqdm import tqdm
import argparse
import os
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass, field

from models import UNet2DAutoencoder

# ---------------------------
# Node Class for Pathfinding
# ---------------------------

class Node:
    def __init__(self, pos, g=float('inf'), h=0, parent=None):
        self.pos = pos  # (x, y, theta)
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

# ---------------------------
# Utility Functions
# ---------------------------

@dataclass
class Room:
    x: int
    y: int
    width: int
    height: int
    children: list = field(default_factory=list)

def generate_large_map(
    width=1600,
    height=1600,
    min_room_size=300,
    max_room_size=600,
    max_depth=5,
    wall_thickness=5,
    min_openings=1,
    max_openings=2,
    min_opening_size=45,
    max_opening_size=80,
    min_obstacles=1,  # Reduced number of obstacles
    max_obstacles=2,  # Reduced number of obstacles
    min_obstacle_size=100,
    max_obstacle_size=200,
    min_circle_radius=50,
    max_circle_radius=80,
    obstacle_attempts=10  # Max attempts to place an obstacle without overlap
):
    """
    Generates a large 2D map with rooms and walls with openings between 45 and 80 units wide.
    Adds both rectangular and circular obstacles without overlapping.

    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        min_room_size (int): Minimum size of a room.
        max_room_size (int): Maximum size of a room.
        max_depth (int): Maximum recursion depth for splitting rooms.
        wall_thickness (int): Thickness of the walls between rooms.
        min_openings (int): Minimum number of openings per wall.
        max_openings (int): Maximum number of openings per wall.
        min_opening_size (int): Minimum size of each opening in units.
        max_opening_size (int): Maximum size of each opening in units.
        min_obstacles (int): Minimum number of obstacles per room.
        max_obstacles (int): Maximum number of obstacles per room.
        min_obstacle_size (int): Minimum size (width/height) of each rectangular obstacle.
        max_obstacle_size (int): Maximum size (width/height) of each rectangular obstacle.
        min_circle_radius (int): Minimum radius of circular obstacles.
        max_circle_radius (int): Maximum radius of circular obstacles.
        obstacle_attempts (int): Number of attempts to place an obstacle without overlap.

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
            possible_positions = wall_length - min_opening_size
            if possible_positions <= 0:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                if opening_size > wall_length:
                    opening_size = wall_length
                opening_start = random.randint(start[1], start[1] + wall_length - opening_size)
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        else:
            wall_length = end[0] - start[0]
            possible_positions = wall_length - min_opening_size
            if possible_positions <= 0:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                if opening_size > wall_length:
                    opening_size = wall_length
                opening_start = random.randint(start[0], start[0] + wall_length - opening_size)
                map_grid[opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness] = 0
    
    # Start splitting from the root room
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
            obstacle_type = random.choice(['rectangle', 'circle'])
            placed = False
            for attempt in range(obstacle_attempts):
                if obstacle_type == 'rectangle':
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
                else:  # 'circle'
                    radius = random.randint(min_circle_radius, max_circle_radius)
                    # Ensure circle fits within the room with some padding
                    if (radius * 2) >= min(room.width, room.height) - 2 * wall_thickness:
                        continue  # Skip if circle is too big for the room
                    obstacle_x = random.randint(room.x + wall_thickness + radius, room.x + room.width - wall_thickness - radius)
                    obstacle_y = random.randint(room.y + wall_thickness + radius, room.y + room.height - wall_thickness - radius)
                    # Create a circular mask
                    y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
                    mask = x_grid**2 + y_grid**2 <= radius**2
                    x_start = obstacle_x - radius
                    y_start = obstacle_y - radius
                    x_end = obstacle_x + radius + 1
                    y_end = obstacle_y + radius + 1
                    # Check bounds
                    if x_start < 0 or y_start < 0 or x_end > width or y_end > height:
                        continue  # Out of bounds
                    # Check overlap
                    existing = map_grid[y_start:y_end, x_start:x_end]
                    if np.any(existing[mask] == 1):
                        continue  # Overlaps with existing obstacle
                    # Place the circular obstacle
                    existing[mask] = 1
                    placed = True
                    break  # Successfully placed
            if not placed:
                print(f"Could not place a {obstacle_type} obstacle in room at ({room.x}, {room.y}) after {obstacle_attempts} attempts.")
                continue  # Skip if unable to place after attempts
    
    # Add outer boundary walls
    # Top and bottom
    map_grid[0:wall_thickness, :] = 1
    map_grid[-wall_thickness:, :] = 1
    # Left and right
    map_grid[:, 0:wall_thickness] = 1
    map_grid[:, -wall_thickness:] = 1
    
    return map_grid

# Robot parameters
ROBOT_WIDTH = 52
ROBOT_HEIGHT = 77
ORIENTATIONS = [i for i in range(0, 360, 45)]  # 0 to 315 degrees in steps of 45

def is_valid(pos, theta, map_grid):
    """
    Checks if the robot at position pos with orientation theta does not collide with obstacles.

    Args:
        pos (tuple): (x, y) position of the robot's center.
        theta (int): Orientation angle in degrees.
        map_grid (np.ndarray): The occupancy grid map.

    Returns:
        bool: True if the robot's footprint is collision-free, False otherwise.
    """
    x, y = pos
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    
    # Compute the corners of the robot
    half_w, half_h = ROBOT_WIDTH / 2, ROBOT_HEIGHT / 2
    corners = np.array([
        [-half_w, -half_h],
        [-half_w, half_h],
        [half_w, half_h],
        [half_w, -half_h]
    ])
    # Rotate corners
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    rotated_corners = corners @ rotation_matrix.T
    # Translate to position
    robot_corners = rotated_corners + np.array([x, y])
    
    # Create a mask of the robot's area
    min_x = int(np.floor(np.min(robot_corners[:, 0])))
    max_x = int(np.ceil(np.max(robot_corners[:, 0])))
    min_y = int(np.floor(np.min(robot_corners[:, 1])))
    max_y = int(np.ceil(np.max(robot_corners[:, 1])))
    
    # Check bounds
    if min_x < 0 or min_y < 0 or max_x >= map_grid.shape[1] or max_y >= map_grid.shape[0]:
        return False  # Out of bounds
    
    # Check for collision
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(x_range, y_range)
    points = np.vstack((xv.flatten(), yv.flatten())).T

    # Transform points back to robot coordinate frame
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    local_points = points - np.array([x, y])
    local_points = local_points @ inv_rotation_matrix.T

    # Check if points are inside the robot's rectangle
    inside_mask = (np.abs(local_points[:, 0]) <= half_w) & (np.abs(local_points[:, 1]) <= half_h)

    # Get map values at these points
    collision_points = points[inside_mask]
    map_indices = (collision_points[:, 1].astype(int), collision_points[:, 0].astype(int))
    map_values = map_grid[map_indices]

    if np.any(map_values == 1):
        return False  # Collision detected

    return True  # No collision

def generate_random_configuration(map_grid):
    """
    Generates a random valid robot configuration (x, y, theta).

    Returns:
        tuple: (x, y, theta)
    """
    while True:
        x = random.randint(ROBOT_WIDTH // 2, map_grid.shape[1] - ROBOT_WIDTH // 2 - 1)
        y = random.randint(ROBOT_HEIGHT // 2, map_grid.shape[0] - ROBOT_HEIGHT // 2 - 1)
        theta = random.choice(ORIENTATIONS)
        if is_valid((x, y), theta, map_grid):
            return (x, y, theta)

def euclidean_distance_3d(a, b):
    """
    Computes the Euclidean distance between two configurations, considering orientation.

    Args:
        a (tuple): (x, y, theta)
        b (tuple): (x, y, theta)

    Returns:
        float: The Euclidean distance.
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dtheta = min(abs(a[2] - b[2]), 360 - abs(a[2] - b[2])) / 45  # Number of 45-degree rotations
    return np.sqrt(dx**2 + dy**2) + dtheta  # Simple heuristic combining distance and rotations

def get_neighbors(state, map_grid):
    """
    Gets the neighboring states for a given state, considering movement and rotation.

    Args:
        state (tuple): (x, y, theta)
        map_grid (np.ndarray): The occupancy grid map.

    Returns:
        list: List of tuples ((new_x, new_y, new_theta), cost)
    """
    x, y, theta = state
    neighbors = []

    # Rotation actions (must be stationary)
    for dtheta in [-45, 45]:
        new_theta = (theta + dtheta) % 360
        if is_valid((x, y), new_theta, map_grid):
            neighbors.append(((x, y, new_theta), ROTATION_COST))

    # Movement actions (only if not rotating)
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),        (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]:
        new_x = x + dx
        new_y = y + dy
        new_theta = theta  # Orientation remains the same
        if is_valid((new_x, new_y), new_theta, map_grid):
            if dx != 0 and dy != 0:
                move_cost = DIAGONAL_MOVE_COST
            else:
                move_cost = ORTHOGONAL_MOVE_COST
            neighbors.append(((new_x, new_y, new_theta), move_cost))

    return neighbors

# Movement costs
ORTHOGONAL_MOVE_COST = 1.0
DIAGONAL_MOVE_COST = np.sqrt(2)
ROTATION_COST = 0.5  # Cost for rotating by 45 degrees

def astar(start, goal, map_grid):
    """
    A* algorithm for pathfinding with orientation.

    Args:
        start (tuple): (x, y, theta)
        goal (tuple): (x, y, theta)
        map_grid (np.ndarray): The occupancy grid map.

    Returns:
        dict: Dictionary of g-values for each state.
    """
    start_node = Node(start, g=0, h=euclidean_distance_3d(start, goal))
    open_list = [start_node]
    closed_set = set()
    g_values = {}

    g_values[start] = 0

    while open_list:
        current = heapq.heappop(open_list)

        if current.pos == goal:
            return g_values

        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)

        for neighbor_state, cost in get_neighbors(current.pos, map_grid):
            if neighbor_state in closed_set:
                continue

            new_g = current.g + cost
            new_h = euclidean_distance_3d(neighbor_state, goal)
            neighbor_node = Node(neighbor_state, g=new_g, h=new_h, parent=current)

            if neighbor_state not in g_values or new_g < g_values[neighbor_state]:
                g_values[neighbor_state] = new_g
                heapq.heappush(open_list, neighbor_node)

    return None  # Path not found

def distance_to_nearest_obstacle(pos, map_grid):
    obstacle_positions = np.argwhere(map_grid == 1)
    if obstacle_positions.size == 0:
        return np.inf  
    distances = np.sqrt(np.sum((obstacle_positions - pos)**2, axis=1))
    return np.min(distances)

def generate_start_goal_biased(map_grid, bias_factor=0.05):
    size = map_grid.shape[0]
    all_positions = [(x, y) for x in range(size)
                     for y in range(size) if map_grid[y, x] == 0]

    if not all_positions:
        raise ValueError("No valid positions found in the map")

    distances = np.array([distance_to_nearest_obstacle(pos, map_grid) for pos in all_positions])
    probabilities = np.exp(-bias_factor * distances)
    probabilities /= np.sum(probabilities)

    while True:
        start = all_positions[np.random.choice(len(all_positions), p=probabilities)]
        goal = all_positions[np.random.choice(len(all_positions), p=probabilities)]

        if start != goal:
            return start, goal

def save_fstar_visualization(map_grid, start, goal, f_star, map_idx, f_star_dir, dataset_type):
    plt.figure(figsize=(10, 10))
    plt.imshow(map_grid, cmap='binary')

    valid_mask = np.isfinite(f_star)

    scatter = plt.scatter(np.where(valid_mask)[1], np.where(valid_mask)[0],
                          c=f_star[valid_mask], cmap='viridis',
                          s=20, alpha=0.7)

    plt.colorbar(scatter, label="f* values")

    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    plt.title(f"F* values for Map {map_idx + 1} ({dataset_type})")
    plt.legend()

    save_path = os.path.join(f_star_dir, f"fstar_map_{map_idx + 1}_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Data Generation Function
# ---------------------------

def generate_dataset(num_maps, num_queries_per_map, encoder, device, height=1600, width=1600, map_save_dir="maps", f_star_dir="f_star_maps_100"):
    # Initialize separate datasets
    datasets = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    all_g_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    all_h_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    all_f_star_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }

    # Create directories for f_star visualizations for each dataset
    for dataset_type in ['vanilla', 'exp', 'mult']:
        f_star_dataset_dir = os.path.join(f_star_dir, dataset_type)
        if not os.path.exists(f_star_dataset_dir):
            os.makedirs(f_star_dataset_dir)

    if not os.path.exists(map_save_dir):
        os.makedirs(map_save_dir)

    for map_idx in tqdm(range(num_maps), desc="Generating dataset"):
        map_grid = generate_large_map(width=width, height=height)

        # Save the map data as .npz
        map_file_path = os.path.join(map_save_dir, f"map_{map_idx+1}.npz")
        np.savez(map_file_path, map_grid=map_grid)

        with torch.no_grad():
            map_tensor = torch.from_numpy(
                map_grid).float().unsqueeze(0).unsqueeze(0).to(device)
            encoded_map = encoder.get_latent_vector(map_tensor).cpu().numpy().flatten()

        first_query = True
        for query_idx in range(num_queries_per_map):
            astar_res = None
            while astar_res is None:
                start = generate_random_configuration(map_grid)
                goal = generate_random_configuration(map_grid)
                if start == goal:
                    continue
                astar_res = astar(start, goal, map_grid)
                if astar_res is None:
                    print(f"No path found for map {map_idx+1}, query {query_idx+1}. Retrying...")

            # f_star calculation should consider both forward and backward paths if needed
            # Here, we assume f_star is the cost from start to all reachable nodes
            # and possibly f_star from goal as well for bidirectional approaches
            # Adjust accordingly based on your specific requirements

            # For simplicity, we'll use g_values as the cost from start to each node
            f_star = astar_res  # Assuming f_star is similar to g_values

            c_star = f_star[goal]

            # Prepare different versions of f_star
            # Vanilla: No penalty
            vanilla_f_star = f_star.copy()

            # Exponential penalty
            finite_mask = np.isfinite(f_star)
            exp_penalty_factor = np.exp((f_star[finite_mask] - c_star) / c_star)
            exp_f_star = f_star.copy()
            exp_f_star[finite_mask] *= exp_penalty_factor

            # Multiplicative penalty
            mult_penalty_factor = (f_star[finite_mask] - c_star) / c_star
            mult_f_star = f_star.copy()
            mult_f_star[finite_mask] *= (1 + mult_penalty_factor)

            # Dictionary to iterate through each dataset type
            f_star_versions = {
                'vanilla': vanilla_f_star,
                'exp': exp_f_star,
                'mult': mult_f_star
            }

            for dataset_type, modified_f_star in f_star_versions.items():
                if first_query:
                    save_fstar_visualization(
                        map_grid, start, goal, modified_f_star, map_idx, f_star_dir, dataset_type)
                first_query = False

                valid_positions = np.argwhere(np.isfinite(modified_f_star))
                for pos in valid_positions:
                    x, y = pos
                    g_star = f_star[x, y]
                    # Since the robot has orientation, compute h accordingly
                    # Here, assuming h is distance from (x, y) to goal ignoring theta
                    h = euclidean_distance_3d((x, y, 0), (goal[0], goal[1], 0))  # Theta can be adjusted if needed
                    f_star_value = modified_f_star[x, y]

                    if np.isinf(g_star) or np.isinf(h) or np.isinf(f_star_value):
                        continue

                    target_value = f_star_value

                    datasets[dataset_type].append((encoded_map, start, goal,
                                                   (x, y), g_star, h, target_value))
                    all_g_values[dataset_type].append(g_star)
                    all_h_values[dataset_type].append(h)
                    all_f_star_values[dataset_type].append(target_value)

    # Calculate normalization values and normalize datasets
    normalized_datasets = {}
    normalization_values = {}
    for dataset_type in ['vanilla', 'exp', 'mult']:
        print(f"Normalizing {dataset_type} dataset...")
        g_min, g_max = np.min(all_g_values[dataset_type]), np.max(all_g_values[dataset_type])
        h_min, h_max = np.min(all_h_values[dataset_type]), np.max(all_h_values[dataset_type])
        f_star_min, f_star_max = np.min(all_f_star_values[dataset_type]), np.max(all_f_star_values[dataset_type])

        normalized_dataset = []
        for encoded_map, start, goal, current, g_star, h, target_value in datasets[dataset_type]:
            g_normalized = (g_star - g_min) / (g_max - g_min) if g_max != g_min else 0.0
            h_normalized = (h - h_min) / (h_max - h_min) if h_max != h_min else 0.0
            target_normalized = (target_value - f_star_min) / (f_star_max - f_star_min) if f_star_max != f_star_min else 0.0

            if np.isfinite(g_normalized) and np.isfinite(h_normalized) and np.isfinite(target_normalized):
                normalized_dataset.append(
                    (encoded_map, start, goal, current, g_normalized, h_normalized, target_normalized))

        normalized_datasets[dataset_type] = normalized_dataset
        normalization_values[dataset_type] = {
            'f_star_min': f_star_min, 'f_star_max': f_star_max,
            'g_min': g_min, 'g_max': g_max,
            'h_min': h_min, 'h_max': h_max
        }

    return normalized_datasets, normalization_values

# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create Datasets for F* Prediction Model with Different Penalties")
    parser.add_argument("encoder_path", type=str,
                        help="Path to the pre-trained UNet2DAutoencoder model")
    parser.add_argument("--num_maps", type=int, default=5,
                        help="Number of maps to generate")
    parser.add_argument("--num_queries_per_map", type=int,
                        default=100, help="Number of queries per map")
    parser.add_argument("--height", type=int, default=1600,
                        help="Height of the map")
    parser.add_argument("--width", type=int, default=1600,
                        help="Width of the map")
    parser.add_argument("--map_save_dir", type=str, default="maps/2d_data",
                        help="Directory to save the generated maps")
    parser.add_argument("--f_star_dir", type=str, default="f_star_maps_100",
                        help="Directory to save F* visualizations")
    parser.add_argument("--save_dataset_dir", type=str, default="datasets",
                        help="Directory to save the generated datasets")
    parser.add_argument("--norm_save_dir", type=str, default="normalization_values",
                        help="Directory to save the normalization values")
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------

def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize UNet2DAutoencoder as Encoder
    print("Initializing UNet2DAutoencoder encoder...")
    encoder = UNet2DAutoencoder(input_channels=1, latent_dim=512).to(device)  

    # Load the pre-trained encoder model
    if not os.path.isfile(args.encoder_path):
        raise FileNotFoundError(f"Encoder model not found at {args.encoder_path}")

    try:
        encoder.load_state_dict(torch.load(
            args.encoder_path, map_location=device))
        print(f"Loaded encoder model from {args.encoder_path}")
    except Exception as e:
        print(f"Error loading encoder model: {e}")
        return

    encoder.eval()

    print("Generating new datasets and calculating normalization values")

    dataset_dict, normalization_values_dict = generate_dataset(
        args.num_maps, args.num_queries_per_map, encoder, device,
        height=args.height, width=args.width,
        map_save_dir=args.map_save_dir, f_star_dir=args.f_star_dir
    )

    if not os.path.exists(args.save_dataset_dir):
        os.makedirs(args.save_dataset_dir)
    if not os.path.exists(args.norm_save_dir):
        os.makedirs(args.norm_save_dir)

    # Save each dataset and its normalization values separately
    for dataset_type in ['vanilla', 'exp', 'mult']:
        dataset = dataset_dict[dataset_type]
        normalization_values = normalization_values_dict[dataset_type]

        # Save dataset
        dataset_save_path = os.path.join(args.save_dataset_dir, f"{dataset_type}_dataset.pkl")
        with open(dataset_save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"{dataset_type.capitalize()} dataset saved to {dataset_save_path}")

        # Save normalization values
        norm_save_path = os.path.join(args.norm_save_dir, f"{dataset_type}_normalization_values.pkl")
        with open(norm_save_path, 'wb') as f:
            pickle.dump(normalization_values, f)
        print(f"{dataset_type.capitalize()} normalization values saved to {norm_save_path}")

    print("All datasets created and saved successfully.")

if __name__ == '__main__':
    main()
