import numpy as np
import torch
import random
import heapq
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # For 3D visualization
from scipy.ndimage import distance_transform_edt

def add_costs_around_obstacles(map_grid, obstacle_cost=10, max_distance=5):
    """
    Adds cost penalties around obstacles to the cost map.

    Args:
        map_grid (np.ndarray): 3D occupancy grid where 1 indicates obstacles.
        obstacle_cost (float): Base cost to add around obstacles.
        max_distance (int): Maximum distance from obstacles to apply penalties.

    Returns:
        np.ndarray: Cost map with added penalties around obstacles.
    """
    # Initialize the cost map with zeros
    cost_map = np.zeros_like(map_grid, dtype=np.float32)
    
    # Invert the map_grid: obstacles are True
    obstacles = map_grid == 1
    
    # Compute the Euclidean distance from each free cell to the nearest obstacle
    distance = distance_transform_edt(~obstacles)
    
    # Apply cost penalties within max_distance from obstacles
    penalty_mask = (distance > 0) & (distance <= max_distance)
    
    # Linear penalty: closer to obstacle, higher the cost
    cost_map[penalty_mask] = obstacle_cost * (1 - (distance[penalty_mask] / max_distance))
    
    return cost_map


# ---------------------------
# Import PointNet++ Autoencoder
# ---------------------------
from pp import PointNetAutoencoder  # Ensure this is correctly imported

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize global variables to None
args = None
device = 'cpu'


# ---------------------------
# Node Class for A*
# ---------------------------
@dataclass(order=True)
class Node:
    f: float
    pos: tuple = field(compare=False)
    g: float = field(compare=False)
    h: float = field(compare=False)
    parent: tuple = field(compare=False, default=None)


# ---------------------------
# Utility Functions
# ---------------------------


@dataclass
class Room:
    x: int
    y: int
    z: int
    width: int
    height: int
    depth: int
    children: list = field(default_factory=list)


def generate_map(
    width=100,
    height=100,
    depth=40,
    min_room_size=20,
    max_room_size=40,
    max_depth=5,
    wall_thickness=2,
    min_openings=1,
    max_openings=2,
    min_opening_size=4,
    max_opening_size=10,
    min_obstacles=5,
    max_obstacles=8,
    min_obstacle_size=7,
    max_obstacle_size=10,
    obstacle_attempts=10,
    trap_probability=0.4
):
    """
    Generates a 3D map with randomly sized and positioned rooms separated by walls with random openings.
    Each room contains a random number of smaller obstacles.

    Args:
        width (int): Width of the map along the x-axis.
        height (int): Height of the map along the y-axis.
        depth (int): Depth of the map along the z-axis.
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
        min_obstacle_size (int): Minimum size of each obstacle.
        max_obstacle_size (int): Maximum size of each obstacle.
        obstacle_attempts (int): Number of attempts to place an obstacle without overlap.
        trap_probability (float): Probability of placing a concave trap instead of a regular obstacle.

    Returns:
        np.ndarray: 3D occupancy map of shape [depth, height, width].
    """
    map_grid = np.zeros((depth, height, width), dtype=np.float32)

    root_room = Room(0, 0, 0, width, height, depth)

    def split_room(room, depth_level):
        if depth_level >= max_depth:
            return
        can_split_horizontally = room.height >= 2 * min_room_size + wall_thickness
        can_split_vertically = room.width >= 2 * min_room_size + wall_thickness
        can_split_depth = room.depth >= 2 * min_room_size + wall_thickness

        if not (can_split_horizontally or can_split_vertically or can_split_depth):
            return  # Cannot split further

        split_options = []
        if can_split_horizontally:
            split_options.append('horizontal')
        if can_split_vertically:
            split_options.append('vertical')
        if can_split_depth:
            split_options.append('depth')

        split_orientation = random.choice(split_options)

        if split_orientation == 'horizontal':
            split_min = room.y + min_room_size
            split_max = room.y + room.height - min_room_size - wall_thickness
            if split_max <= split_min:
                return
            split_pos = random.randint(split_min, split_max)
            child1 = Room(room.x, room.y, room.z, room.width, split_pos - room.y, room.depth)
            child2 = Room(room.x, split_pos + wall_thickness, room.z, room.width,
                          room.y + room.height - split_pos - wall_thickness, room.depth)
            map_grid[:, split_pos:split_pos + wall_thickness, room.x:room.x + room.width] = 1
            add_openings((split_pos, room.x, room.z), (split_pos, room.x + room.width, room.z), orientation='horizontal')
        elif split_orientation == 'vertical':
            split_min = room.x + min_room_size
            split_max = room.x + room.width - min_room_size - wall_thickness
            if split_max <= split_min:
                return
            split_pos = random.randint(split_min, split_max)
            child1 = Room(room.x, room.y, room.z, split_pos - room.x, room.height, room.depth)
            child2 = Room(split_pos + wall_thickness, room.y, room.z, room.x + room.width - split_pos - wall_thickness,
                          room.height, room.depth)
            map_grid[:, room.y:room.y + room.height, split_pos:split_pos + wall_thickness] = 1
            add_openings((room.y, split_pos, room.z), (room.y + room.height, split_pos, room.z), orientation='vertical')
        else:  # depth
            split_min = room.z + min_room_size
            split_max = room.z + room.depth - min_room_size - wall_thickness
            if split_max <= split_min:
                return
            split_pos = random.randint(split_min, split_max)
            child1 = Room(room.x, room.y, room.z, room.width, room.height, split_pos - room.z)
            child2 = Room(room.x, room.y, split_pos + wall_thickness, room.width,
                          room.height, room.depth - split_pos - wall_thickness)
            map_grid[split_pos:split_pos + wall_thickness, room.y:room.y + room.height, room.x:room.x + room.width] = 1
            add_openings((room.y, room.x, split_pos), (room.y + room.height, room.x + room.width, split_pos), orientation='depth')

        room.children = [child1, child2]
        split_room(child1, depth_level + 1)
        split_room(child2, depth_level + 1)

    def add_openings(start, end, orientation='horizontal'):
        """
        Adds random openings to a wall.

        Args:
            start (tuple): Starting coordinate (y, x, z).
            end (tuple): Ending coordinate (y, x, z).
            orientation (str): 'horizontal', 'vertical', or 'depth'.
        """
        num_openings = random.randint(min_openings, max_openings)
        if orientation == 'horizontal':
            wall_length = end[1] - start[1]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)
                opening_start = random.randint(start[1], end[1] - opening_size)
                z = start[2]
                map_grid[:, start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        elif orientation == 'vertical':
            wall_length = end[0] - start[0]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)
                opening_start = random.randint(start[0], end[0] - opening_size)
                z = start[2]
                map_grid[:, opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness] = 0
        else:  # depth
            wall_length = end[2] - start[2]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)
                opening_start = random.randint(start[2], end[2] - opening_size)
                y = start[0]
                x = start[1]
                map_grid[opening_start:opening_start + opening_size, y:y + wall_thickness, x:x + wall_thickness] = 0

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
        trap_thickness = wall_thickness

        if (trap_size * 2 + wall_thickness) > room.width or \
           (trap_size * 2 + wall_thickness) > room.height or \
           (trap_size * 2 + wall_thickness) > room.depth:
            return False

        corner_x = random.randint(room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
        corner_y = random.randint(room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)
        corner_z = random.randint(room.z + wall_thickness, room.z + room.depth - trap_size - wall_thickness)

        orientation = random.choice(['left', 'right', 'up', 'down', 'forward', 'backward'])

        if orientation == 'left':
            arm1 = ((corner_y, corner_x - trap_size, corner_z), (trap_size, trap_thickness, trap_size))
            arm2 = ((corner_y - trap_size, corner_x, corner_z), (trap_thickness, trap_size, trap_size))
        elif orientation == 'right':
            arm1 = ((corner_y, corner_x, corner_z), (trap_size, trap_thickness, trap_size))
            arm2 = ((corner_y - trap_size, corner_x + trap_size - trap_thickness, corner_z), (trap_thickness, trap_size, trap_size))
        elif orientation == 'up':
            arm1 = ((corner_y - trap_size, corner_x, corner_z), (trap_thickness, trap_size, trap_size))
            arm2 = ((corner_y - trap_size, corner_x - trap_size, corner_z), (trap_size, trap_thickness, trap_size))
        elif orientation == 'down':
            arm1 = ((corner_y, corner_x, corner_z), (trap_thickness, trap_size, trap_size))
            arm2 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness, corner_z),
                    (trap_size, trap_thickness, trap_size))
        elif orientation == 'forward':
            arm1 = ((corner_y, corner_x, corner_z - trap_size), (trap_size, trap_size, trap_thickness))
            arm2 = ((corner_y, corner_x, corner_z), (trap_size, trap_size, trap_thickness))
        else:  # 'backward'
            arm1 = ((corner_y, corner_x, corner_z), (trap_size, trap_size, trap_thickness))
            arm2 = ((corner_y, corner_x, corner_z + trap_size - trap_thickness),
                    (trap_size, trap_size, trap_thickness))

        (y1, x1, z1), (h1, w1, d1) = arm1
        (y2, x2, z2), (h2, w2, d2) = arm2

        # Boundary checks
        if (x1 < 0 or y1 < 0 or z1 < 0 or
            x1 + w1 > width or y1 + h1 > height or z1 + d1 > depth or
            x2 < 0 or y2 < 0 or z2 < 0 or
            x2 + w2 > width or y2 + h2 > height or z2 + d2 > depth):
            return False

        # Overlap checks
        if (np.any(map_grid[z1:z1 + d1, y1:y1 + h1, x1:x1 + w1] == 1) or
            np.any(map_grid[z2:z2 + d2, y2:y2 + h2, x2:x2 + w2] == 1)):
            return False

        # Place the trap
        map_grid[z1:z1 + d1, y1:y1 + h1, x1:x1 + w1] = 1
        map_grid[z2:z2 + d2, y2:y2 + h2, x2:x2 + w2] = 1

        return True

    def place_triangular_trap(room):
        """
        Places a triangular concave trap within the given room.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        # Placeholder for triangular trap implementation
        return False  # Implement triangular traps if needed

    split_room(root_room, 0)

    leaf_rooms = []

    def collect_leaf_rooms(room):
        if not room.children:
            leaf_rooms.append(room)
        else:
            for child in room.children:
                collect_leaf_rooms(child)

    collect_leaf_rooms(root_room)

    for room in leaf_rooms:
        num_obstacles = random.randint(min_obstacles, max_obstacles)
        for _ in range(num_obstacles):
            if random.random() < trap_probability:
                placed = False
                for attempt in range(obstacle_attempts):
                    if place_concave_trap(room):
                        placed = True
                        break
                if not placed:
                    pass
                continue
            else:
                placed = False
                for attempt in range(obstacle_attempts):
                    obstacle_w = random.randint(min_obstacle_size, max_obstacle_size)
                    obstacle_h = random.randint(min_obstacle_size, max_obstacle_size)
                    obstacle_d = random.randint(min_obstacle_size, max_obstacle_size)
                    if (obstacle_w >= room.width - 2 * wall_thickness or
                        obstacle_h >= room.height - 2 * wall_thickness or
                        obstacle_d >= room.depth - 2 * wall_thickness):
                        continue  # Skip if obstacle is too big for the room
                    obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
                    obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
                    obstacle_z = random.randint(room.z + wall_thickness, room.z + room.depth - obstacle_d - wall_thickness)
                    if np.any(map_grid[obstacle_z:obstacle_z + obstacle_d,
                                       obstacle_y:obstacle_y + obstacle_h,
                                       obstacle_x:obstacle_x + obstacle_w] == 1):
                        continue  # Overlaps with existing obstacle
                    map_grid[obstacle_z:obstacle_z + obstacle_d,
                             obstacle_y:obstacle_y + obstacle_h,
                             obstacle_x:obstacle_x + obstacle_w] = 1
                    placed = True
                    break  # Successfully placed
                if not placed:
                    pass

    # Optionally, add outer boundary walls
    map_grid[:, 0:wall_thickness, :] = 1
    map_grid[:, -wall_thickness:, :] = 1
    map_grid[:, :, 0:wall_thickness] = 1
    map_grid[:, :, -wall_thickness:] = 1

    return map_grid


def is_valid(pos, map_grid):
    return (0 <= pos[0] < map_grid.shape[0] and
            0 <= pos[1] < map_grid.shape[1] and
            0 <= pos[2] < map_grid.shape[2] and
            map_grid[pos] == 0)


def generate_start_goal(map_grid):
    depth, height, width = map_grid.shape
    while True:
        start = (random.randint(0, depth - 1), random.randint(0, height - 1), random.randint(0, width - 1))
        goal = (random.randint(0, depth - 1), random.randint(0, height - 1), random.randint(0, width - 1))
        if (is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal):
            return start, goal


def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def astar(start, goal, map_grid, cost_map):
    start_node = Node(f=euclidean_distance(start, goal), pos=start, g=0, h=euclidean_distance(start, goal))
    open_list = [start_node]
    closed_set = set()
    g_values = np.full(map_grid.shape, float('inf'))
    g_values[start] = 0
    expansions = 0

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions>400000:
            logger.warning("A* expanded more than max allowable nodes")
            return 

        if current.pos == goal:
            return g_values

        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)

        for next_pos, move_cost in get_neighbors(current.pos, map_grid):
            if not is_valid(next_pos, map_grid):
                continue

            new_g = current.g + move_cost + cost_map[next_pos]
            new_h = euclidean_distance(next_pos, goal)

            if new_g < g_values[next_pos]:
                g_values[next_pos] = new_g
                f = new_g + new_h
                next_node = Node(f=f, pos=next_pos, g=new_g, h=new_h)
                heapq.heappush(open_list, next_node)

    return None


def get_neighbors(pos, map_grid):
    x, y, z = pos
    # 26 possible neighbors in 3D
    directions = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    neighbors = []
    for dx, dy, dz in directions:
        neighbor = (x + dx, y + dy, z + dz)
        if is_valid(neighbor, map_grid):
            # Diagonal moves have higher cost
            cost = np.sqrt(dx**2 + dy**2 + dz**2)
            neighbors.append((neighbor, cost))
    return neighbors


def distance_to_nearest_obstacle(pos, map_grid):
    obstacle_positions = np.argwhere(map_grid == 1)
    if obstacle_positions.size == 0:
        return np.inf
    distances = np.sqrt(np.sum((obstacle_positions - pos)**2, axis=1))
    return np.min(distances)


def generate_start_goal_biased(map_grid):
    """
    Generates random start and goal positions on the map without any bias.

        map_grid (np.ndarray): 3D occupancy map where 0 indicates free space.

        tuple: (start, goal) positions as (x, y, z) tuples.

        ValueError: If no valid positions are found on the map.
    """
    depth, height, width = map_grid.shape
    all_positions = [(x, y, z) for x in range(depth)
                     for y in range(height)
                     for z in range(width) if map_grid[x, y, z] == 0]

    if not all_positions:
        raise ValueError("No valid positions found in the map")

    # Randomly sample two distinct positions for start and goal
    start, goal = random.sample(all_positions, 2)
    return start, goal


def save_fstar_visualization(map_grid, start, goal, f_star, map_idx, f_star_dir, dataset_type):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a voxel grid for visualization
    ax.voxels(map_grid, facecolors='blue', edgecolor='k', alpha=0.3)

    # Plot f_star values as a scatter plot
    valid_mask = np.isfinite(f_star)
    x, y, z = np.where(valid_mask)
    scatter = ax.scatter(x, y, z, c=f_star[valid_mask], cmap='viridis', s=1, alpha=0.6)

    plt.colorbar(scatter, label="f* values")

    # Plot start and goal
    ax.scatter(start[0], start[1], start[2], c='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='red', s=100, label='Goal')

    ax.set_title(f"F* values for Map {map_idx + 1} ({dataset_type})")
    ax.legend()

    save_path = os.path.join(f_star_dir, dataset_type,
                             f"fstar_map_{map_idx + 1}_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_map(map_idx, encoded_map, map_grid, args, dataset_types):
    """
    Processes a single map by performing A* queries and writes the datasets to disk.

    Args:
        map_idx (int): Index of the map being processed.
        encoded_map (np.ndarray): Encoded representation of the map.
        map_grid (np.ndarray): 3D occupancy grid of the map.
        args (Namespace): Command-line arguments.
        dataset_types (list): List of dataset types to generate.

    Returns:
        tuple: (map_idx, success_flag)
    """
    try:
        # Initialize datasets
        datasets = {dataset_type: [] for dataset_type in dataset_types}

        # Add cost penalties around obstacles
        cost_map = add_costs_around_obstacles(map_grid)

        first_query = True
        for query_idx in tqdm(range(args.num_queries_per_map), desc=f"Generating data for map {map_idx + 1}", leave=False):
            astar_res = False
            attempt_counter = 0

            while not astar_res and attempt_counter < 5:
                attempt_counter += 1
                try:
                    start, goal = generate_start_goal(map_grid)
                except Exception as e:
                    logger.error(
                        f"Map {map_idx + 1}: Error generating start/goal for query {query_idx + 1}: {e}")
                    continue

                forward_g = astar(start, goal, map_grid, cost_map)
                if forward_g is None:
                    logger.warning(
                        f"Map {map_idx + 1}: Forward A* failed for query {query_idx + 1}")
                    continue

                backward_g = astar(goal, start, map_grid, cost_map)
                if backward_g is None:
                    logger.warning(
                        f"Map {map_idx + 1}: Backward A* failed for query {query_idx + 1}")
                    continue

                astar_res = True

            if not astar_res:
                logger.warning(
                    f"Map {map_idx + 1}: Could not perform A* for query {query_idx + 1} after {attempt_counter} attempts.")
                continue  # Skip this query

            # Compute f_star
            f_star = forward_g + backward_g
            c_star = f_star[goal]

            # Vanilla: No penalty
            vanilla_f_star = f_star.copy()

            # Exponential penalty
            exp_f_star = None
            if 'exp' in dataset_types:
                finite_mask = np.isfinite(f_star)
                exp_penalty_factor = np.exp(
                    (f_star[finite_mask] - c_star) / c_star)
                exp_f_star = f_star.copy()
                exp_f_star[finite_mask] *= exp_penalty_factor

            # Multiplicative penalty
            mult_f_star = None
            if 'mult' in dataset_types:
                finite_mask = np.isfinite(f_star)
                mult_penalty_factor = (f_star[finite_mask] - c_star) / c_star
                mult_f_star = f_star.copy()
                mult_f_star[finite_mask] *= (1 + mult_penalty_factor)

            # Visualization for the first query of the first few maps
            if first_query and map_idx < 5:
                f_star_dir = args.f_star_dir
                for dataset_type in dataset_types:
                    if dataset_type == 'vanilla':
                        current_f_star = vanilla_f_star
                    elif dataset_type == 'exp':
                        current_f_star = exp_f_star
                    elif dataset_type == 'mult':
                        current_f_star = mult_f_star
                    else:
                        continue
                    save_fstar_visualization(
                        map_grid, start, goal, 
                        current_f_star, 
                        map_idx, f_star_dir, dataset_type)
            first_query = False

            # Collect valid positions
            valid_positions = np.argwhere(np.isfinite(f_star))

            # Limit the number of samples to 50,000 to manage memory
            if len(valid_positions) > 5000:
                sampled_indices = random.sample(range(len(valid_positions)), 5000)
                sampled_positions = valid_positions[sampled_indices]
            else:
                sampled_positions = valid_positions

            for pos in sampled_positions:
                x, y, z = pos
                g_star_val = forward_g[x, y, z]
                h = euclidean_distance((x, y, z), goal)
                f_star_value = f_star[x, y, z]

                if np.isinf(g_star_val) or np.isinf(h) or np.isinf(f_star_value):
                    continue

                target_value = f_star_value

                for dataset_type in dataset_types:
                    if dataset_type == 'vanilla':
                        modified_f_star = vanilla_f_star
                    elif dataset_type == 'exp':
                        modified_f_star = exp_f_star
                    elif dataset_type == 'mult':
                        modified_f_star = mult_f_star
                    else:
                        continue

                    datasets[dataset_type].append((
                        encoded_map,
                        start,
                        goal,
                        (x, y, z),
                        g_star_val,
                        h,
                        modified_f_star[x, y, z]
                    ))

        # After processing all queries, write datasets to disk
        for dataset_type in dataset_types:
            dataset = datasets[dataset_type]

            if not dataset:
                logger.warning(
                    f"Map {map_idx + 1}: No data collected for '{dataset_type}' dataset. Skipping save.")
                continue

            # Define filename
            dataset_filename = f"dataset_map_{map_idx}_{dataset_type}.pkl"
            dataset_filepath = os.path.join(args.save_dataset_dir, dataset_filename)

            # Save datasets with highest protocol for efficiency
            with open(dataset_filepath, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                f"Map {map_idx + 1}: '{dataset_type}' dataset saved to {dataset_filepath}")

        return (map_idx, True)

    except Exception as e:
        logger.error(
            f"Map {map_idx + 1}: Unexpected error during processing: {e}")
        return (map_idx, False)


# ---------------------------
# Argument Parser
# ---------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create Datasets for F* Prediction Model with Different Penalties in 3D"
    )
    parser.add_argument("encoder_path", type=str,
                        help="Path to the pre-trained PointNet++ autoencoder model")
    parser.add_argument("--num_maps", type=int, default=2,
                        help="Number of maps to generate")
    parser.add_argument("--num_queries_per_map", type=int,
                        default=1, help="Number of queries per map")
    parser.add_argument("--depth", type=int, default=40,
                        help="Depth of the map")
    parser.add_argument("--height", type=int, default=100,
                        help="Height of the map")
    parser.add_argument("--width", type=int, default=100,
                        help="Width of the map")
    parser.add_argument("--map_save_dir", type=str, default="maps/3d_data_comp",
                        help="Directory to save the generated maps")
    parser.add_argument("--f_star_dir", type=str, default="f_star_maps_100",
                        help="Directory to save F* visualizations")
    parser.add_argument("--save_dataset_dir", type=str, default="datasets_comp",
                        help="Directory to save the generated datasets")
    parser.add_argument("--norm_save_dir", type=str, default="normalization_values_comp",
                        help="Directory to save the normalization values")
    # Map generation parameters
    parser.add_argument("--min_room_size", type=int,
                        default=60, help="Minimum size of a room")
    parser.add_argument("--max_room_size", type=int,
                        default=120, help="Maximum size of a room")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Maximum recursion depth for splitting rooms")
    parser.add_argument("--wall_thickness", type=int, default=5,
                        help="Thickness of the walls between rooms")
    parser.add_argument("--min_openings", type=int, default=1,
                        help="Minimum number of openings per wall")
    parser.add_argument("--max_openings", type=int, default=2,
                        help="Maximum number of openings per wall")
    parser.add_argument("--min_opening_size", type=int, default=10,
                        help="Minimum size of each opening in units")
    parser.add_argument("--max_opening_size", type=int, default=20,
                        help="Maximum size of each opening in units")
    parser.add_argument("--min_obstacles", type=int, default=4,
                        help="Minimum number of obstacles per room")
    parser.add_argument("--max_obstacles", type=int, default=14,
                        help="Maximum number of obstacles per room")
    parser.add_argument("--min_obstacle_size", type=int,
                        default=10, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int,
                        default=20, help="Maximum size of each obstacle")
    parser.add_argument("--obstacle_attempts", type=int, default=10,
                        help="Number of attempts to place an obstacle without overlap")
    parser.add_argument("--trap_probability", type=float, default=0.4,
                        help="Probability of placing a concave trap instead of a regular obstacle")
    parser.add_argument("--latent_dim", type=int, default=1024,
                        help="Dimensionality of the latent space")
    # Flags to generate 'mult' and 'exp' datasets
    parser.add_argument("--generate_mult", action='store_true',
                        help="Flag to generate 'mult' datasets")
    parser.add_argument("--generate_exp", action='store_true',
                        help="Flag to generate 'exp' datasets")
    return parser.parse_args()


# ---------------------------
# Normalization Utility Functions
# ---------------------------


def compute_min_max(global_dataset):
    """
    Computes the min and max values for g_star, h, and f_star from the combined global dataset.

    Args:
        global_dataset (list): Combined list of all dataset entries.

    Returns:
        dict: Dictionary containing global min and max values for g_star, h, and f_star.
    """
    try:
        if not global_dataset:
            logger.error("Global dataset is empty. Cannot compute min/max values.")
            return None

        # Initialize min and max
        g_min = float('inf')
        g_max = float('-inf')
        h_min = float('inf')
        h_max = float('-inf')
        f_star_min = float('inf')
        f_star_max = float('-inf')

        for entry in global_dataset:
            # Ensure the entry has the expected number of elements
            if len(entry) < 7:
                logger.error(f"Entry has insufficient elements: {entry}")
                continue
            g_star, h, f_star = entry[4], entry[5], entry[6]
            if g_star < g_min:
                g_min = g_star
            if g_star > g_max:
                g_max = g_star
            if h < h_min:
                h_min = h
            if h > h_max:
                h_max = h
            if f_star < f_star_min:
                f_star_min = f_star
            if f_star > f_star_max:
                f_star_max = f_star

        return {
            'g_min': g_min,
            'g_max': g_max,
            'h_min': h_min,
            'h_max': h_max,
            'f_star_min': f_star_min,
            'f_star_max': f_star_max
        }

    except Exception as e:
        logger.error(f"Failed to compute global min/max values: {e}")
        return None


def normalize_and_save_dataset(args_tuple):
    """
    Normalizes a single dataset file and saves the normalized data.

    Args:
        args_tuple (tuple): Contains (dataset_filepath, normalization_values, dataset_type, save_dataset_dir).

    Returns:
        bool: True if successful, False otherwise.
    """
    dataset_filepath, normalization_values, dataset_type, save_dataset_dir = args_tuple
    try:
        with open(dataset_filepath, 'rb') as f:
            dataset = pickle.load(f)

        g_min = normalization_values['g_min']
        g_max = normalization_values['g_max']
        h_min = normalization_values['h_min']
        h_max = normalization_values['h_max']
        f_star_min = normalization_values['f_star_min']
        f_star_max = normalization_values['f_star_max']

        normalized_dataset = []
        for entry in dataset:
            encoded_map, start, goal, current, g_star, h, target_value = entry
            g_normalized = (g_star - g_min) / (g_max - g_min) if g_max != g_min else 0.0
            h_normalized = (h - h_min) / (h_max - h_min) if h_max != h_min else 0.0
            target_normalized = (target_value - f_star_min) / (f_star_max - f_star_min) if f_star_max != f_star_min else 0.0

            if np.isfinite(g_normalized) and np.isfinite(h_normalized) and np.isfinite(target_normalized):
                normalized_dataset.append(
                    (encoded_map, start, goal, current, g_normalized, h_normalized, target_normalized))

        # Define normalized dataset filename
        basename = os.path.basename(dataset_filepath)
        normalized_dataset_filename = f"normalized_{basename}"
        normalized_dataset_filepath = os.path.join(save_dataset_dir, normalized_dataset_filename)

        # Save normalized dataset with highest protocol for efficiency
        with open(normalized_dataset_filepath, 'wb') as f:
            pickle.dump(normalized_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            f"Normalized dataset saved to {normalized_dataset_filepath}")

        # Delete the original dataset file to free up memory
        os.remove(dataset_filepath)
        logger.info(
            f"Deleted original dataset file {dataset_filepath}")

        return True

    except Exception as e:
        logger.error(f"Failed to normalize and save {dataset_filepath}: {e}")
        return False


def load_and_combine_datasets(dataset_type, save_dataset_dir):
    try:
        dataset_files = [
            os.path.join(save_dataset_dir, f)
            for f in os.listdir(save_dataset_dir)
            if f.endswith(f"_{dataset_type}.pkl") and not f.startswith('normalized_')
        ]

        global_dataset = []
        for dataset_file in dataset_files:
            try:
                with open(dataset_file, 'rb') as f:
                    dataset = pickle.load(f)
                    global_dataset.extend(dataset)
                logger.info(f"Loaded dataset from {dataset_file} with {len(dataset)} samples.")
            except EOFError:
                logger.error(f"Dataset file {dataset_file} is empty or corrupted. Skipping.")
            except Exception as e:
                logger.error(f"Failed to load {dataset_file}: {e}")
        logger.info(f"Combined {len(dataset_files)} datasets for '{dataset_type}' with {len(global_dataset)} total samples.")
        return global_dataset, dataset_files  # Return the list of dataset files to delete later

    except Exception as e:
        logger.error(f"Failed to load and combine datasets for '{dataset_type}': {e}")
        return None, None


def compute_global_min_max(global_dataset):
    """
    Computes global min and max values for g_star, h, and f_star from the combined global dataset.

    Args:
        global_dataset (list): Combined list of all dataset entries.

    Returns:
        dict: Dictionary containing global min and max values for g_star, h, and f_star.
    """
    try:
        g_values = [entry[4] for entry in global_dataset]
        h_values = [entry[5] for entry in global_dataset]
        f_star_values = [entry[6] for entry in global_dataset]

        normalization_values = {
            'g_min': min(g_values),
            'g_max': max(g_values),
            'h_min': min(h_values),
            'h_max': max(h_values),
            'f_star_min': min(f_star_values),
            'f_star_max': max(f_star_values)
        }

        return normalization_values

    except Exception as e:
        logger.error(f"Failed to compute global min and max values: {e}")
        return None


def normalize_global_dataset(global_dataset, normalization_values):
    """
    Normalizes the global dataset using the provided normalization values.

    Args:
        global_dataset (list): Combined list of all dataset entries.
        normalization_values (dict): Dictionary containing global min and max values.

    Returns:
        list: Normalized global dataset.
    """
    try:
        g_min = normalization_values['g_min']
        g_max = normalization_values['g_max']
        h_min = normalization_values['h_min']
        h_max = normalization_values['h_max']
        f_star_min = normalization_values['f_star_min']
        f_star_max = normalization_values['f_star_max']

        normalized_dataset = []
        for entry in global_dataset:
            encoded_map, start, goal, current, g_star, h, target_value = entry
            g_normalized = (g_star - g_min) / (g_max - g_min) if g_max != g_min else 0.0
            h_normalized = (h - h_min) / (h_max - h_min) if h_max != h_min else 0.0
            target_normalized = (target_value - f_star_min) / (f_star_max - f_star_min) if f_star_max != f_star_min else 0.0

            if np.isfinite(g_normalized) and np.isfinite(h_normalized) and np.isfinite(target_normalized):
                normalized_dataset.append(
                    (encoded_map, start, goal, current, g_normalized, h_normalized, target_normalized))

        logger.info(f"Normalized global dataset with {len(normalized_dataset)} entries.")
        return normalized_dataset

    except Exception as e:
        logger.error(f"Failed to normalize global dataset: {e}")
        return None


# ---------------------------
# Point Cloud Conversion Function
# ---------------------------

def occupancy_grid_to_point_cloud(map_grid, num_points=2048):
    """
    Converts a 3D occupancy grid into a point cloud by extracting the coordinates of occupied voxels.

    Args:
        map_grid (np.ndarray): 3D occupancy grid of shape [depth, height, width].
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray: Point cloud of shape [num_points, 3].
    """
    occupied_indices = np.argwhere(map_grid == 1)
    if len(occupied_indices) >= num_points:
        sampled_indices = occupied_indices[np.random.choice(len(occupied_indices), num_points, replace=False)]
    else:
        # If not enough points, duplicate some
        pad_size = num_points - len(occupied_indices)
        pad_indices = occupied_indices[np.random.choice(len(occupied_indices), pad_size, replace=True)]
        sampled_indices = np.concatenate([occupied_indices, pad_indices], axis=0)
    return sampled_indices.astype(np.float32)


# ---------------------------
# Training Loop
# ---------------------------


def feature_transform_regularizer(trans):
    """
    Regularization term for the feature transformation matrix.

    Args:
        trans (torch.Tensor): Transformation matrix of shape [B, K, K].

    Returns:
        torch.Tensor: Regularization loss.
    """
    batchsize = trans.size(0)
    K = trans.size(1)
    I = torch.eye(K, requires_grad=True).view(1, K, K).repeat(batchsize, 1, 1).to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# ---------------------------
# Main Function
# ---------------------------


def main():
    global args
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Determine which datasets to generate
    dataset_types = ['vanilla']
    if args.generate_exp:
        dataset_types.append('exp')
    if args.generate_mult:
        dataset_types.append('mult')

    logger.info(f"Datasets to generate: {dataset_types}")

    # Create necessary directories
    directories = [
        args.map_save_dir,
        args.f_star_dir,
        args.save_dataset_dir,
        args.norm_save_dir
    ]
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

    # Create F* dataset directories based on dataset_types
    for dataset_type in dataset_types:
        f_star_dataset_dir = os.path.join(args.f_star_dir, dataset_type)
        if not os.path.exists(f_star_dataset_dir):
            os.makedirs(f_star_dataset_dir)
            logger.info(f"Created F* dataset directory: {f_star_dataset_dir}")

    # **Load the Encoder Model**
    try:
        logger.info("Loading encoder model.")
        encoder = PointNetAutoencoder(
            num_points=2048,
            feature_transform=True
        ).to(device)
        encoder.load_state_dict(torch.load(
            args.encoder_path, map_location=device))
        encoder.eval()
        logger.info("Encoder model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load encoder model: {e}")
        return

    # Generate and encode all maps
    encoded_maps = []
    map_grids = []
    for map_idx in range(args.num_maps):
        logger.info(f"Generating and encoding map {map_idx + 1}/{args.num_maps}")
        try:
            # Generate the 3D map with provided parameters
            map_grid = generate_map(
                width=args.width,
                height=args.height,
                depth=args.depth,
                min_room_size=args.min_room_size,
                max_room_size=args.max_room_size,
                max_depth=args.max_depth,
                wall_thickness=args.wall_thickness,
                min_openings=args.min_openings,
                max_openings=args.max_openings,
                min_opening_size=args.min_opening_size,
                max_opening_size=args.max_opening_size,
                min_obstacles=args.min_obstacles,
                max_obstacles=args.max_obstacles,
                min_obstacle_size=args.min_obstacle_size,
                max_obstacle_size=args.max_obstacle_size,
                obstacle_attempts=args.obstacle_attempts,
                trap_probability=args.trap_probability
            )
            logger.info(f"Map {map_idx + 1} generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate map {map_idx + 1}: {e}")
            encoded_maps.append(None)
            map_grids.append(None)
            continue

        try:
            # Convert the occupancy grid to a point cloud
            point_cloud = occupancy_grid_to_point_cloud(map_grid, num_points=2048)
            # Normalize point cloud to [-1, 1] based on map size
            point_cloud = point_cloud / np.array([args.width, args.height, args.depth]) * 2 - 1
            point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).permute(0, 2, 1).to(device)  # [1, 3, 2048]

            # Encode the point cloud
            with torch.no_grad():
                # The encoder returns (recon, latent, trans_feat)
                encoder_output = encoder(point_cloud_tensor)
                
                if not isinstance(encoder_output, tuple) or len(encoder_output) < 2:
                    raise ValueError(f"Unexpected encoder output format: {type(encoder_output)}")
                
                # Extract the latent vector (second element of the tuple)
                latent = encoder_output[1]  # This is the latent vector we want
                
                if not isinstance(latent, torch.Tensor):
                    raise ValueError(f"Latent vector is not a tensor: {type(latent)}")
                
                logger.debug(f"Latent tensor shape before flattening: {latent.shape}")
                
                # Ensure we're getting just the latent vector as a flattened numpy array
                encoded_map = latent.cpu().numpy().flatten()
                logger.debug(f"Final encoded map shape: {encoded_map.shape}")

                if len(encoded_map.shape) != 1:
                    raise ValueError(f"Encoded map should be 1D, got shape: {encoded_map.shape}")

            logger.info(f"Map {map_idx + 1} encoded successfully with latent vector of size {encoded_map.shape[0]}")
            encoded_maps.append(encoded_map)
            map_grids.append(map_grid)
        except Exception as e:
            logger.error(f"Failed to encode map {map_idx + 1}: {str(e)}\nFull error: {e.__class__.__name__}")
            logger.exception("Full traceback:")
            encoded_maps.append(None)
            map_grids.append(None)
    # Prepare arguments for worker processes
    worker_args = []
    for map_idx, (encoded_map, map_grid) in enumerate(zip(encoded_maps, map_grids)):
        if encoded_map is not None and map_grid is not None:
            worker_args.append((map_idx, encoded_map, map_grid, args, dataset_types))
        else:
            logger.warning(f"Skipping map {map_idx + 1} due to previous errors.")

    # Initialize multiprocessing pool to handle A*-related workload
    try:
        pool_size = min(60, cpu_count())
        with Pool(processes=pool_size) as pool:
            logger.info(f"Starting multiprocessing pool with {len(worker_args)} maps using {pool_size} processes.")
            # Using starmap to pass multiple arguments to process_map
            results = list(tqdm(pool.starmap(process_map, worker_args), total=len(worker_args), desc="Processing Maps"))
            logger.info("Multiprocessing pool completed.")
    except Exception as e:
        logger.error(f"Multiprocessing failed: {e}")
        return

    logger.info("All maps processed and raw datasets saved.")

    # ---------------------------
    # Global Normalization Phase
    # ---------------------------
    for dataset_type in dataset_types:
        logger.info(f"Starting global normalization for '{dataset_type}' dataset.")

        # Load and combine all raw datasets
        global_dataset, dataset_files_to_delete = load_and_combine_datasets(dataset_type, args.save_dataset_dir)
        if global_dataset is None or not global_dataset:
            logger.warning(f"Failed to load datasets for '{dataset_type}' or dataset is empty. Skipping normalization.")
            continue

        # Compute global min and max values
        normalization_values = compute_min_max(global_dataset)
        if normalization_values is None:
            logger.warning(f"Failed to compute global min/max for '{dataset_type}'. Skipping normalization.")
            continue

        # Save global normalization parameters
        norm_save_path = os.path.join(
            args.norm_save_dir, f"{dataset_type}_normalization_values.pkl")
        try:
            with open(norm_save_path, 'wb') as f:
                pickle.dump(normalization_values, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                f"Global normalization parameters for '{dataset_type}' saved to {norm_save_path}")
        except Exception as e:
            logger.error(f"Failed to save normalization parameters for '{dataset_type}': {e}")
            continue

        # Normalize the global dataset
        normalized_global_dataset = normalize_global_dataset(global_dataset, normalization_values)
        if normalized_global_dataset is None:
            logger.warning(f"Failed to normalize global dataset for '{dataset_type}'.")
            continue

        # Save the normalized global dataset
        global_dataset_filename = f"dataset_{dataset_type}.pkl"
        global_dataset_filepath = os.path.join(args.save_dataset_dir, global_dataset_filename)
        try:
            with open(global_dataset_filepath, 'wb') as f:
                pickle.dump(normalized_global_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Global normalized dataset saved to {global_dataset_filepath}")
        except Exception as e:
            logger.error(f"Failed to save global normalized dataset for '{dataset_type}': {e}")
            continue

        # Delete individual raw dataset files
        try:
            for dataset_file in dataset_files_to_delete:
                os.remove(dataset_file)
                logger.info(f"Deleted intermediate raw dataset file {dataset_file}")
        except Exception as e:
            logger.error(f"Failed to delete intermediate dataset files for '{dataset_type}': {e}")

    logger.info("All datasets combined, normalized, and saved successfully.")

    # ---------------------------
    # Train the PointNet++ Autoencoder (Optional)
    # ---------------------------
    # If you wish to train the PointNet++ autoencoder, uncomment the following line
    # train()


if __name__ == '__main__':
    main()
