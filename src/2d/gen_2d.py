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
from multiprocessing import Pool
from models import UNet2DAutoencoder

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

global_encoder = None
args = None
device = 'cpu'


class Node:
    def __init__(self, pos, g=float('inf'), h=0, parent=None):
        self.pos = pos
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

def generate_map(
    width=512,
    height=512,
    min_room_size=60,
    max_room_size=120,
    max_depth=5,
    wall_thickness=5,
    min_openings=1,
    max_openings=2,
    min_opening_size=10,
    max_opening_size=20,
    min_obstacles=4,
    max_obstacles=14,
    min_obstacle_size=10,
    max_obstacle_size=20,
    obstacle_attempts=10,
    trap_probability=0.4
):
    """
    Generates a 2D map with rooms and walls with openings.
    Adds rectangular obstacles and concave traps without overlapping.

    Returns:
        np.ndarray: 2D occupancy map of shape [height, width].
    """
    map_grid = np.zeros((height, width), dtype=np.float32)
    
    root_room = Room(0, 0, width, height)
    
    def split_room(room, depth):
        if depth >= max_depth:
            return
        can_split_horizontally = room.height >= 2 * min_room_size + wall_thickness
        can_split_vertically = room.width >= 2 * min_room_size + wall_thickness
        
        if not can_split_horizontally and not can_split_vertically:
            return  # Cannot split further
        
        if can_split_horizontally and can_split_vertically:
            split_horizontally = random.choice([True, False])
        elif can_split_horizontally:
            split_horizontally = True
        else:
            split_horizontally = False
        
        if split_horizontally:
            split_min = room.y + min_room_size
            split_max = room.y + room.height - min_room_size - wall_thickness
            if split_max <= split_min:
                return  # Not enough space to split
            split_pos = random.randint(split_min, split_max)
            child1 = Room(room.x, room.y, room.width, split_pos - room.y)
            child2 = Room(room.x, split_pos + wall_thickness, room.width, room.y + room.height - split_pos - wall_thickness)
            map_grid[split_pos:split_pos + wall_thickness, room.x:room.x + room.width] = 1
            add_openings((split_pos, room.x), (split_pos, room.x + room.width), orientation='horizontal')
        else:
            split_min = room.x + min_room_size
            split_max = room.x + room.width - min_room_size - wall_thickness
            if split_max <= split_min:
                return  
            split_pos = random.randint(split_min, split_max)
            child1 = Room(room.x, room.y, split_pos - room.x, room.height)
            child2 = Room(split_pos + wall_thickness, room.y, room.x + room.width - split_pos - wall_thickness, room.height)
            map_grid[room.y:room.y + room.height, split_pos:split_pos + wall_thickness] = 1
            add_openings((room.y, split_pos), (room.y + room.height, split_pos), orientation='vertical')
        
        room.children = [child1, child2]
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
                opening_size = min(opening_size, wall_length)  
                opening_start = random.randint(start[1], end[1] - opening_size)
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        else:
            wall_length = end[0] - start[0]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)  
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
        trap_thickness = wall_thickness  

        if (trap_size * 2 + wall_thickness) > room.width or (trap_size * 2 + wall_thickness) > room.height:
            return False 

        corner_x = random.randint(room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
        corner_y = random.randint(room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)

        orientation = random.choice(['left', 'right', 'up', 'down'])

        if orientation == 'left':
            arm1 = ((corner_y, corner_x - trap_size), (trap_size, trap_thickness))
            arm2 = ((corner_y - trap_size, corner_x), (trap_thickness, trap_size))
        elif orientation == 'right':
            arm1 = ((corner_y, corner_x), (trap_size, trap_thickness))
            arm2 = ((corner_y - trap_size, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))
        elif orientation == 'up':
            arm1 = ((corner_y - trap_size, corner_x), (trap_thickness, trap_size))
            arm2 = ((corner_y - trap_size, corner_x - trap_size), (trap_size, trap_thickness))
        else:  # 'down'
            arm1 = ((corner_y, corner_x), (trap_thickness, trap_size))
            arm2 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_size, trap_thickness))
        
        (y1, x1), (h1, w1) = arm1
        (y2, x2), (h2, w2) = arm2

        if (x1 < 0 or y1 < 0 or x1 + w1 > width or y1 + h1 > height or
            x2 < 0 or y2 < 0 or x2 + w2 > width or y2 + h2 > height):
            return False 

        if (np.any(map_grid[y1:y1 + h1, x1:x1 + w1] == 1) or
            np.any(map_grid[y2:y2 + h2, x2:x2 + w2] == 1)):
            return False  

        map_grid[y1:y1 + h1, x1:x1 + w1] = 1
        map_grid[y2:y2 + h2, x2:x2 + w2] = 1

        return True 
    
    def place_triangular_trap(room):
        """
        Places a triangular concave trap within the given room.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        return False
    
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
                    if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
                        continue  # Skip if obstacle is too big for the room
                    obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
                    obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
                    if np.any(map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] == 1):
                        continue  # Overlaps with existing obstacle
                    map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 1
                    placed = True
                    break  # Successfully placed
                if not placed:
                    pass

    map_grid[0:wall_thickness, :] = 1
    map_grid[-wall_thickness:, :] = 1
    map_grid[:, 0:wall_thickness] = 1
    map_grid[:, -wall_thickness:] = 1
    
    return map_grid

def is_valid(pos, map_grid):
    return 0 <= pos[0] < map_grid.shape[0] and 0 <= pos[1] < map_grid.shape[1] and map_grid[pos] == 0

def generate_start_goal(map_grid):
    height, width = map_grid.shape
    while True:
        start = (random.randint(0, height - 1), random.randint(0, width - 1))
        goal = (random.randint(0, height - 1), random.randint(0, width - 1))
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal:
            return start, goal

def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(start, goal, map_grid):
    start_node = Node(start, g=0, h=euclidean_distance(start, goal))
    open_list = [start_node]
    closed_set = set()
    g_values = np.full(map_grid.shape, float('inf'))
    g_values[start] = 0

    while open_list:
        current = heapq.heappop(open_list)

        if current.pos == goal:
            return g_values

        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)

        for next_pos, cost in get_neighbors(current.pos, map_grid):
            if not is_valid(next_pos, map_grid):
                continue

            new_g = current.g + cost
            new_h = euclidean_distance(next_pos, goal)

            if new_g < g_values[next_pos]:
                g_values[next_pos] = new_g
                next_node = Node(next_pos, g=new_g, h=new_h, parent=current)
                heapq.heappush(open_list, next_node)

    return None

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

def distance_to_nearest_obstacle(pos, map_grid):
    obstacle_positions = np.argwhere(map_grid == 1)
    if obstacle_positions.size == 0:
        return np.inf  
    distances = np.sqrt(np.sum((obstacle_positions - pos)**2, axis=1))
    return np.min(distances)

def generate_start_goal_biased(map_grid, bias_factor=0.05):
    height, width = map_grid.shape
    all_positions = [(x, y) for x in range(height)
                     for y in range(width) if map_grid[x, y] == 0]

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

def save_fstar_visualization(map_grid, start, goal, f_star, map_idx, f_star_dir, dataset_type):
    plt.figure(figsize=(10, 10))
    plt.imshow(map_grid, cmap='binary')

    valid_mask = np.isfinite(f_star)

    scatter = plt.scatter(np.where(valid_mask)[1], np.where(valid_mask)[0],
                          c=f_star[valid_mask], cmap='viridis',
                          s=20, alpha=0.7)

    plt.colorbar(scatter, label="f* values")

    plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')

    plt.title(f"F* values for Map {map_idx + 1} ({dataset_type})")
    plt.legend()

    save_path = os.path.join(f_star_dir, dataset_type, f"fstar_map_{map_idx + 1}_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_map(map_idx):
    """
    Processes a single map and returns the datasets and normalization values.

    Args:
        map_idx (int): Index of the map being processed.

    Returns:
        tuple: (datasets, all_g_values, all_h_values, all_f_star_values)
    """
    global global_encoder, args, device
    logger.info(f"Processing map {map_idx + 1}/{args.num_maps}")

    # Generate the map
    map_grid = generate_map(
        width=args.width,
        height=args.height,
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
    logger.info(f"Map {map_idx + 1} generated.")

    # Encode the map
    with torch.no_grad():
        map_tensor = torch.from_numpy(
            map_grid).float().unsqueeze(0).unsqueeze(0).to(device)
        encoded_map = global_encoder.get_latent_vector(map_tensor).cpu().numpy().flatten()

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

    first_query = True
    for query_idx in range(args.num_queries_per_map):
        astar_res = None
        attempt_counter = 0
        while astar_res is None:
            attempt_counter += 1
            if attempt_counter > 5:
                logger.warning(f"Could not find valid start and goal after {attempt_counter} attempts in map {map_idx + 1}. Skipping this query.")
                break
            try:
                start, goal = generate_start_goal_biased(map_grid)
            except Exception as e:
                logger.error(f"Error generating start/goal in map {map_idx + 1}: {e}")
                continue
            forward_g = astar(start, goal, map_grid)
            backward_g = astar(goal, start, map_grid)
            if forward_g is not None and backward_g is not None:
                astar_res = True

        if astar_res is None:
            continue  # Skip this query if no valid path found

        f_star = forward_g + backward_g

        c_star = f_star[goal]

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
            if first_query and map_idx < 5:
                f_star_dir = args.f_star_dir
                dataset_type_dir = os.path.join(f_star_dir, dataset_type)
                if not os.path.exists(dataset_type_dir):
                    os.makedirs(dataset_type_dir)
                    logger.info(f"Created F* dataset directory: {dataset_type_dir}")
                save_fstar_visualization(
                    map_grid, start, goal, modified_f_star, map_idx, f_star_dir, dataset_type)
            first_query = False

            valid_positions = np.argwhere(np.isfinite(modified_f_star))
            for pos in valid_positions:
                x, y = pos
                g_star = forward_g[x, y]
                h = euclidean_distance((x, y), goal)
                f_star_value = modified_f_star[x, y]

                if np.isinf(g_star) or np.isinf(h) or np.isinf(f_star_value):
                    continue

                target_value = f_star_value

                datasets[dataset_type].append((encoded_map, start, goal,
                                               (x, y), g_star, h, target_value))
                all_g_values[dataset_type].append(g_star)
                all_h_values[dataset_type].append(h)
                all_f_star_values[dataset_type].append(target_value)

    logger.info(f"Map {map_idx + 1} processing completed.")

    return datasets, all_g_values, all_h_values, all_f_star_values

# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create Datasets for F* Prediction Model with Different Penalties")
    parser.add_argument("encoder_path", type=str,
                        help="Path to the pre-trained UNet2DAutoencoder model")
    parser.add_argument("--num_maps", type=int, default=2,
                        help="Number of maps to generate")
    parser.add_argument("--num_queries_per_map", type=int,
                        default=5, help="Number of queries per map")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the map")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the map")
    parser.add_argument("--map_save_dir", type=str, default="maps/2d_data_comp",
                        help="Directory to save the generated maps")
    parser.add_argument("--f_star_dir", type=str, default="f_star_maps_100",
                        help="Directory to save F* visualizations")
    parser.add_argument("--save_dataset_dir", type=str, default="datasets_comp",
                        help="Directory to save the generated datasets")
    parser.add_argument("--norm_save_dir", type=str, default="normalization_values_comp",
                        help="Directory to save the normalization values")
    # Map generation parameters
    parser.add_argument("--min_room_size", type=int, default=60, help="Minimum size of a room")
    parser.add_argument("--max_room_size", type=int, default=120, help="Maximum size of a room")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum recursion depth for splitting rooms")
    parser.add_argument("--wall_thickness", type=int, default=5, help="Thickness of the walls between rooms")
    parser.add_argument("--min_openings", type=int, default=1, help="Minimum number of openings per wall")
    parser.add_argument("--max_openings", type=int, default=2, help="Maximum number of openings per wall")
    parser.add_argument("--min_opening_size", type=int, default=10, help="Minimum size of each opening in pixels")
    parser.add_argument("--max_opening_size", type=int, default=20, help="Maximum size of each opening in pixels")
    parser.add_argument("--min_obstacles", type=int, default=4, help="Minimum number of obstacles per room")
    parser.add_argument("--max_obstacles", type=int, default=14, help="Maximum number of obstacles per room")
    parser.add_argument("--min_obstacle_size", type=int, default=10, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int, default=20, help="Maximum size of each obstacle")
    parser.add_argument("--obstacle_attempts", type=int, default=10, help="Number of attempts to place an obstacle without overlap")
    parser.add_argument("--trap_probability", type=float, default=0.4, help="Probability of placing a concave trap instead of a regular obstacle")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Dimensionality of the latent space")
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------


def main():
    global global_encoder, args, device
    args = parse_arguments()

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

    for dataset_type in ['vanilla', 'exp', 'mult']:
        f_star_dataset_dir = os.path.join(args.f_star_dir, dataset_type)
        if not os.path.exists(f_star_dataset_dir):
            os.makedirs(f_star_dataset_dir)
            logger.info(f"Created F* dataset directory: {f_star_dataset_dir}")

    try:
        logger.info("Loading encoder model.")
        global_encoder = UNet2DAutoencoder(input_channels=1, latent_dim=args.latent_dim).to(device)
        global_encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        global_encoder.eval()
        global_encoder.share_memory() 
        logger.info("Encoder model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load encoder model: {e}")
        return

    pool = Pool()

    results = pool.map(process_map, range(args.num_maps))

    pool.close()
    pool.join()

    combined_datasets = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    combined_g_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    combined_h_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    combined_f_star_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }

    for datasets, all_g_values, all_h_values, all_f_star_values in results:
        for dataset_type in ['vanilla', 'exp', 'mult']:
            combined_datasets[dataset_type].extend(datasets[dataset_type])
            combined_g_values[dataset_type].extend(all_g_values[dataset_type])
            combined_h_values[dataset_type].extend(all_h_values[dataset_type])
            combined_f_star_values[dataset_type].extend(all_f_star_values[dataset_type])

    logger.info("All maps processed.")

    normalized_datasets = {}
    normalization_values = {}
    for dataset_type in ['vanilla', 'exp', 'mult']:
        logger.info(f"Normalizing {dataset_type} dataset...")
        try:
            g_min, g_max = np.min(combined_g_values[dataset_type]), np.max(combined_g_values[dataset_type])
            h_min, h_max = np.min(combined_h_values[dataset_type]), np.max(combined_h_values[dataset_type])
            f_star_min, f_star_max = np.min(combined_f_star_values[dataset_type]), np.max(combined_f_star_values[dataset_type])

            normalized_dataset = []
            for encoded_map, start, goal, current, g_star, h, target_value in combined_datasets[dataset_type]:
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
            logger.info(f"Normalization for {dataset_type} dataset completed.")
        except Exception as e:
            logger.error(f"Error during normalization of {dataset_type} dataset: {e}")
            normalized_datasets[dataset_type] = []
            normalization_values[dataset_type] = {}

    for dataset_type in ['vanilla', 'exp', 'mult']:
        dataset = normalized_datasets.get(dataset_type, [])
        normalization_values_dataset = normalization_values.get(dataset_type, {})

        try:
            dataset_save_path = os.path.join(args.save_dataset_dir, f"{dataset_type}_dataset.pkl")
            with open(dataset_save_path, 'wb') as f:
                pickle.dump(dataset, f)
            logger.info(f"{dataset_type.capitalize()} dataset saved to {dataset_save_path}")

            norm_save_path = os.path.join(args.norm_save_dir, f"{dataset_type}_normalization_values.pkl")
            with open(norm_save_path, 'wb') as f:
                pickle.dump(normalization_values_dataset, f)
            logger.info(f"{dataset_type.capitalize()} normalization values saved to {norm_save_path}")
        except Exception as e:
            logger.error(f"Failed to save {dataset_type} dataset or normalization values: {e}")

    logger.info("All datasets created and saved successfully.")

if __name__ == '__main__':
    main()