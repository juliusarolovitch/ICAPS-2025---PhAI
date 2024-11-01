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
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Updated to include 8 orientations: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
theta_to_index = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}
index_to_theta = {v: k for k, v in theta_to_index.items()}



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
    width=256,  # Reduced from 1600 to 512
    height=256,  # Reduced from 1600 to 512
    min_room_size=40,  # Adjusted for smaller maps
    max_room_size=50,  # Adjusted for smaller maps
    max_depth=10,  # Reduced depth for smaller maps
    wall_thickness=4,
    min_openings=2,
    max_openings=3,
    min_opening_size=10,  # Adjusted for smaller maps
    max_opening_size=15,  # Adjusted for smaller maps
    min_obstacles=1,
    max_obstacles=2,
    min_obstacle_size=6,
    max_obstacle_size=8,
    obstacle_attempts=10,
    trap_probability=0.0
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
    x, y, theta = pos
    robot_length = 12  # Length along the robot's facing direction
    robot_width = 8    # Width perpendicular to the facing direction

    # Adjust dimensions based on orientation
    if theta % 45 == 0:  # Supports 0°, 45°, ..., 315°
        if theta % 90 == 0:
            dx = robot_length
            dy = robot_width
        else:
            # Diagonal orientations: approximate footprint size
            dx = robot_length * np.sqrt(2) / 2 + robot_width * np.sqrt(2) / 2
            dy = robot_length * np.sqrt(2) / 2 + robot_width * np.sqrt(2) / 2
    else:
        # Invalid orientation (not a multiple of 45°)
        return False

    # Calculate the footprint coordinates
    x_min = int(x - dx / 2)
    x_max = int(x + dx / 2)
    y_min = int(y - dy / 2)
    y_max = int(y + dy / 2)

    # Check if the robot footprint is within the map boundaries
    if x_min < 0 or x_max >= map_grid.shape[0] or y_min < 0 or y_max >= map_grid.shape[1]:
        return False

    # Check for collisions in the area covered by the robot's footprint
    footprint = map_grid[x_min:x_max+1, y_min:y_max+1]
    return np.all(footprint == 0)



def generate_start_goal(map_grid):
    height, width = map_grid.shape
    thetas = [0, 45, 90, 135, 180, 225, 270, 315]
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while attempts < max_attempts:
        start_x = random.randint(0, height-1)
        start_y = random.randint(0, width-1)
        start_theta = random.choice(thetas)
        goal_x = random.randint(0, height - 1)
        goal_y = random.randint(0, width - 1)
        goal_theta = random.choice(thetas)

        start = (start_x, start_y, start_theta)
        goal = (goal_x, goal_y, goal_theta)

        if (
            is_valid(start, map_grid) and
            is_valid(goal, map_grid) and
            start[:2] != goal[:2]
        ):
            return start, goal

        attempts += 1

    raise ValueError("Unable to find valid start and goal positions.")



def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def heuristic(current_pos, goal_pos, current_theta, goal_theta):
    distance = euclidean_distance(current_pos, goal_pos)
    theta_diff = abs(current_theta - goal_theta) % 360
    orientation_cost = min(theta_diff, 360 - theta_diff) / 45  # Normalize to [0, 8]
    return distance + orientation_cost


def get_motion_primitives(node, map_grid):
    x, y = node.pos
    theta = node.theta
    motion_primitives = []

    # Define movement step (adjust as needed)
    step_size = 1  # This should be adjusted to match the robot's actual movement capability

    # Define all 8 possible movement directions based on orientation
    # Each movement corresponds to a direction the agent can face after the move
    # For simplicity, moving forward maintains the current orientation
    # Rotations can change the orientation by ±45 degrees

    # Forward movement
    movements = {
        0: (-step_size, 0),
        45: (-step_size, step_size),
        90: (0, step_size),
        135: (step_size, step_size),
        180: (step_size, 0),
        225: (step_size, -step_size),
        270: (0, -step_size),
        315: (-step_size, -step_size)
    }
    dx, dy = movements[theta]

    # Move forward
    next_x = x + dx
    next_y = y + dy
    next_theta = theta
    if is_valid((next_x, next_y, next_theta), map_grid):
        motion_primitives.append(((next_x, next_y), next_theta, 1.0))  # Cost for moving forward

    # Rotate left by 45 degrees
    left_theta = (theta + 45) % 360
    if is_valid((x, y, left_theta), map_grid):
        motion_primitives.append(((x, y), left_theta, 0.5))  # Cost for rotation

    # Rotate right by 45 degrees
    right_theta = (theta - 45) % 360
    if is_valid((x, y, right_theta), map_grid):
        motion_primitives.append(((x, y), right_theta, 0.5))  # Cost for rotation

    return motion_primitives



# ---------------------------
# Visualization Function
# ---------------------------

def visualize_comparison(map_grid, start, goal, paths, f_maps, output_dir="output", run=0):
    """
    Generates and saves a comprehensive visualization comparing multiple A* algorithms.
    Visualizes f_map slices per theta for each algorithm.
    """
    # List of thetas (orientations)
    thetas = sorted(theta_to_index.keys())  # [0, 45, 90, 135, 180, 225, 270, 315]
    num_thetas = len(thetas)
    num_algorithms = len(paths)
    
    # Set up the figure with adjusted figsize for better uniformity
    fig, axes = plt.subplots(nrows=num_thetas, ncols=num_algorithms, figsize=(6 * num_algorithms, 6 * num_thetas))
    
    # Ensure axes is a 2D array even if there's only one row or column
    if num_thetas == 1 and num_algorithms == 1:
        axes = np.array([[axes]])
    elif num_thetas == 1:
        axes = axes[np.newaxis, :]
    elif num_algorithms == 1:
        axes = axes[:, np.newaxis]
    
    # Create custom colormap for the map_grid
    cmap = ListedColormap(['white', 'black'])
    
    # Determine global vmin and vmax for f_maps to ensure consistent color scaling
    all_f_values = []
    for algo, f_map in f_maps.items():
        if f_map is not None and not np.all(np.isnan(f_map)):
            if f_map.ndim == 3:
                all_f_values.extend(f_map.flatten()[~np.isnan(f_map.flatten())])
            else:
                all_f_values.extend(f_map.flatten()[~np.isnan(f_map.flatten())])
    global_vmin = min(all_f_values) if all_f_values else 0
    global_vmax = max(all_f_values) if all_f_values else 1
    
    for col_idx, (algo_name, path) in enumerate(paths.items()):
        f_map = f_maps.get(algo_name, None)
        for row_idx, theta in enumerate(thetas):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(algo_name, fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(f"Theta={theta}°", fontsize=12)
            
            # Display the map grid with equal aspect ratio
            ax.imshow(map_grid, cmap=cmap, interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
    
            if f_map is not None and not np.all(np.isnan(f_map)):
                if f_map.ndim == 3:
                    # Get the slice for the current theta
                    theta_idx = theta_to_index[theta]
                    f_map_theta = f_map[:, :, theta_idx]
                    f_map_theta = np.where(np.isfinite(f_map_theta), f_map_theta, np.nan)
                else:
                    f_map_theta = f_map  # If f_map is 2D, use it directly
    
                valid_positions = np.argwhere(np.isfinite(f_map_theta))
                if valid_positions.size > 0:
                    scatter = ax.scatter(valid_positions[:, 1], valid_positions[:, 0],
                                         c=f_map_theta[valid_positions[:, 0], valid_positions[:, 1]],
                                         cmap='viridis', s=10, alpha=0.8, vmin=global_vmin, vmax=global_vmax)
                    # Add colorbar to the rightmost subplot in the row
                    if col_idx == num_algorithms - 1:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = plt.colorbar(scatter, cax=cax)
                        cbar.set_label('f-value', rotation=270, labelpad=15)
                else:
                    # No valid positions, skip plotting f_map
                    pass
    
            # Plot the path
            if path:
                # Plot the path with orientation indicators
                path_x = [p[1] for p in path]
                path_y = [p[0] for p in path]
                path_theta = [p[2] for p in path]
                ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')  # Path only
    
                # Optionally, plot orientation arrows
                arrow_indices = np.linspace(0, len(path) - 1, num=min(20, len(path)), dtype=int)
                for i in arrow_indices:
                    if i >= len(path_theta):
                        continue
                    ax.arrow(path_x[i], path_y[i],
                             3 * np.cos(np.deg2rad(path_theta[i])),
                             3 * np.sin(np.deg2rad(path_theta[i])),
                             head_width=2, head_length=2, fc='red', ec='red')
    
            # # Plot start and goal positions
            # ax.arrow(start[1], start[0],
            #          5 * np.cos(np.deg2rad(start[2])),
            #          5 * np.sin(np.deg2rad(start[2])),
            #          head_width=3, head_length=3, fc='green', ec='green', label='Start')
            # ax.arrow(goal[1], goal[0],
            #          5 * np.cos(np.deg2rad(goal[2])),
            #          5 * np.sin(np.deg2rad(goal[2])),
            #          head_width=3, head_length=3, fc='red', ec='red', label='Goal')
    
            
    
    # Adjust layout manually to ensure uniform subplot sizes
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
    
    # Save the figure
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
    bottlenecks = []
    for pos in path:
        y, x, _ = pos  # Extract (y, x)
        free_neighbors = sum(is_valid((ny, nx, 0), map_grid) for (ny, nx), _ in get_neighbors((y, x)))
        if free_neighbors <= 2:
            bottlenecks.append((y, x))
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
    h0_map = compute_heuristic_map(map_grid, goal[:2])  # Pass only (y, x)

    # Dual heuristic (h1) by inflating obstacles
    inflated_map = inflate_obstacles(map_grid, radius=inflation_radius)
    h1_map = compute_heuristic_map(inflated_map, goal[:2])

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
        h_map = compute_heuristic_map(map_copy, goal[:2])
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

def generate_mha_heuristics(map_grid, start, goal, robot_diameter=12):
    """
    Generates multiple heuristics following the PH and PG schemes from the MHA* paper.
    
    Args:
        map_grid: Original occupancy grid
        start: Start position
        goal: Goal position
        robot_diameter: Robot's outer diameter for bottleneck detection
        
    Returns:
        List of heuristic functions
    """
    def compute_2d_dijkstra(grid, start_pos, goal_pos):
        """Computes 2D Dijkstra path on the grid."""
        height, width = grid.shape
        distances = np.full((height, width), np.inf)
        distances[start_pos] = 0
        visited = set()
        pq = [(0, start_pos)]
        came_from = {}
        
        while pq:
            dist, current = heapq.heappop(pq)
            if current == goal_pos:
                break
                
            if current in visited:
                continue
                
            visited.add(current)
            
            # Check 8-connected neighbors
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < height and 
                    0 <= next_pos[1] < width and 
                    not grid[next_pos]):
                    cost = np.sqrt(dx*dx + dy*dy)
                    new_dist = dist + cost
                    if new_dist < distances[next_pos]:
                        distances[next_pos] = new_dist
                        came_from[next_pos] = current
                        heapq.heappush(pq, (new_dist, next_pos))
        
        # Reconstruct path
        path = []
        current = goal_pos
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start_pos)
        return path[::-1], distances

    def find_bottlenecks(grid, path, robot_diameter):
        """
        Identifies bottlenecks (narrow passages) along the path.
        A bottleneck is defined as any point where the passage width is ≤ robot_diameter.
        """
        bottlenecks = []
        radius = robot_diameter // 2
        
        for pos in path:
            # Check passage width by growing a circle until hitting obstacles
            width = 0
            while width <= robot_diameter:
                # Create a circular mask
                y, x = np.ogrid[-width:width+1, -width:width+1]
                mask = x*x + y*y <= width*width
                
                # Get region around current position
                y_start = max(0, pos[0]-width)
                y_end = min(grid.shape[0], pos[0]+width+1)
                x_start = max(0, pos[1]-width)
                x_end = min(grid.shape[1], pos[1]+width+1)
                
                region = grid[y_start:y_end, x_start:x_end]
                mask = mask[:region.shape[0], :region.shape[1]]
                
                # If any obstacle within radius, this is passage width
                if np.any(region[mask] == 1):
                    if width <= robot_diameter:
                        bottlenecks.append(pos)
                    break
                width += 1
        
        return bottlenecks

    def create_tunnel_map(grid, path, tunnel_width=30):
        """Creates a map with all cells blocked except a tunnel around the path."""
        tunnel_map = np.ones_like(grid)  # Start with all blocked
        radius = tunnel_width // 2
        
        for pos in path:
            # Create a tunnel of specified width around the path
            y_start = max(0, pos[0]-radius)
            y_end = min(grid.shape[0], pos[0]+radius+1)
            x_start = max(0, pos[1]-radius)
            x_end = min(grid.shape[1], pos[1]+radius+1)
            
            tunnel_map[y_start:y_end, x_start:x_end] = 0  # Clear tunnel
            
        return tunnel_map

    heuristic_maps = []
    current_map = map_grid.copy()
    
    # Generate Progressive Heuristics (PH)
    max_iterations = 10
    for i in range(max_iterations):
        # Compute 2D Dijkstra path
        path, distance_map = compute_2d_dijkstra(current_map, start[:2], goal[:2])
        heuristic_maps.append(distance_map)
        
        # Find bottlenecks
        bottlenecks = find_bottlenecks(current_map, path, robot_diameter)
        if not bottlenecks:
            break  # No more bottlenecks
            
        # Create new map with bottlenecks blocked
        new_map = current_map.copy()
        for pos in bottlenecks:
            new_map[pos] = 1
        current_map = new_map
        
    # Generate Progressive Heuristic with Guidance (PG)
    if path:  # If we found at least one path
        tunnel_map = create_tunnel_map(map_grid, path)
        _, tunnel_distance_map = compute_2d_dijkstra(tunnel_map, start[:2], goal[:2])
        heuristic_maps.append(tunnel_distance_map)
    
    # Convert distance maps to heuristic functions
    def create_heuristic(distance_map):
        def h(pos, goal_pos=None, theta=None, goal_theta=None):
            # Only use the position component, ignore orientation for this heuristic
            return distance_map[pos[0], pos[1]]
        return h
    
    return [create_heuristic(m) for m in heuristic_maps]



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
        encoded_map = encoder.encode(map_tensor).cpu().numpy().flatten()  # Should be latent_dim-dimensional

    h = euclidean_distance(current[:2], goal[:2])  # Use positions only for distance

    g_normalized = normalize(g, g_min, g_max)
    h_normalized = normalize(h, h_min, h_max)

    # Normalize positions including theta
    start_normalized = np.array([
        start[0] / (map_grid.shape[0] - 1),
        start[1] / (map_grid.shape[1] - 1),
        start[2] / 315.0  # Updated to 315 for normalization
    ])

    goal_normalized = np.array([
        goal[0] / (map_grid.shape[0] - 1),
        goal[1] / (map_grid.shape[1] - 1),
        goal[2] / 315.0  # Updated to 315 for normalization
    ])

    current_normalized = np.array([
        current[0] / (map_grid.shape[0] - 1),
        current[1] / (map_grid.shape[1] - 1),
        current[2] / 315.0  # Updated to 315 for normalization
    ])


    # Concatenate features in the correct order
    input_tensor = np.concatenate([
        encoded_map,               # latent_dim elements
        start_normalized,          # 3 elements
        goal_normalized,           # 3 elements
        current_normalized,        # 3 elements
        np.array([g_normalized, h_normalized])    # 2 elements
    ])  # Total length: latent_dim + 3 + 3 + 3 + 2

    input_tensor = torch.from_numpy(input_tensor).float().to(device)  # Shape: [total_length]

    # Ensure input_tensor has shape [1, total_length]
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, total_length]

    model.eval()
    with torch.no_grad():
        f_star_predicted = model(input_tensor)
        if isinstance(f_star_predicted, tuple) or isinstance(f_star_predicted, list):
            f_star_predicted = f_star_predicted[0]

        # Handle tensor output
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
    def __init__(self, pos, theta, g=float('inf'), h=0, parent=None):
        self.pos = pos            # (x, y)
        self.theta = theta        # Orientation: 0, 45, 90, 135, 180, 225, 270, 315
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f



class ModelNode:
    def __init__(self, pos, theta, f_star, g, parent=None, h=None):
        assert isinstance(pos, tuple) and len(pos) == 2, "Position must be a tuple of (x, y)"
        # Include all 8 orientations
        assert theta in [0, 45, 90, 135, 180, 225, 270, 315], "Theta must be one of [0°, 45°, ..., 315°]"

        self.pos = pos
        self.theta = theta
        self.f_star = f_star
        self.g = g
        self.parent = parent
        self.h = h 

    def __lt__(self, other):
        return self.f_star < other.f_star



@dataclass(order=True)
class Node:
    f: float
    pos: tuple = field(compare=False)      # (x, y)
    theta: int = field(compare=False)      # Orientation: 0, 90, 180, 270
    g: float = field(compare=False, default=float('inf'))
    h: float = field(compare=False, default=0)
    parent: any = field(compare=False, default=None)


# ---------------------------
# Heuristic-Based Search Algorithms
# ---------------------------

def astar(start, goal, map_grid, max_expansions):
    """
    Performs A* search considering orientation.

    Args:
        start (tuple): (x, y, theta)
        goal (tuple): (x, y, theta)
        map_grid (np.ndarray): 2D occupancy grid.
        max_expansions (int): Maximum node expansions allowed.

    Returns:
        tuple or None:
            - path (list): List of positions from start to goal. Each position is a tuple (x, y, theta).
            - total_cost (float): Total cost of the path.
            - expansions (int): Number of nodes expanded.
            - f_map (np.ndarray): 2D array of f-values for visualization.
    """
    # Define orientation indices
    start_theta_idx = theta_to_index[start[2]]
    goal_theta_idx = theta_to_index[goal[2]]

    # Initialize g_values as a 3D array (x, y, theta)
   # Correct size for 8 orientations
    g_values = np.full((map_grid.shape[0], map_grid.shape[1], 8), float('inf'))

    g_values[start[0], start[1], start_theta_idx] = 0

    # Calculate heuristic for the start node
    h_start = heuristic(start[:2], goal[:2], start[2], goal[2])

    # Initialize the start node
    start_node = Node(f=h_start, pos=start[:2], theta=start[2], g=0, h=h_start, parent=None)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    closed_set = set()

    # Initialize f_map for visualization
    f_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    expansions = 0

    while open_list:
        current_f, current = heapq.heappop(open_list)
        expansions += 1

        # Check if maximum expansions exceeded
        if expansions > max_expansions:
            logger.warning("A* search exceeded maximum expansions.")
            return None,0,expansions,None

        current_theta_idx = theta_to_index[current.theta]
        current_state = (current.pos, current.theta)

        # Check if goal is reached with the correct orientation
        if current.pos == goal[:2] and current.theta == goal[2]:
            logger.info(f"Goal reached at position {current.pos} with orientation {current.theta}")
            # Reconstruct path
            path = []
            total_cost = current.g
            temp = current
            while temp:
                path.append((*temp.pos, temp.theta))
                temp = temp.parent
            path = path[::-1]  # Reverse to get path from start to goal
            logger.debug(f"Path: {path}")
            return path, total_cost, expansions, f_map

        # Skip if already processed
        if current_state in closed_set:
            continue

        closed_set.add(current_state)

        # Record f-value for visualization
        f_map[current.pos[0], current.pos[1]] = current.f

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            # Validate the neighbor's position and orientation
            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            neighbor_theta_idx = theta_to_index[neighbor_theta]

            tentative_g = current.g + cost

            if tentative_g < g_values[neighbor_pos[0], neighbor_pos[1], neighbor_theta_idx]:
                g_values[neighbor_pos[0], neighbor_pos[1], neighbor_theta_idx] = tentative_g
                h = heuristic(neighbor_pos, goal[:2], neighbor_theta, goal[2])
                neighbor_node = Node(f=tentative_g + h, pos=neighbor_pos, theta=neighbor_theta, g=tentative_g, h=h, parent=current)
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))

    logger.warning(f"A* failed to find a path from {start[:2]} to {goal[:2]}")
    return None,0,0,None


def astar_weighted_with_reopening_tracking(start, goal, map_grid, epsilon=3.0, allow_reopening=True, max_expansions=100000):
    open_list = []
    closed_set = set()
    g_score = {}
    reopening_count = 0
    expansion_count = 0

    h_start = heuristic(start[:2], goal[:2], start[2], goal[2])
    start_node = Node(f=h_start * epsilon, pos=start[:2], theta=start[2], g=0, h=h_start)
    heapq.heappush(open_list, (start_node.f, start_node))
    g_score[(start[0], start[1], start[2])] = 0

    # Use a dictionary to keep track of nodes in the open list
    open_dict = { (start[0], start[1], start[2]) : start_node }

    f_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    while open_list:
        current_f, current = heapq.heappop(open_list)
        current_state = (current.pos[0], current.pos[1], current.theta)
        if current_state in closed_set:
            continue

        expansion_count += 1
        if expansion_count > max_expansions:
            logger.warning("Weighted A* search exceeded maximum expansions.")
            return None, None, expansion_count, reopening_count, f_map

        if current.pos == goal[:2] and current.theta == goal[2]:
            logger.info(f"Weighted A*: Goal reached at position {current.pos} with orientation {current.theta}")
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append((current.pos[0], current.pos[1], current.theta))
                current = current.parent
            return path[::-1], total_cost, expansion_count, reopening_count, f_map

        closed_set.add(current_state)
        open_dict.pop(current_state, None)  # Remove from open_dict

        theta_idx = theta_to_index[current.theta]
        f_map[current.pos[0], current.pos[1], theta_idx] = current.f

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            neighbor_state = (neighbor_pos[0], neighbor_pos[1], neighbor_theta)
            tentative_g = current.g + cost

            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                h = heuristic(neighbor_pos, goal[:2], neighbor_theta, goal[2])
                neighbor_node = Node(f=tentative_g + epsilon * h, pos=neighbor_pos, theta=neighbor_theta, g=tentative_g, h=h, parent=current)

                if neighbor_state in closed_set:
                    if allow_reopening:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                        open_dict[neighbor_state] = neighbor_node
                elif neighbor_state in open_dict:
                    existing_node = open_dict[neighbor_state]
                    if tentative_g < existing_node.g:
                        # Update node in open_list
                        existing_node.g = tentative_g
                        existing_node.f = tentative_g + epsilon * h
                        existing_node.parent = current
                        heapq.heappush(open_list, (existing_node.f, existing_node))
                else:
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                    open_dict[neighbor_state] = neighbor_node

    logger.warning(f"Weighted A* failed to find a path from {start[:2]} to {goal[:2]}")
    return None, None, expansion_count, reopening_count, f_map


def potential_search_with_reopening_tracking(start, goal, map_grid, B=3.0, allow_reopening=True, max_expansions=100000):
    class PotentialNode:
        def __init__(self, pos, theta, g, h, parent=None):
            self.pos = pos
            self.theta = theta
            self.g = g
            self.h = h
            self.f = g + h
            self.parent = parent
            self.flnr = h / max(B * self.f - g, 1e-8)

        def __lt__(self, other):
            return self.flnr < other.flnr

    open_list = []
    closed_set = set()
    g_score = {}
    reopening_count = 0
    expansion_count = 0

    h_start = euclidean_distance(start[:2], goal[:2])
    f_min = h_start
    C = B * f_min

    start_node = PotentialNode(start[:2], start[2], 0, h_start)
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1], start[2])] = 0

    # Use a dictionary to keep track of nodes in the open list
    open_dict = { (start[0], start[1], start[2]) : start_node }

    flnr_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        current_state = (current.pos[0], current.pos[1], current.theta)
        if current_state in closed_set:
            continue

        expansion_count += 1
        if expansion_count > max_expansions:
            logger.warning("Potential Search exceeded maximum expansions.")
            return None, None, expansion_count, reopening_count, flnr_map

        if current.pos == goal[:2] and current.theta == goal[2] and current.g <= C:
            logger.info(f"Potential Search: Goal reached at position {current.pos} with orientation {current.theta}")
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append((current.pos[0], current.pos[1], current.theta))
                current = current.parent
            return path[::-1], total_cost, expansion_count, reopening_count, flnr_map

        closed_set.add(current_state)
        open_dict.pop(current_state, None)

        theta_idx = theta_to_index[current.theta]
        flnr_map[current.pos[0], current.pos[1], theta_idx] = current.flnr

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            neighbor_state = (neighbor_pos[0], neighbor_pos[1], neighbor_theta)
            tentative_g = current.g + cost

            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            if tentative_g > C:
                continue

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                h = euclidean_distance(neighbor_pos, goal[:2])
                f = tentative_g + h

                if f < f_min:
                    f_min = f
                    C = B * f_min
                    # Update FLNR values for open list
                    new_open_list = []
                    open_dict.clear()
                    while open_list:
                        node = heapq.heappop(open_list)
                        node.flnr = node.h / max(C - node.g, 1e-8)
                        heapq.heappush(new_open_list, node)
                        open_dict[(node.pos[0], node.pos[1], node.theta)] = node
                    open_list = new_open_list

                flnr = h / max(C - tentative_g, 1e-8)
                neighbor_node = PotentialNode(neighbor_pos, neighbor_theta, tentative_g, h, parent=current)
                neighbor_node.flnr = flnr

                if neighbor_state in closed_set:
                    if allow_reopening:
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

    logger.warning(f"Potential Search failed to find a path from {start[:2]} to {goal[:2]}")
    return None, None, expansion_count, reopening_count, flnr_map


def imha_star_with_reopening_tracking(start, goal, map_grid, heuristic_functions, w=1.0, allow_reopening=True, max_expansions=100000):
    open_lists = [[] for _ in range(len(heuristic_functions))]
    closed_set = set()
    g_score = {}
    expansions = 0
    reopening_count = 0

    # Initialize nodes for each heuristic
    open_dicts = [{} for _ in range(len(heuristic_functions))]

    for i, h_func in enumerate(heuristic_functions):
        h_value = h_func(start)
        node = AStarNode(start[:2], start[2], g=0, h=h_value)
        heapq.heappush(open_lists[i], (node.f, node))
        g_score[(start[0], start[1], start[2], i)] = 0  # Separate g-scores per heuristic
        open_dicts[i][(start[0], start[1], start[2])] = node

    # Initialize f_map with NaNs for visualization (3D: x, y, theta)
    f_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    while any(open_lists):
        for i, open_list in enumerate(open_lists):
            if not open_list:
                continue
            f_current, current = heapq.heappop(open_list)
            current_state = (current.pos[0], current.pos[1], current.theta)

            if current_state in closed_set:
                continue

            expansions += 1

            if expansions > max_expansions:
                logger.warning("IMHA* exceeded maximum expansions.")
                return None, None, expansions, reopening_count, f_map

            # Check if goal is reached with correct orientation
            if current.pos == goal[:2] and current.theta == goal[2]:
                logger.info(f"IMHA*: Goal reached at position {current.pos} with orientation {current.theta}")
                # Reconstruct path
                path = []
                total_cost = current.g
                while current:
                    path.append((current.pos[0], current.pos[1], current.theta))
                    current = current.parent
                return path[::-1], total_cost, expansions, reopening_count, f_map

            closed_set.add(current_state)
            open_dicts[i].pop(current_state, None)
            theta_idx = theta_to_index[current.theta]
            f_map[current.pos[0], current.pos[1], theta_idx] = current.f

            for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
                neighbor_state = (neighbor_pos[0], neighbor_pos[1], neighbor_theta)
                tentative_g = current.g + cost

                if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                    continue

                if tentative_g < g_score.get((neighbor_state[0], neighbor_state[1], neighbor_state[2], i), float('inf')):
                    g_score[(neighbor_state[0], neighbor_state[1], neighbor_state[2], i)] = tentative_g
                    h = heuristic(neighbor_pos, goal[:2], neighbor_theta, goal[2])
                    neighbor_node = AStarNode(neighbor_pos, neighbor_theta, tentative_g, h, parent=current)

                    if neighbor_state in closed_set:
                        if allow_reopening:
                            closed_set.remove(neighbor_state)
                            reopening_count += 1
                            heapq.heappush(open_lists[i], (neighbor_node.f, neighbor_node))
                            open_dicts[i][neighbor_state] = neighbor_node
                    elif neighbor_state in open_dicts[i]:
                        existing_node = open_dicts[i][neighbor_state]
                        if tentative_g < existing_node.g:
                            # Update node in open_list
                            existing_node.g = tentative_g
                            existing_node.f = tentative_g + h
                            existing_node.parent = current
                            heapq.heappush(open_lists[i], (existing_node.f, existing_node))
                    else:
                        heapq.heappush(open_lists[i], (neighbor_node.f, neighbor_node))
                        open_dicts[i][neighbor_state] = neighbor_node

    logger.warning("IMHA* failed to find a path.")
    return None, None, expansions, reopening_count, f_map


def smha_star(start, goal, map_grid, heuristic_functions, w1=1.0, w2=1.0, max_expansions=100000):
    """
    Shared Multi-Heuristic A* (SMHA*) considering orientation.
    All heuristics share a single open list.
    
    Args:
        start (tuple): Starting position (x, y, theta).
        goal (tuple): Goal position (x, y, theta).
        map_grid (np.ndarray): 2D occupancy grid.
        heuristic_functions (list): List of heuristic functions.
        w1 (float): Weight for the anchor heuristic.
        w2 (float): Weight for inadmissible heuristics.
        max_expansions (int): Maximum node expansions allowed.
    
    Returns:
        tuple or None:
            - path (list): List of positions from start to goal as (x, y, theta). None if no path found.
            - total_cost (float): Total cost of the path.
            - expansions (int): Number of nodes expanded.
            - f_map (np.ndarray): 3D array of f-values for visualization.
    """
    class SMHANode:
        def __init__(self, pos, theta, g, h_values, parent=None):
            self.pos = pos            # (x, y)
            self.theta = theta        # Orientation: 0, 90, 180, 270
            self.g = g
            self.h_values = h_values  # List of heuristic values
            self.parent = parent
            self.f_anchor = g + h_values[0]  # Anchor heuristic
            self.f_inadmissible = [g + h for h in h_values[1:]]

        def __lt__(self, other):
            return self.f_anchor < other.f_anchor

    open_list = []
    closed_set = set()
    g_score = {(start[0], start[1], start[2]): 0}
    expansions = 0

    # Initialize start node
    h_values_start = [h(start) for h in heuristic_functions]
    start_node = SMHANode(start[:2], start[2], 0, h_values_start)
    heapq.heappush(open_list, (start_node.f_anchor, start_node))

    # Initialize f_map with NaNs for visualization (3D: x, y, theta)
    f_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    while open_list:
        f_current, current = heapq.heappop(open_list)
        current_state = (current.pos[0], current.pos[1], current.theta)

        expansions += 1

        if expansions > max_expansions:
            logger.warning("SMHA* exceeded maximum expansions.")
            return None, None, expansions, f_map

        # Check if goal is reached with correct orientation
        if current.pos == goal[:2] and current.theta == goal[2]:
            logger.info(f"SMHA*: Goal reached at position {current.pos} with orientation {current.theta}")
            # Reconstruct path
            path = []
            total_cost = current.g
            while current:
                path.append((current.pos[0], current.pos[1], current.theta))
                current = current.parent
            return path[::-1], total_cost, expansions, f_map

        if current_state in closed_set:
            continue

        closed_set.add(current_state)
        theta_idx = theta_to_index[current.theta]
        f_map[current.pos[0], current.pos[1], theta_idx] = f_current

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            neighbor_state = (neighbor_pos[0], neighbor_pos[1], neighbor_theta)

            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            tentative_g = current.g + cost

            if tentative_g >= g_score.get(neighbor_state, float('inf')):
                continue  # Not a better path

            g_score[neighbor_state] = tentative_g
            h_values = [h(neighbor_pos) for h in heuristic_functions]
            neighbor_node = SMHANode(neighbor_pos, neighbor_theta, tentative_g, h_values, parent=current)
            neighbor_node.f_anchor = tentative_g + w1 * h_values[0]
            neighbor_node.f_inadmissible = [tentative_g + w2 * h for h in h_values[1:]]
            f_min = min([neighbor_node.f_anchor] + neighbor_node.f_inadmissible)
            heapq.heappush(open_list, (f_min, neighbor_node))

    logger.warning("SMHA* failed to find a path.")
    return None, None, expansions, f_map

# ---------------------------
# Model-Based Search Algorithms
# ---------------------------

def astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, allow_reopening=True, max_expansions=100000):
    open_list = []
    closed_set = set()
    g_score = {}
    reopening_count = 0
    expansions = 0  # Using a single variable for expansion counting

    # Compute initial f_star for the start node using the neural network model
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_theta = start[2]
    start_node = ModelNode(pos=start[:2], theta=start_theta, f_star=start_f_star, g=0, parent=None)
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1], start[2])] = 0

    # Use a dictionary to keep track of nodes in the open list
    open_dict = { (start[0], start[1], start[2]) : start_node }

    # Initialize f_star_map with NaNs for visualization (3D: x, y, theta)
    f_star_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    while open_list:
        current = heapq.heappop(open_list)
        current_state = (current.pos[0], current.pos[1], current.theta)
        if current_state in closed_set:
            continue

        expansions += 1  # Increment expansions when we actually expand a node
        if expansions > max_expansions:
            logger.warning("A* with Model exceeded maximum expansions.")
            return None, None, expansions, reopening_count, f_star_map

        # Check if goal is reached with correct orientation
        if current.pos == goal[:2] and current.theta == goal[2]:
            logger.info(f"A* with Model: Goal reached at position {current.pos} with orientation {current.theta}")
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append((current.pos[0], current.pos[1], current.theta))
                current = current.parent
            return path[::-1], total_cost, expansions, reopening_count, f_star_map

        closed_set.add(current_state)
        open_dict.pop(current_state, None)

        # Record f_star value for visualization
        theta_idx = theta_to_index[current.theta]
        f_star_map[current.pos[0], current.pos[1], theta_idx] = current.f_star

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            neighbor_state = (neighbor_pos[0], neighbor_pos[1], neighbor_theta)

            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            tentative_g = current.g + cost

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                f_star = run_inference(map_grid, start, goal, (*neighbor_pos, neighbor_theta), tentative_g, encoder, model, normalization_values, device)
                neighbor_node = ModelNode(pos=neighbor_pos, theta=neighbor_theta, f_star=f_star, g=tentative_g, parent=current)

                if neighbor_state in closed_set:
                    if allow_reopening:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        heapq.heappush(open_list, neighbor_node)
                        open_dict[neighbor_state] = neighbor_node
                elif neighbor_state in open_dict:
                    existing_node = open_dict[neighbor_state]
                    if tentative_g < existing_node.g:
                        # Update node in open_list
                        existing_node.g = tentative_g
                        existing_node.f_star = f_star
                        existing_node.parent = current
                        heapq.heappush(open_list, existing_node)
                else:
                    heapq.heappush(open_list, neighbor_node)
                    open_dict[neighbor_state] = neighbor_node

    logger.warning(f"A* with Model-based Heuristic failed to find a path from {start[:2]} to {goal[:2]}")
    return None, None, expansions, reopening_count, f_star_map

def focal_astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, epsilon=3.0, allow_reopening=True, max_expansions=100000):
    import bisect

    class OpenList:
        def __init__(self):
            self.elements = []
            self.entry_finder = {}
            self.counter = itertools.count()

        def add_or_update_node(self, node):
            f = node.f_star
            node_key = (node.pos[0], node.pos[1], node.theta)
            if node_key in self.entry_finder:
                existing_count, existing_node = self.entry_finder[node_key]
                if node.g < existing_node.g:
                    # Remove the old node
                    self.remove_node(existing_node)
                    # Add the updated node
                    count_value = next(self.counter)
                    bisect.insort_left(self.elements, (f, count_value, node))
                    self.entry_finder[node_key] = (count_value, node)
            else:
                count_value = next(self.counter)
                bisect.insort_left(self.elements, (f, count_value, node))
                self.entry_finder[node_key] = (count_value, node)

        def remove_node(self, node):
            node_key = (node.pos[0], node.pos[1], node.theta)
            if node_key in self.entry_finder:
                count_value, existing_node = self.entry_finder[node_key]
                f = existing_node.f_star
                idx = bisect.bisect_left(self.elements, (f, count_value, existing_node))
                while idx < len(self.elements):
                    if (self.elements[idx][2].pos == node.pos and
                        self.elements[idx][2].theta == node.theta):
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

    open_list = OpenList()
    g_score = {}
    reopening_count = 0
    expansions = 0
    closed_set = set()

    # Compute initial f_star for the start node using the neural network model
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_theta = start[2]
    start_node = ModelNode(pos=start[:2], theta=start_theta, f_star=start_f_star, g=0, parent=None)
    open_list.add_or_update_node(start_node)
    g_score[(start[0], start[1], start[2])] = 0

    # Initialize f_star_map with NaNs for visualization (3D: x, y, theta)
    f_star_map = np.full((map_grid.shape[0], map_grid.shape[1], 8), np.nan)

    while not open_list.is_empty():
        current_f_min = open_list.get_f_min()
        focal_nodes = open_list.get_focal_nodes(current_f_min, epsilon)

        if not focal_nodes:
            return None, None, expansions, reopening_count, f_star_map

        # Select the node with the smallest f_star from the focal list
        current = min(focal_nodes, key=lambda node: node.f_star)
        open_list.remove_node(current)
        current_state = (current.pos[0], current.pos[1], current.theta)
        if current_state in closed_set:
            continue

        expansions += 1
        if expansions > max_expansions:
            logger.warning("Focal A* with Model exceeded maximum expansions.")
            return None, None, expansions, reopening_count, f_star_map

        # Check if goal is reached with correct orientation
        if current.pos == goal[:2] and current.theta == goal[2]:
            logger.info(f"Focal A* with Model: Goal reached at position {current.pos} with orientation {current.theta}")
            # Reconstruct path and compute total cost
            path = []
            total_cost = current.g
            while current:
                path.append((current.pos[0], current.pos[1], current.theta))
                current = current.parent
            return path[::-1], total_cost, expansions, reopening_count, f_star_map

        closed_set.add(current_state)

        # Record f_star value for visualization
        theta_idx = theta_to_index[current.theta]
        f_star_map[current.pos[0], current.pos[1], theta_idx] = current.f_star

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            neighbor_state = (neighbor_pos[0], neighbor_pos[1], neighbor_theta)

            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            tentative_g = current.g + cost

            if tentative_g < g_score.get(neighbor_state, float('inf')):
                g_score[neighbor_state] = tentative_g
                f_star = run_inference(map_grid, start, goal, (*neighbor_pos, neighbor_theta), tentative_g, encoder, model, normalization_values, device)
                neighbor_node = ModelNode(pos=neighbor_pos, theta=neighbor_theta, f_star=f_star, g=tentative_g, parent=current)

                if neighbor_state in closed_set:
                    if allow_reopening:
                        closed_set.remove(neighbor_state)
                        reopening_count += 1
                        open_list.add_or_update_node(neighbor_node)
                else:
                    open_list.add_or_update_node(neighbor_node)

    logger.warning("Focal A* with Model failed to find a path.")
    return None, None, expansions, reopening_count, f_star_map



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

    # Define all algorithms including model-based ones with and without reopening
    algorithms = [
        'Traditional A*',
        'Weighted A* ε=3.0 Allow Reopening=True',
        'Weighted A* ε=3.0 Allow Reopening=False',
        'Potential Search Allow Reopening=True',
        'Potential Search Allow Reopening=False',
        'IMHA* Allow Reopening=True',
        'IMHA* Allow Reopening=False',
        'SMHA*',
    ]
    for model_name in models:
        algorithms.append(f'{model_name} A* Allow Reopening=True')
        algorithms.append(f'{model_name} A* Allow Reopening=False')
        algorithms.append(f'{model_name} Focal A* ε=3.0 Allow Reopening=True')
        algorithms.append(f'{model_name} Focal A* ε=3.0 Allow Reopening=False')

    # Initialize statistics dictionary with re-openings tracking
    stats = {algo: {
        'total_expansions_reduction': 0,
        'total_path_cost_increase': 0,
        'path_cost_increases': [],
        'optimal_paths': 0,
        'total_queries': 0,
        'total_reopenings': 0,          # Total re-openings across all queries
        'reopenings_per_query': [],      # List of reopenings per query
        'total_expansions': 0            # Total expansions across all queries
    } for algo in algorithms}

    query_counter = 0
    for map_idx in range(num_maps):
        map_data = generate_map()
        print(f"\nRunning assessment on generated map {map_idx + 1}\n")

        for query in range(num_queries_per_map):
            print(f"\nRunning assessment on query {query + 1} for map {map_idx + 1}...\n")

            # Generate start and goal positions
            while True:
                start, goal = generate_start_goal(map_data)
                traditional_path, traditional_path_cost, traditional_expanded, f_map = astar(start, goal, map_data, max_expansions=1000000)
                if traditional_path is not None:
                    break
                else:
                    map_data = generate_map()

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

            # Define algorithm variants with reopening options
            algorithm_variants = [
                ('Weighted A* ε=3.0 Allow Reopening=True', lambda: astar_weighted_with_reopening_tracking(start, goal, map_data, epsilon=3.0, allow_reopening=True, max_expansions=500000)),
                ('Weighted A* ε=3.0 Allow Reopening=False', lambda: astar_weighted_with_reopening_tracking(start, goal, map_data, epsilon=3.0, allow_reopening=False, max_expansions=500000)),
                ('Potential Search Allow Reopening=True', lambda: potential_search_with_reopening_tracking(start, goal, map_data, B=3.0, allow_reopening=True, max_expansions=500000)),
                ('Potential Search Allow Reopening=False', lambda: potential_search_with_reopening_tracking(start, goal, map_data, B=3.0, allow_reopening=False, max_expansions=500000)),
                ('IMHA* Allow Reopening=True', lambda: imha_star_with_reopening_tracking(start, goal, map_data, heuristic_functions=generate_mha_heuristics(map_data, start, goal, robot_diameter=12), w=3.0, allow_reopening=True, max_expansions=500000)),
                ('IMHA* Allow Reopening=False', lambda: imha_star_with_reopening_tracking(start, goal, map_data, heuristic_functions=generate_mha_heuristics(map_data, start, goal, robot_diameter=12), w=3.0, allow_reopening=False, max_expansions=500000)),
            ]

            # Execute algorithm variants
            for algo_name, algo_func in algorithm_variants:
                print(f"Running {algo_name} on query {query + 1}...")
                path, path_cost, expanded, reopenings, f_map_algo = algo_func()

                if path is None:
                    print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                else:
                    print(f"{algo_name} expansions: {expanded}, cost: {path_cost}, reopenings: {reopenings}")
                    expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                    cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                    stats[algo_name]['total_expansions_reduction'] += expansions_diff
                    stats[algo_name]['total_path_cost_increase'] += cost_diff
                    stats[algo_name]['path_cost_increases'].append(cost_diff)
                    if abs(path_cost - traditional_path_cost) < 1e-6:
                        stats[algo_name]['optimal_paths'] += 1
                    stats[algo_name]['total_queries'] += 1

                    # Track reopenings
                    stats[algo_name]['total_reopenings'] += reopenings
                    stats[algo_name]['reopenings_per_query'].append(reopenings)

                    # Add total expansions tracking
                    stats[algo_name]['total_expansions'] += expanded

                    # Save algorithm results for visualization
                    visualization_data['paths'][algo_name] = path
                    visualization_data['f_maps'][algo_name] = f_map_algo

            # Model-based A* variants
            for model_name, model in models.items():
                # Define variants for A* with Model
                model_variants = [
                    (f'{model_name} A* Allow Reopening=True', lambda: astar_with_model(start, goal, map_data, encoder, model, normalization_values, device, allow_reopening=True, max_expansions=300000)),
                    (f'{model_name} A* Allow Reopening=False', lambda: astar_with_model(start, goal, map_data, encoder, model, normalization_values, device, allow_reopening=False, max_expansions=300000)),
                    (f'{model_name} Focal A* ε=3.0 Allow Reopening=True', lambda: focal_astar_with_model(start, goal, map_data, encoder, model, normalization_values, device, epsilon=3.0, allow_reopening=True, max_expansions=300000)),
                    (f'{model_name} Focal A* ε=3.0 Allow Reopening=False', lambda: focal_astar_with_model(start, goal, map_data, encoder, model, normalization_values, device, epsilon=3.0, allow_reopening=False, max_expansions=300000)),
                ]

                for algo_name, algo_func in model_variants:
                    print(f"Running {algo_name} on query {query + 1}...")
                    path, path_cost, expanded, reopenings, f_star_map_algo = algo_func()

                    if path is None:
                        print(f"No path found for {algo_name} on query {query + 1}. Skipping...\n")
                        continue

                    print(f"{algo_name} expansions: {expanded}, cost: {path_cost}, reopenings: {reopenings}")
                    expansions_diff = 100 * (traditional_expanded - expanded) / traditional_expanded
                    cost_diff = 100 * (path_cost - traditional_path_cost) / traditional_path_cost

                    stats[algo_name]['total_expansions_reduction'] += expansions_diff
                    stats[algo_name]['total_path_cost_increase'] += cost_diff
                    stats[algo_name]['path_cost_increases'].append(cost_diff)
                    if abs(path_cost - traditional_path_cost) < 1e-6:
                        stats[algo_name]['optimal_paths'] += 1
                    stats[algo_name]['total_queries'] += 1

                    # Track reopenings
                    stats[algo_name]['total_reopenings'] += reopenings
                    stats[algo_name]['reopenings_per_query'].append(reopenings)

                    # Add total expansions tracking
                    stats[algo_name]['total_expansions'] += expanded

                    # Save algorithm results for visualization
                    visualization_data['paths'][algo_name] = path
                    visualization_data['f_maps'][algo_name] = f_star_map_algo

            # # Save comprehensive visualization for the current query
            # visualize_comparison(
            #     map_grid=map_data,
            #     start=start,
            #     goal=goal,
            #     paths=visualization_data['paths'],
            #     f_maps=visualization_data['f_maps'],
            #     output_dir=output_dir,
            #     run=query_counter
            # )

            query_counter += 1

            # Print cumulative results after each query
            print(f"\nCumulative Results after {query_counter} queries:")
            header = f"{'Algorithm':<40} {'Avg Exp Reduction (%)':<25} {'Path Cost Percent Increase (%)':<30} {'Avg ReOpenings':<20} {'Mean ReOpenings (%)':<25} {'Optimal Paths':<15} {'Total Queries':<15}"
            print(header)
            print("-" * len(header))
            for algo_name, data in stats.items():
                if data['total_queries'] > 0:
                    avg_exp_reduction = data['total_expansions_reduction'] / data['total_queries']
                    avg_cost_increase = data['total_path_cost_increase'] / data['total_queries']
                    std_cost_increase = np.std(data['path_cost_increases'])
                    avg_reopenings = data['total_reopenings'] / data['total_queries']
                    mean_reopen_percent = (data['total_reopenings'] / data['total_expansions'] * 100) if data['total_expansions'] > 0 else 0
                    optimal_paths = data['optimal_paths']
                    total_queries = data['total_queries']
                else:
                    avg_exp_reduction = 0
                    avg_cost_increase = 0
                    std_cost_increase = 0
                    avg_reopenings = 0
                    mean_reopen_percent = 0
                    optimal_paths = 0
                    total_queries = 0
                print(f"{algo_name:<40} {avg_exp_reduction:<25.2f} {avg_cost_increase:<30.2f} {avg_reopenings:<20.2f} {mean_reopen_percent:<25.2f} {optimal_paths:<15} {total_queries:<15}")
            print("\n" + "-"*200 + "\n")

    # ---------------------------
    # CSV Writing Enhancement
    # ---------------------------

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'Algorithm',
            'Avg_Expansions_Reduction',
            'Path_Cost_Percent_Increase',
            'Path_Cost_Percent_STD',
            'Avg_ReOpenings',
            'Mean_ReOpenings_Percent',
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
                avg_reopenings = data['total_reopenings'] / data['total_queries']
                mean_reopen_percent = (data['total_reopenings'] / data['total_expansions'] * 100) if data['total_expansions'] > 0 else 0
                optimal_paths = data['optimal_paths']
                total_queries = data['total_queries']
            else:
                avg_exp_reduction = 0
                avg_cost_increase = 0
                std_cost_increase = 0
                avg_reopenings = 0
                mean_reopen_percent = 0
                optimal_paths = 0
                total_queries = 0
            writer.writerow({
                'Algorithm': algo_name,
                'Avg_Expansions_Reduction': round(avg_exp_reduction, 2),
                'Path_Cost_Percent_Increase': round(avg_cost_increase, 2),
                'Path_Cost_Percent_STD': round(std_cost_increase, 2),
                'Avg_ReOpenings': round(avg_reopenings, 2),
                'Mean_ReOpenings_Percent': round(mean_reopen_percent, 2),
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
    # Initialize Encoder with latent_dim=1024
    encoder = UNet2DAutoencoder(input_channels=1, latent_dim=1024).to(device)  # Updated latent_dim to match main script
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    encoder.eval()

    # Initialize Models with appropriate input_size
    models = {}
    input_size = 1024 + 3 + 3 + 3 + 2  # encoded_map + start + goal + current + [g_normalized, h_normalized]
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
