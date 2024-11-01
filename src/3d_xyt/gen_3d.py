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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

# ---------------------------
# Global Orientation Mapping
# ---------------------------
# Updated to include 8 orientations: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
theta_to_index = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}
index_to_theta = {v: k for k, v in theta_to_index.items()}

# ---------------------------
# UNet2DAutoencoder Class
# ---------------------------
class UNet2DAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=1024):
        super(UNet2DAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Adjusted for 256x256 input size:
        # After pooling, the feature map size is reduced to 8x8
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, latent_dim)  # Changed from 256 * 16 * 16

        # Decoder fully connected layer
        self.fc2 = nn.Linear(latent_dim, 256 * 8 * 8)  # Changed from 256 * 16 * 16
        self.unflatten = nn.Unflatten(1, (256, 8, 8))  # Changed from (256, 16, 16)

        # Decoder
        self.up5 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec5 = self.conv_block(256, 256)

        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 32)

        # Final layer
        self.final = nn.Conv2d(32, input_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def encode(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        b = F.max_pool2d(b, 2)  # Additional pooling to reduce size to 8x8

        # Flatten and pass through fc1
        b_flat = self.flatten(b)
        latent_vector = self.fc1(b_flat)
        return latent_vector

    def decode(self, latent_vector):
        x = self.fc2(latent_vector)
        x = self.unflatten(x)

        d5 = self.up5(x)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return torch.sigmoid(out)

    def forward(self, x):
        latent_vector = self.encode(x)
        reconstruction = self.decode(latent_vector)
        return reconstruction

    def get_latent_vector(self, x):
        latent_vector = self.encode(x)
        return latent_vector



# ---------------------------
# Node Class for A*
# ---------------------------
class Node:
    def __init__(self, pos, theta, g=float('inf'), h=0, parent=None):
        self.pos = pos            # (x, y)
        self.theta = theta        # Orientation: 0, 45, 90, 135, 180, 225, 270, 315
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

# ---------------------------
# Room Dataclass
# ---------------------------
@dataclass
class Room:
    x: int
    y: int
    width: int
    height: int
    children: list = field(default_factory=list)

# ---------------------------
# Map Generation Function
# ---------------------------
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
                    continue  # Skip if unable to place after attempts

    # Add outer boundary walls
    # Top and bottom
    map_grid[0:wall_thickness, :] = 1
    map_grid[-wall_thickness:, :] = 1
    # Left and right
    map_grid[:, 0:wall_thickness] = 1
    map_grid[:, -wall_thickness:] = 1
    
    return map_grid

# ---------------------------
# Collision Checking Function
# ---------------------------
def is_valid(pos, map_grid):
    x, y, theta = pos
    robot_length = 12  # Length along the robot's facing direction
    robot_width = 8    # Width perpendicular to the facing direction

    # Adjust dimensions based on orientation
    if theta % 90 == 0:  # 0°, 90°, 180°, 270°
        dx = robot_length
        dy = robot_width
    else:  # 45°, 135°, 225°, 315°
        # For diagonal orientations, adjust the footprint to accommodate rotation
        # Approximate by increasing the footprint size
        dx = robot_length * np.sqrt(2) / 2 + robot_width * np.sqrt(2) / 2
        dy = robot_length * np.sqrt(2) / 2 + robot_width * np.sqrt(2) / 2

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

# ---------------------------
# Euclidean Distance Function
# ---------------------------
def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ---------------------------
# Heuristic Function with Orientation
# ---------------------------
def heuristic(current_pos, goal_pos, current_theta, goal_theta):
    distance = euclidean_distance(current_pos, goal_pos)
    theta_diff = abs(current_theta - goal_theta) % 360
    orientation_cost = min(theta_diff, 360 - theta_diff) / 45  # Normalize to [0, 8]
    return distance + orientation_cost

# ---------------------------
# A* Algorithm with Orientation
# ---------------------------
def astar(start, goal, map_grid):
    """
    Performs A* search considering orientation.

    Args:
        start (tuple): (x, y, theta)
        goal (tuple): (x, y, theta)
        map_grid (np.ndarray): 2D occupancy grid.

    Returns:
        np.ndarray: g_values map for all orientations if path found, else None.
    """
    # Define orientation indices
    start_theta_idx = theta_to_index[start[2]]
    goal_theta_idx = theta_to_index[goal[2]]

    # Initialize g_values as a 3D array (x, y, theta)
    g_values = np.full((map_grid.shape[0], map_grid.shape[1], 8), float('inf'))
    g_values[start[0], start[1], start_theta_idx] = 0

    # Initialize the start node
    start_node = Node(start[:2], start[2], g=0, h=euclidean_distance(start[:2], goal[:2]))
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    closed_set = set()

    while open_list:
        current_f, current = heapq.heappop(open_list)

        current_theta_idx = theta_to_index[current.theta]
        current_state = (current.pos, current.theta)

        # Check if goal is reached with the correct orientation
        if current.pos == goal[:2] and current.theta == goal[2]:
            logger.info(f"Goal reached at position {current.pos} with orientation {current.theta}")
            return g_values  # Return the full g_values array

        if (current.pos, current.theta) in closed_set:
            continue

        closed_set.add((current.pos, current.theta))

        for neighbor_pos, neighbor_theta, cost in get_motion_primitives(current, map_grid):
            if not is_valid((*neighbor_pos, neighbor_theta), map_grid):
                continue

            neighbor_theta_idx = theta_to_index[neighbor_theta]

            new_g = current.g + cost

            if new_g < g_values[neighbor_pos[0], neighbor_pos[1], neighbor_theta_idx]:
                g_values[neighbor_pos[0], neighbor_pos[1], neighbor_theta_idx] = new_g
                h = heuristic(neighbor_pos, goal[:2], neighbor_theta, goal[2])
                neighbor_node = Node(neighbor_pos, neighbor_theta, g=new_g, h=h, parent=current)
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))

    logger.warning(f"A* failed to find a path from {start[:2]} to {goal[:2]}")
    return None

# ---------------------------
# Motion Primitives Function
# ---------------------------
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
    motion_primitives.append(((next_x, next_y), next_theta, 1.0))  # Cost for moving forward

    # Rotate left by 45 degrees
    left_theta = (theta + 45) % 360
    motion_primitives.append(((x, y), left_theta, 0.5))  # Cost for rotation

    # Rotate right by 45 degrees
    right_theta = (theta - 45) % 360
    motion_primitives.append(((x, y), right_theta, 0.5))  # Cost for rotation

    return motion_primitives

# ---------------------------
# Generate Start and Goal Biased Function
# ---------------------------
def generate_start_goal(map_grid):
    height, width = map_grid.shape
    thetas = [0, 45, 90, 135, 180, 225, 270, 315]
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while attempts < max_attempts:
        start_x = random.randint(0, height - 1)
        start_y = random.randint(0, width - 1)
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

# ---------------------------
# Save F* Visualization Function
# ---------------------------
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

    save_path = os.path.join(f_star_dir, dataset_type,
                             f"fstar_map_{map_idx + 1}_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Process Map Function
# ---------------------------
def process_map(map_idx, encoded_map, map_grid, args):
    """
    Processes a single map by performing A* queries and writes the datasets to disk.

    Args:
        map_idx (int): Index of the map being processed.
        encoded_map (np.ndarray): Encoded representation of the map.
        map_grid (np.ndarray): 2D occupancy grid of the map.
        args (Namespace): Command-line arguments.

    Returns:
        tuple: (map_idx, success_flag)
    """
    try:
        # Initialize datasets
        datasets = {
            'vanilla': [],
            'exp': [],
            'mult': []
        }

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

                forward_g = astar(start, goal, map_grid)
                if forward_g is None:
                    logger.warning(
                        f"Map {map_idx + 1}: Forward A* failed for query {query_idx + 1}")
                    continue

                backward_g = astar(goal, start, map_grid)
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
            c_star = f_star[goal[0], goal[1], theta_to_index[goal[2]]]

            # Vanilla: No penalty
            vanilla_f_star = f_star.copy()

            # Exponential penalty
            finite_mask = np.isfinite(f_star)
            exp_penalty_factor = np.exp(
                (f_star[finite_mask] - c_star) / c_star)
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
                        logger.info(f"Created directory: {dataset_type_dir}")
                    save_fstar_visualization(
                        map_grid, start, goal, modified_f_star, map_idx, f_star_dir, dataset_type)
                first_query = False

                valid_positions = np.argwhere(np.isfinite(modified_f_star))

                # ---------------------------
                # Sampling to Limit to 5000 Points
                # ---------------------------
                if len(valid_positions) > 5000:
                    sampled_indices = random.sample(range(len(valid_positions)), 5000)
                    sampled_positions = valid_positions[sampled_indices]
                else:
                    sampled_positions = valid_positions

                for pos in sampled_positions:
                    x, y, theta_idx = pos
                    current_theta = index_to_theta[theta_idx]
                    current_theta_idx = theta_idx

                    g_star = forward_g[x, y, theta_to_index[start[2]]]
                    h = euclidean_distance((x, y), goal[:2])
                    f_star_value = modified_f_star[x, y, current_theta_idx]

                    if np.isinf(g_star) or np.isinf(h) or np.isinf(f_star_value):
                        continue

                    target_value = f_star_value

                    current = (x, y, current_theta)

                    datasets[dataset_type].append((
                        encoded_map,
                        start,          # (x, y, theta)
                        goal,           # (x, y, theta)
                        current,        # (x, y, theta)
                        g_star,
                        h,
                        target_value
                    ))


        # After processing all queries, write datasets to disk
        for dataset_type in ['vanilla', 'exp', 'mult']:
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
        description="Create Datasets for F* Prediction Model with Different Penalties")
    parser.add_argument("encoder_path", type=str,
                        help="Path to the pre-trained UNet2DAutoencoder model")
    parser.add_argument("--num_maps", type=int, default=2,
                        help="Number of maps to generate")
    parser.add_argument("--num_queries_per_map", type=int,
                        default=1, help="Number of queries per map")
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
    parser.add_argument("--min_room_size", type=int,
                        default=100, help="Minimum size of a room")
    parser.add_argument("--max_room_size", type=int,
                        default=150, help="Maximum size of a room")
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Maximum recursion depth for splitting rooms")
    parser.add_argument("--wall_thickness", type=int, default=3,
                        help="Thickness of the walls between rooms")
    parser.add_argument("--min_openings", type=int, default=2,
                        help="Minimum number of openings per wall")
    parser.add_argument("--max_openings", type=int, default=3,
                        help="Maximum number of openings per wall")
    parser.add_argument("--min_opening_size", type=int, default=10,
                        help="Minimum size of each opening in pixels")
    parser.add_argument("--max_opening_size", type=int, default=15,
                        help="Maximum size of each opening in pixels")
    parser.add_argument("--min_obstacles", type=int, default=4,
                        help="Minimum number of obstacles per room")
    parser.add_argument("--max_obstacles", type=int, default=5,
                        help="Maximum number of obstacles per room")
    parser.add_argument("--min_obstacle_size", type=int,
                        default=10, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int,
                        default=15, help="Maximum size of each obstacle")
    parser.add_argument("--obstacle_attempts", type=int, default=10,
                        help="Number of attempts to place an obstacle without overlap")
    parser.add_argument("--trap_probability", type=float, default=0.0,
                        help="Probability of placing a concave trap instead of a regular obstacle")
    parser.add_argument("--latent_dim", type=int, default=1024,
                        help="Dimensionality of the latent space")
    return parser.parse_args()

# ---------------------------
# Normalization Utility Functions
# ---------------------------
def load_and_combine_datasets(dataset_type, save_dataset_dir):
    """
    Loads all individual raw datasets of a given type and combines them into a single global dataset.

    Args:
        dataset_type (str): The type of dataset ('vanilla', 'exp', 'mult').
        save_dataset_dir (str): Directory where the raw datasets are saved.

    Returns:
        tuple: (Combined global dataset list, List of dataset file paths to delete)
    """
    try:
        dataset_files = [
            os.path.join(save_dataset_dir, f)
            for f in os.listdir(save_dataset_dir)
            if f.endswith(f"_{dataset_type}.pkl") and not f.startswith('normalized_')
        ]

        if not dataset_files:
            logger.warning(f"No dataset files found for '{dataset_type}'.")
            return [], []

        global_dataset = []
        for dataset_file in dataset_files:
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
                global_dataset.extend(dataset)
        logger.info(f"Loaded and combined {len(dataset_files)} datasets for '{dataset_type}'.")
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
        if not global_dataset:
            logger.error("Global dataset is empty. Cannot compute min/max values.")
            return None

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
# Main Function
# ---------------------------
def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load the encoder model once in the main thread
    try:
        logger.info("Loading encoder model in the main thread.")
        encoder = UNet2DAutoencoder(
            input_channels=1, latent_dim=args.latent_dim).to(device)
        encoder.load_state_dict(torch.load(
            args.encoder_path, map_location=device))
        encoder.eval()
        logger.info("Encoder model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load encoder model: {e}")
        return

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

    for dataset_type in ['vanilla', 'exp', 'mult']:
        f_star_dataset_dir = os.path.join(args.f_star_dir, dataset_type)
        if not os.path.exists(f_star_dataset_dir):
            os.makedirs(f_star_dataset_dir)
            logger.info(f"Created F* dataset directory: {f_star_dataset_dir}")

    # Generate and encode all maps
    encoded_maps = []
    map_grids = []
    for map_idx in range(args.num_maps):
        logger.info(
            f"Generating and encoding map {map_idx + 1}/{args.num_maps}")
        try:
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
            logger.info(f"Map {map_idx + 1} generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate map {map_idx + 1}: {e}")
            encoded_maps.append(None)
            map_grids.append(None)
            continue

        try:
            # Encode the map
            with torch.no_grad():
                map_tensor = torch.from_numpy(
                    map_grid).float().unsqueeze(0).unsqueeze(0).to(device)
                encoded_map = encoder.encode(map_tensor).cpu().numpy().flatten()

            logger.info(f"Map {map_idx + 1} encoded successfully.")
            encoded_maps.append(encoded_map)
            map_grids.append(map_grid)
        except Exception as e:
            logger.error(f"Failed to encode map {map_idx + 1}: {e}")
            encoded_maps.append(None)
            map_grids.append(None)

    # Prepare arguments for worker processes
    worker_args = []
    for map_idx, (encoded_map, map_grid) in enumerate(zip(encoded_maps, map_grids)):
        if encoded_map is not None and map_grid is not None:
            worker_args.append((map_idx, encoded_map, map_grid, args))
        else:
            logger.warning(
                f"Skipping map {map_idx + 1} due to previous errors.")

    # Initialize multiprocessing pool to handle A*-related workload
    try:
        pool_size = min(70, cpu_count())
        with Pool(processes=pool_size) as pool:
            logger.info(
                f"Starting multiprocessing pool with {len(worker_args)} maps using {pool_size} processes.")
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
    for dataset_type in ['vanilla', 'exp', 'mult']:
        logger.info(f"Starting global normalization for '{dataset_type}' dataset.")

        # Load and combine all raw datasets
        global_dataset, dataset_files_to_delete = load_and_combine_datasets(dataset_type, args.save_dataset_dir)
        if global_dataset is None or not global_dataset:
            logger.warning(f"Failed to load datasets for '{dataset_type}' or dataset is empty. Skipping normalization.")
            continue

        # Compute global min and max values
        normalization_values = compute_global_min_max(global_dataset)
        if normalization_values is None:
            logger.warning(f"Failed to compute global min/max for '{dataset_type}'. Skipping normalization.")
            continue

        # Save global normalization parameters
        norm_save_path = os.path.join(
            args.norm_save_dir, f"{dataset_type}_global_normalization_values.pkl")
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
        global_dataset_filename = f"global_normalized_dataset_{dataset_type}.pkl"
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
# Run Main Function
# ---------------------------
if __name__ == '__main__':
    main()
