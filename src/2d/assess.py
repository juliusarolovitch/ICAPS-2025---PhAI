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
from models import MLPModel
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import random
import itertools
from dataclasses import dataclass, field
from scipy.ndimage import binary_dilation

# ---------------------------
# Neural Network Models
# ---------------------------

# class UNet2DAutoencoder(nn.Module):
#     def __init__(self, input_channels=1, latent_dim=512):
#         super(UNet2DAutoencoder, self).__init__()

#         self.latent_dim = latent_dim

#         # Encoder
#         self.enc1 = self.conv_block(input_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.enc4 = self.conv_block(128, 256)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True)
#         )

#         # Flatten and fully connected layer for encoding
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 32 * 32, latent_dim)  # Updated

#         # Decoder fully connected layer
#         self.fc2 = nn.Linear(latent_dim, 256 * 32 * 32)  # Updated
#         self.unflatten = nn.Unflatten(1, (256, 32, 32))  # Updated

#         # Decoder
#         self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.dec4 = self.conv_block(256, 256)

#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = self.conv_block(128, 128)

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = self.conv_block(64, 64)

#         self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = self.conv_block(32, 32)

#         # Final layer
#         self.final = nn.Conv2d(32, input_channels, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#     def encode(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(F.max_pool2d(e1, 2))  # 256x256
#         e3 = self.enc3(F.max_pool2d(e2, 2))  # 128x128
#         e4 = self.enc4(F.max_pool2d(e3, 2))  # 64x64

#         # Bottleneck
#         b = self.bottleneck(F.max_pool2d(e4, 2))  # 32x32

#         # Flatten and pass through fc1
#         b_flat = self.flatten(b)  # [B, 256*32*32] = [B, 262144]
#         latent_vector = self.fc1(b_flat)  # [B, latent_dim]

#         return latent_vector

#     def decode(self, latent_vector):
#         # Pass through fc2 and reshape
#         x = self.fc2(latent_vector)  # [B, 256*32*32] = [B, 262144]
#         x = self.unflatten(x)        # [B, 256, 32, 32]

#         # Decoder
#         d4 = self.up4(x)  # [B, 256, 64, 64]
#         d4 = self.dec4(d4)  # [B, 256, 64, 64]

#         d3 = self.up3(d4)  # [B, 128, 128, 128]
#         d3 = self.dec3(d3)  # [B, 128, 128, 128]

#         d2 = self.up2(d3)  # [B, 64, 256, 256]
#         d2 = self.dec2(d2)  # [B, 64, 256, 256]

#         d1 = self.up1(d2)  # [B, 32, 512, 512]
#         d1 = self.dec1(d1)  # [B, 32, 512, 512]

#         # Final upsampling to match original size
#         d0 = F.interpolate(d1, scale_factor=1, mode='bilinear', align_corners=True)  # No scaling needed

#         # Final output
#         out = self.final(d0)  # [B, 1, 512, 512]
#         return torch.sigmoid(out)

#     def forward(self, x):
#         latent_vector = self.encode(x)
#         reconstruction = self.decode(latent_vector)
#         return reconstruction

#     def get_latent_vector(self, x):
#         latent_vector = self.encode(x)
#         return latent_vector

class UNet2DAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=1):
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

        # Flatten and fully connected layer for encoding
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder fully connected layer
        self.fc2 = nn.Linear(latent_dim, 256 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (256, 8, 8))

        # Decoder
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
        b = self.bottleneck(F.max_pool2d(e4, 2))  # [B, 256, 8, 8]

        # Flatten and pass through fc1
        b_flat = self.flatten(b)  # [B, 256*8*8] = [B, 16384]
        latent_vector = self.fc1(b_flat)  # [B, latent_dim]

        return latent_vector

    def decode(self, latent_vector):
        # Pass through fc2 and reshape
        x = self.fc2(latent_vector)  # [B, 16384]
        x = self.unflatten(x)        # [B, 256, 8, 8]

        # Decoder
        d4 = self.up4(x)  # [B, 256, 16, 16]
        d4 = self.dec4(d4)  # [B, 256, 16, 16]

        d3 = self.up3(d4)  # [B, 128, 32, 32]
        d3 = self.dec3(d3)  # [B, 128, 32, 32]

        d2 = self.up2(d3)  # [B, 64, 64, 64]
        d2 = self.dec2(d2)  # [B, 64, 64, 64]

        d1 = self.up1(d2)  # [B, 32, 128, 128]
        d1 = self.dec1(d1)  # [B, 32, 128, 128]

        # Final output
        out = self.final(d1)  # [B, 1, 128, 128]
        return torch.sigmoid(out)

    def forward(self, x):
        latent_vector = self.encode(x)
        reconstruction = self.decode(latent_vector)
        return reconstruction


    def get_latent_vector(self, x):
        """
        Extracts the latent vector from the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W].
        
        Returns:
            torch.Tensor: Latent vector of shape [B, latent_dim].
        """
        latent_vector = self.encode(x)
        return latent_vector

class UNet3DAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=512):
        super(UNet3DAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True)
        )

        # Flatten and fully connected layer for encoding
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, latent_dim)  # 16384 -> latent_dim

        # Decoder fully connected layer
        self.fc2 = nn.Linear(latent_dim, 256 * 4 * 4 * 4)  # latent_dim -> 16384
        self.unflatten = nn.Unflatten(1, (256, 4, 4, 4))

        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256 + 256, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128 + 128, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64 + 64, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32 + 32, 32)

        # Final layer
        self.final = nn.Conv3d(32, input_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))

        b = self.bottleneck(F.max_pool3d(e4, 2))

        b_flat = self.flatten(b)
        latent_vector = self.fc1(b_flat)

        return latent_vector, (e1, e2, e3, e4)

    def decode(self, latent_vector, enc_features):
        x = self.fc2(latent_vector)  
        x = self.unflatten(x) 
        e1, e2, e3, e4 = enc_features

        d4 = self.up4(x)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return torch.sigmoid(out)

    def forward(self, x):
        latent_vector, enc_features = self.encode(x)
        reconstruction = self.decode(latent_vector, enc_features)
        return reconstruction

    def get_latent_vector(self, x):
        """
        Extracts the latent vector from the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, D, H, W].
        
        Returns:
            torch.Tensor: Latent vector of shape [B, latent_dim].
        """
        latent_vector, _ = self.encode(x)
        return latent_vector

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
    width=128,
    height=128,
    min_room_size=20,
    max_room_size=40,
    max_depth=5,
    wall_thickness=2,
    min_openings=1,
    max_openings=2,
    opening_size=4,
    min_obstacles=5,
    max_obstacles=8,
    min_obstacle_size=7,
    max_obstacle_size=10
):
    """
    Generates a 2D map with randomly sized and positioned rooms separated by walls with random openings.
    Each room contains a random number of smaller obstacles.

    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        min_room_size (int): Minimum size of a room.
        max_room_size (int): Maximum size of a room.
        max_depth (int): Maximum recursion depth for splitting rooms.
        wall_thickness (int): Thickness of the walls between rooms.
        min_openings (int): Minimum number of openings per wall.
        max_openings (int): Maximum number of openings per wall.
        opening_size (int): Size of each opening in pixels.
        min_obstacles (int): Minimum number of obstacles per room.
        max_obstacles (int): Maximum number of obstacles per room.
        min_obstacle_size (int): Minimum size of each obstacle.
        max_obstacle_size (int): Maximum size of each obstacle.

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
            possible_positions = wall_length - opening_size
            if possible_positions <= 0:
                return
            for _ in range(num_openings):
                opening_start = random.randint(start[1], start[1] + possible_positions)
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        else:
            wall_length = end[0] - start[0]
            possible_positions = wall_length - opening_size
            if possible_positions <= 0:
                return
            for _ in range(num_openings):
                opening_start = random.randint(start[0], start[0] + possible_positions)
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
            obstacle_w = random.randint(min_obstacle_size, max_obstacle_size)
            obstacle_h = random.randint(min_obstacle_size, max_obstacle_size)
            # Ensure obstacle fits within the room with some padding
            if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
                continue  # Skip if obstacle is too big for the room
            obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
            obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
            # Avoid placing obstacles on walls
            map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 1
    
    # Optionally, add outer boundary walls
    # Top and bottom
    map_grid[0:wall_thickness, :] = 1
    map_grid[-wall_thickness:, :] = 1
    # Left and right
    map_grid[:, 0:wall_thickness] = 1
    map_grid[:, -wall_thickness:] = 1
    
    return map_grid

# def generate_map(
#     width=512,
#     height=512,
#     min_room_size=60,
#     max_room_size=120,
#     max_depth=5,
#     wall_thickness=5,
#     min_openings=1,
#     max_openings=2,
#     min_opening_size=10,
#     max_opening_size=20,
#     min_obstacles=4,
#     max_obstacles=14,
#     min_obstacle_size=10,
#     max_obstacle_size=20,
#     obstacle_attempts=10,
#     trap_probability=0.4
# ):
#     """
#     Generates a 2D map with rooms and walls with openings.
#     Adds rectangular obstacles and concave traps without overlapping.

#     Returns:
#         np.ndarray: 2D occupancy map of shape [height, width].
#     """
#     map_grid = np.zeros((height, width), dtype=np.float32)

#     root_room = Room(0, 0, width, height)

#     def split_room(room, depth):
#         if depth >= max_depth:
#             return
#         can_split_horizontally = room.height >= 2 * min_room_size + wall_thickness
#         can_split_vertically = room.width >= 2 * min_room_size + wall_thickness

#         if not can_split_horizontally and not can_split_vertically:
#             return  # Cannot split further

#         if can_split_horizontally and can_split_vertically:
#             split_horizontally = random.choice([True, False])
#         elif can_split_horizontally:
#             split_horizontally = True
#         else:
#             split_horizontally = False

#         if split_horizontally:
#             split_min = room.y + min_room_size
#             split_max = room.y + room.height - min_room_size - wall_thickness
#             if split_max <= split_min:
#                 return  # Not enough space to split
#             split_pos = random.randint(split_min, split_max)
#             child1 = Room(room.x, room.y, room.width, split_pos - room.y)
#             child2 = Room(room.x, split_pos + wall_thickness, room.width,
#                           room.y + room.height - split_pos - wall_thickness)
#             map_grid[split_pos:split_pos + wall_thickness,
#                      room.x:room.x + room.width] = 1
#             add_openings((split_pos, room.x), (split_pos, room.x +
#                              room.width), orientation='horizontal')
#         else:
#             split_min = room.x + min_room_size
#             split_max = room.x + room.width - min_room_size - wall_thickness
#             if split_max <= split_min:
#                 return
#             split_pos = random.randint(split_min, split_max)
#             child1 = Room(room.x, room.y, split_pos - room.x, room.height)
#             child2 = Room(split_pos + wall_thickness, room.y, room.x +
#                           room.width - split_pos - wall_thickness, room.height)
#             map_grid[room.y:room.y + room.height,
#                      split_pos:split_pos + wall_thickness] = 1
#             add_openings((room.y, split_pos), (room.y + room.height,
#                              split_pos), orientation='vertical')

#         room.children = [child1, child2]
#         split_room(child1, depth + 1)
#         split_room(child2, depth + 1)

#     def add_openings(start, end, orientation='horizontal'):
#         """
#         Adds random openings to a wall.

#         Args:
#             start (tuple): Starting coordinate (y, x).
#             end (tuple): Ending coordinate (y, x).
#             orientation (str): 'horizontal' or 'vertical'.
#         """
#         num_openings = random.randint(min_openings, max_openings)
#         if orientation == 'horizontal':
#             wall_length = end[1] - start[1]
#             if wall_length <= min_opening_size:
#                 return
#             for _ in range(num_openings):
#                 opening_size = random.randint(
#                     min_opening_size, max_opening_size)
#                 opening_size = min(opening_size, wall_length)
#                 opening_start = random.randint(start[1], end[1] - opening_size)
#                 map_grid[start[0]:start[0] + wall_thickness,
#                          opening_start:opening_start + opening_size] = 0
#         else:
#             wall_length = end[0] - start[0]
#             if wall_length <= min_opening_size:
#                 return
#             for _ in range(num_openings):
#                 opening_size = random.randint(
#                     min_opening_size, max_opening_size)
#                 opening_size = min(opening_size, wall_length)
#                 opening_start = random.randint(start[0], end[0] - opening_size)
#                 map_grid[opening_start:opening_start + opening_size,
#                          start[1]:start[1] + wall_thickness] = 0

#     def place_concave_trap(room):
#         """
#         Places a concave trap within the given room. The trap can be L-shaped or triangular.

#         Args:
#             room (Room): The room where the trap will be placed.

#         Returns:
#             bool: True if the trap was successfully placed, False otherwise.
#         """
#         trap_type = random.choice(['L', 'triangle'])
#         if trap_type == 'L':
#             return place_L_shaped_trap(room)
#         else:
#             return place_triangular_trap(room)

#     def place_L_shaped_trap(room):
#         """
#         Places an L-shaped concave trap within the given room.

#         Args:
#             room (Room): The room where the trap will be placed.

#         Returns:
#             bool: True if the trap was successfully placed, False otherwise.
#         """
#         trap_size = random.randint(min_obstacle_size, max_obstacle_size)
#         trap_thickness = wall_thickness

#         if (trap_size * 2 + wall_thickness) > room.width or (trap_size * 2 + wall_thickness) > room.height:
#             return False

#         corner_x = random.randint(
#             room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
#         corner_y = random.randint(
#             room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)

#         orientation = random.choice(['left', 'right', 'up', 'down'])

#         if orientation == 'left':
#             arm1 = ((corner_y, corner_x - trap_size),
#                     (trap_size, trap_thickness))
#             arm2 = ((corner_y - trap_size, corner_x),
#                     (trap_thickness, trap_size))
#         elif orientation == 'right':
#             arm1 = ((corner_y, corner_x), (trap_size, trap_thickness))
#             arm2 = ((corner_y - trap_size, corner_x + trap_size -
#                     trap_thickness), (trap_thickness, trap_size))
#         elif orientation == 'up':
#             arm1 = ((corner_y - trap_size, corner_x),
#                     (trap_thickness, trap_size))
#             arm2 = ((corner_y - trap_size, corner_x - trap_size),
#                     (trap_size, trap_thickness))
#         else:  # 'down'
#             arm1 = ((corner_y, corner_x), (trap_thickness, trap_size))
#             arm2 = ((corner_y + trap_size - trap_thickness, corner_x +
#                     trap_size - trap_thickness), (trap_size, trap_thickness))

#         (y1, x1), (h1, w1) = arm1
#         (y2, x2), (h2, w2) = arm2

#         if (x1 < 0 or y1 < 0 or x1 + w1 > width or y1 + h1 > height or
#                 x2 < 0 or y2 < 0 or x2 + w2 > width or y2 + h2 > height):
#             return False

#         if (np.any(map_grid[y1:y1 + h1, x1:x1 + w1] == 1) or
#                 np.any(map_grid[y2:y2 + h2, x2:x2 + w2] == 1)):
#             return False

#         map_grid[y1:y1 + h1, x1:x1 + w1] = 1
#         map_grid[y2:y2 + h2, x2:x2 + w2] = 1

#         return True

#     def place_triangular_trap(room):
#         """
#         Places a triangular concave trap within the given room.

#         Args:
#             room (Room): The room where the trap will be placed.

#         Returns:
#             bool: True if the trap was successfully placed, False otherwise.
#         """
#         return False  # Implement triangular traps if needed

#     split_room(root_room, 0)

#     leaf_rooms = []

#     def collect_leaf_rooms(room):
#         if not room.children:
#             leaf_rooms.append(room)
#         else:
#             for child in room.children:
#                 collect_leaf_rooms(child)

#     collect_leaf_rooms(root_room)

#     for room in leaf_rooms:
#         num_obstacles = random.randint(min_obstacles, max_obstacles)
#         for _ in range(num_obstacles):
#             if random.random() < trap_probability:
#                 placed = False
#                 for attempt in range(obstacle_attempts):
#                     if place_concave_trap(room):
#                         placed = True
#                         break
#                 if not placed:
#                     pass
#                 continue
#             else:
#                 placed = False
#                 for attempt in range(obstacle_attempts):
#                     obstacle_w = random.randint(
#                         min_obstacle_size, max_obstacle_size)
#                     obstacle_h = random.randint(
#                         min_obstacle_size, max_obstacle_size)
#                     if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
#                         continue  # Skip if obstacle is too big for the room
#                     obstacle_x = random.randint(
#                         room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
#                     obstacle_y = random.randint(
#                         room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
#                     if np.any(map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] == 1):
#                         continue  # Overlaps with existing obstacle
#                     map_grid[obstacle_y:obstacle_y + obstacle_h,
#                              obstacle_x:obstacle_x + obstacle_w] = 1
#                     placed = True
#                     break  # Successfully placed
#                 if not placed:
#                     pass

#     map_grid[0:wall_thickness, :] = 1
#     map_grid[-wall_thickness:, :] = 1
#     map_grid[:, 0:wall_thickness] = 1
#     map_grid[:, -wall_thickness:] = 1

#     return map_grid

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
                                  cmap='viridis', s=4, alpha=0.8)
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
        if current.pos in closed_set:
            continue  # Skip if already expanded
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


def focal_astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, epsilon=3.0, max_expansions=100000):
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

# ---------------------------
# Heuristic Functions Generation for MHA*
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
# Potential Search
# ---------------------------

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
                    next_node = PotentialNode(pos=next_pos, g=tentative_g, h=h, flnr=flnr, parent=current)
                    heapq.heappush(open_list, next_node)
                else:
                    continue  # Skip nodes where C - tentative_g <= 0

    return None, None, expansions, flnr_map

# ---------------------------
# Traditional A* Search
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
# Weighted A* Search
# ---------------------------

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


# ---------------------------
# IMHA* Search
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

# ---------------------------
# SMHA* Search
# ---------------------------

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
# Assessment Function
# ---------------------------

def run_assessment(encoder, models, normalization_values, device, num_maps=1, num_queries_per_map=10, output_csv="output.csv", output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define all algorithms including model-based ones
    algorithms = ['Traditional A*', 'Weighted A* ε=3.0', 'Potential Search', 'IMHA*', 'SMHA*']
    for model_name in models:
        algorithms.append(f'{model_name} A*')
        algorithms.append(f'{model_name} Focal A* ε=3.0')

    # Initialize statistics dictionary
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
            print(f"\nRunning assessment on query {query + 1} for map {map_idx + 1}...\n")

            # Traditional A*
            print(f"Running Traditional A* on query {query + 1}...")
            while True:
                start, goal = generate_start_goal(map_data)
                traditional_path, traditional_path_cost, traditional_expanded, f_map = astar_traditional(start, goal, map_data)
                if traditional_path is not None and traditional_expanded > 100:
                    break

            # Store results for visualization
            visualization_data = {
                'start': start,
                'goal': goal,
                'paths': {},
                'f_maps': {}
            }

            if traditional_path_cost is None:
                print(f"No path found for Traditional A* on query {query + 1}. Skipping...\n")
                continue

            print(f"Traditional A* expansions: {traditional_expanded}, cost: {traditional_path_cost}")

            # Save Traditional A* results for visualization
            visualization_data['paths']['Traditional A*'] = traditional_path
            visualization_data['f_maps']['Traditional A*'] = f_map

            # Weighted A* ε=3.0
            epsilon = 3.0
            algo_name = f'Weighted A* ε={epsilon}'
            print(f"Running {algo_name} on query {query + 1}...")
            path, path_cost, expanded, f_map_weighted = astar_weighted(start, goal, map_data, epsilon=epsilon)

            if path_cost is None:
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

                # Save Weighted A* results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = f_map_weighted

            # Potential Search
            print(f"Running Potential Search on query {query + 1}...")
            C = traditional_path_cost * 3.0  # Set to 3 times the optimal cost for suboptimality factor of 3
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

                # Save Potential Search results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = flnr_map

            # Generate Heuristics for MHA* based algorithms
            heuristic_functions = generate_mha_heuristics(map_data, goal, start)

            # IMHA*
            print(f"Running IMHA* on query {query + 1}...")
            algo_name = 'IMHA*'
            path, path_cost, expanded, f_map_imha = imha_star(start, goal, map_data, heuristic_functions)

            if path_cost is None:
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

                # Save IMHA* results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = f_map_imha

            # SMHA*
            print(f"Running SMHA* on query {query + 1}...")
            algo_name = 'SMHA*'
            path, path_cost, expanded, f_map_smha = smha_star(start, goal, map_data, heuristic_functions)

            if path_cost is None:
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

                # Save SMHA* results for visualization
                visualization_data['paths'][algo_name] = path
                visualization_data['f_maps'][algo_name] = f_map_smha

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

                # Save Model-based A* results for visualization
                visualization_data['paths'][algo_name] = model_path
                visualization_data['f_maps'][algo_name] = f_star_map

                # Focal A* with ε=3.0
                algo_name = f'{model_name} Focal A* ε=3.0'
                print(f"Running {algo_name} on query {query + 1}...")
                focal_path, focal_path_cost, focal_expanded, f_star_map_focal = focal_astar_with_model(start, goal, map_data, encoder, model, normalization_values, device, epsilon=3.0)

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
            print(f"{'Algorithm':<30} {'Avg Exp Reduction (%)':<25} {'Path_Cost_Percent_Increase (%)':<30} {'Path_Cost_Percent_STD (%)':<30} {'Optimal Paths':<15} {'Total Queries':<15}")
            for algo_name, data in stats.items():
                if data['total_queries'] > 0:
                    avg_exp_reduction = data['total_expansions_reduction'] / data['total_queries']
                    avg_cost_increase = data['total_path_cost_increase'] / data['total_queries']
                    std_cost_increase = np.std(data['path_cost_increases'])
                    optimal_paths = data['optimal_paths']
                    total_queries = data['total_queries']
                    print(f"{algo_name:<30} {avg_exp_reduction:<25.2f} {avg_cost_increase:<30.2f} {std_cost_increase:<30.2f} {optimal_paths:<15} {total_queries:<15}")
            print("\n" + "-"*120 + "\n")


    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Avg_Expansions_Reduction', 'Path_Cost_Percent_Increase', 'Path_Cost_Percent_STD', 'Optimal_Paths', 'Total_Queries']
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
                    'Path_Cost_Percent_Increase': avg_cost_increase,
                    'Path_Cost_Percent_STD': std_cost_increase,
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
    input_size = (3 * 2) + 512 + 2  # start, goal, current (3 positions * 2 coordinates) + latent_dim=1024 + 2 heuristic values
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
