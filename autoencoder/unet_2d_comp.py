import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import os  # For directory operations

# ---------------------------
# Map Generation
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

# ---------------------------
# Dataset Class
# ---------------------------

class Grid2DDataset(Dataset):
    def __init__(self, maps, augment=False, pad=False, multiple=16):
        """
        Initializes the dataset with 2D occupancy grids.
        
        Args:
            maps (list of np.ndarray): List of 2D occupancy maps.
            augment (bool): Whether to apply data augmentation.
            pad (bool): Whether to pad the maps to the nearest multiple.
            multiple (int): The multiple to pad to.
        """
        self.maps = maps
        self.augment = augment
        self.pad = pad
        self.multiple = multiple

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        grid = self.maps[idx]
        if self.augment:
            # Apply augmentations if desired (e.g., random rotations, flips)
            grid = self.augment_map(grid)
        grid = torch.from_numpy(grid).unsqueeze(0).float()  # [1, H, W]
        if self.pad:
            grid = pad_tensor(grid, self.multiple)
        return grid

    def augment_map(self, grid):
        """
        Applies random augmentations to a 2D map.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            grid = np.flip(grid, axis=1)
        # Random vertical flip
        if random.random() > 0.5:
            grid = np.flip(grid, axis=0)
        # Random rotation
        k = random.randint(0, 3)
        grid = np.rot90(grid, k)
        return grid

def pad_tensor(x, multiple):
    """
    Pads a 2D tensor to make its spatial dimensions multiples of 'multiple'.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W].
        multiple (int): The multiple to pad to.
    
    Returns:
        torch.Tensor: Padded tensor.
    """
    _, _, H, W = x.size()
    pad_height = (multiple - H % multiple) % multiple
    pad_width = (multiple - W % multiple) % multiple
    # Pad in the order of (left, right, top, bottom)
    padding = (0, pad_width, 0, pad_height)
    return F.pad(x, padding)

# ---------------------------
# 2D U-Net Autoencoder
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

        # Flatten and fully connected layer for encoding
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 32 * 32, latent_dim)  # Updated

        # Decoder fully connected layer
        self.fc2 = nn.Linear(latent_dim, 256 * 32 * 32)  # Updated
        self.unflatten = nn.Unflatten(1, (256, 32, 32))  # Updated

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
        e2 = self.enc2(F.max_pool2d(e1, 2))  # 256x256
        e3 = self.enc3(F.max_pool2d(e2, 2))  # 128x128
        e4 = self.enc4(F.max_pool2d(e3, 2))  # 64x64

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))  # 32x32

        # Flatten and pass through fc1
        b_flat = self.flatten(b)  # [B, 256*32*32] = [B, 262144]
        latent_vector = self.fc1(b_flat)  # [B, latent_dim]

        return latent_vector

    def decode(self, latent_vector):
        # Pass through fc2 and reshape
        x = self.fc2(latent_vector)  # [B, 256*32*32] = [B, 262144]
        x = self.unflatten(x)        # [B, 256, 32, 32]

        # Decoder
        d4 = self.up4(x)  # [B, 256, 64, 64]
        d4 = self.dec4(d4)  # [B, 256, 64, 64]

        d3 = self.up3(d4)  # [B, 128, 128, 128]
        d3 = self.dec3(d3)  # [B, 128, 128, 128]

        d2 = self.up2(d3)  # [B, 64, 256, 256]
        d2 = self.dec2(d2)  # [B, 64, 256, 256]

        d1 = self.up1(d2)  # [B, 32, 512, 512]
        d1 = self.dec1(d1)  # [B, 32, 512, 512]

        # Final upsampling to match original size
        d0 = F.interpolate(d1, scale_factor=1, mode='bilinear', align_corners=True)  # No scaling needed
        # Optionally, add another conv_block here if desired
        # d0 = self.conv_block(32, 32)(d0)  # Uncomment if needed

        # Final output
        out = self.final(d0)  # [B, 1, 512, 512]
        return torch.sigmoid(out)

    def forward(self, x):
        latent_vector = self.encode(x)
        reconstruction = self.decode(latent_vector)
        return reconstruction

    def get_latent_vector(self, x):
        latent_vector = self.encode(x)
        return latent_vector


# ---------------------------
# Loss Function
# ---------------------------

# Using Binary Cross-Entropy Loss for occupancy
def voxel_loss(recon, target):
    return F.binary_cross_entropy(recon, target)

# ---------------------------
# Training and Validation Functions
# ---------------------------

def train_autoencoder(model, train_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)  # [B, 1, H, W]
        optimizer.zero_grad()

        # Forward pass
        recon_batch = model(batch)  # [B, 1, H, W]

        # Compute loss
        loss = loss_fn(recon_batch, batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_autoencoder(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            batch = batch.to(device)  # [B, 1, H, W]

            # Forward pass
            recon_batch = model(batch)  # [B, 1, H, W]

            # Compute loss
            loss = loss_fn(recon_batch, batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def visualize_reconstruction(model, dataset, device, num_samples=5, epoch=0):
    """
    Visualizes original and reconstructed 2D maps.

    Args:
        model (nn.Module): Trained autoencoder model.
        dataset (Dataset): Validation dataset.
        device (torch.device): Device to run on.
        num_samples (int): Number of samples to visualize.
        epoch (int): Current epoch number.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    with torch.no_grad():
        for idx in indices:
            original = dataset[idx].unsqueeze(0).to(device)  # [1, 1, H, W]
            reconstructed = model(original)  # [1, 1, H, W]
            original = original.cpu().numpy()[0, 0]
            reconstructed = reconstructed.cpu().numpy()[0, 0]

            # Plot original and reconstructed maps side by side
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            axes[0].imshow(original, cmap='Greys', origin='upper')
            axes[0].set_title("Original 2D Map")
            axes[0].axis('off')

            axes[1].imshow(reconstructed, cmap='Greys', origin='upper')
            axes[1].set_title("Reconstructed 2D Map")
            axes[1].axis('off')

            plt.tight_layout()
            # Create a directory for reconstructions if it doesn't exist
            recon_dir = os.path.join('models', 'reconstructions')
            os.makedirs(recon_dir, exist_ok=True)
            # Save the figure with epoch and sample index
            plt.savefig(os.path.join(recon_dir, f'unet_results_2D_epoch_{epoch}_sample_{idx}.png'))
            plt.close(fig)  

# ---------------------------
# Training Loop with Early Stopping
# ---------------------------

def train(model, train_loader, val_loader, optimizer, device, epochs, patience, loss_fn, models_dir, val_dataset):
    """
    Trains the autoencoder with early stopping and saves reconstructions after each epoch.

    Args:
        model (nn.Module): Autoencoder model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run on.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        loss_fn (callable): Loss function.
        models_dir (str): Directory to save models.
        val_dataset (Dataset): Validation dataset for visualization.

    Returns:
        float: Best validation loss.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training phase
        train_loss = train_autoencoder(model, train_loader, optimizer, device, loss_fn)

        # Validation phase
        val_loss = validate_autoencoder(model, val_loader, device, loss_fn)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            best_model_path = os.path.join(models_dir, "unet_2d_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Save reconstructions after each epoch
        visualize_reconstruction(model, val_dataset, device, num_samples=5, epoch=epoch+1)

    print(f"Training completed. Best Validation Loss: {best_val_loss:.6f}")
    return best_val_loss

# ---------------------------
# Main Function
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure the 'models' directory exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Generate 2D maps
    print("Generating 2D maps...")
    maps = []
    for _ in tqdm(range(args.num_maps), desc="Generating maps"):
        map_ = generate_map(
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
        # Ensure the map has at least one occupied cell
        if np.any(map_):
            maps.append(map_)

    if len(maps) == 0:
        print("No valid maps generated. Exiting.")
        return

    # Split into training and validation sets
    split_idx = int(len(maps) * args.train_split)
    train_maps = maps[:split_idx]
    val_maps = maps[split_idx:]

    print(f"Total maps: {len(maps)}, Training: {len(train_maps)}, Validation: {len(val_maps)}")

    # Create datasets
    train_dataset = Grid2DDataset(train_maps, augment=args.augment, pad=args.pad, multiple=args.pad_multiple)
    val_dataset = Grid2DDataset(val_maps, augment=False, pad=args.pad, multiple=args.pad_multiple)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Initialize U-Net model and optimizer
    model = UNet2DAutoencoder(input_channels=1, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)  # Added weight_decay for regularization

    # Start training with early stopping
    best_val_loss = train(model, train_loader, val_loader, optimizer, device, args.epochs, args.patience, voxel_loss, models_dir, val_dataset)

    # Load the best model
    best_model_path = os.path.join(models_dir, "unet_2d_best.pth")
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best model loaded from {best_model_path}.")

    # Save the final model
    final_model_path = os.path.join(models_dir, "unet_2d_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}.")

    # Optional: Visualize some reconstructions
    if args.visualize:
        visualize_reconstruction(model, val_dataset, device, num_samples=args.num_samples, epoch='final')

# ---------------------------
# Argument Parser
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D U-Net Autoencoder for Occupancy Grid Reconstruction")
    parser.add_argument("--width", type=int, default=512, help="Width of the 2D map")  # Changed from 1600 to 512
    parser.add_argument("--height", type=int, default=512, help="Height of the 2D map")  # Changed from 1600 to 512
    parser.add_argument("--min_room_size", type=int, default=60, help="Minimum size of a room")  # Adjusted
    parser.add_argument("--max_room_size", type=int, default=120, help="Maximum size of a room")  # Adjusted
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum recursion depth for splitting rooms")  # Adjusted
    parser.add_argument("--wall_thickness", type=int, default=5, help="Thickness of the walls between rooms")
    parser.add_argument("--min_openings", type=int, default=1, help="Minimum number of openings per wall")
    parser.add_argument("--max_openings", type=int, default=2, help="Maximum number of openings per wall")
    parser.add_argument("--min_opening_size", type=int, default=10, help="Minimum size of each opening in pixels")  # Adjusted
    parser.add_argument("--max_opening_size", type=int, default=20, help="Maximum size of each opening in pixels")  # Adjusted
    parser.add_argument("--min_obstacles", type=int, default=4, help="Minimum number of obstacles per room")
    parser.add_argument("--max_obstacles", type=int, default=14, help="Maximum number of obstacles per room")
    parser.add_argument("--min_obstacle_size", type=int, default=10, help="Minimum size of each obstacle")  # Adjusted
    parser.add_argument("--max_obstacle_size", type=int, default=20, help="Maximum size of each obstacle")  # Adjusted
    parser.add_argument("--obstacle_attempts", type=int, default=10, help="Number of attempts to place an obstacle without overlap")
    parser.add_argument("--trap_probability", type=float, default=0.4, help="Probability of placing a concave trap instead of a regular obstacle")
    parser.add_argument("--num_maps", type=int, default=512, help="Number of maps to generate for training")  # Reduced from 20000 to 512
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")  # Increased from 1 to 4 for 512x512 maps
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Dimensionality of the latent space")  # Changed from 512 to 1024
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize reconstructions after training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize if --visualize is set")
    parser.add_argument("--augment", action='store_true', help="Whether to apply data augmentation to training data")
    parser.add_argument("--pad", action='store_true', help="Whether to pad the maps to the nearest multiple")
    parser.add_argument("--pad_multiple", type=int, default=16, help="The multiple to pad to (default: 16)")
    args = parser.parse_args()

    main(args)
