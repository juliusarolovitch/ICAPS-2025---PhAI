import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import os  # For directory operations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import random

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

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
        latent_vector, _ = self.encode(x)
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

def visualize_reconstruction(model, dataset, device, num_samples=5):
    """
    Visualizes original and reconstructed 2D maps.
    
    Args:
        model (nn.Module): Trained autoencoder model.
        dataset (Dataset): Validation dataset.
        device (torch.device): Device to run on.
        num_samples (int): Number of samples to visualize.
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
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(original, cmap='Greys', origin='upper')
            axes[0].set_title("Original 2D Map")
            axes[0].axis('off')

            axes[1].imshow(reconstructed, cmap='Greys', origin='upper')
            axes[1].set_title("Reconstructed 2D Map")
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(f'models/unet_results_2D_{idx}.png')  
            plt.close(fig)  

# ---------------------------
# Training Loop with Early Stopping
# ---------------------------

def train(model, train_loader, val_loader, optimizer, device, epochs, patience, loss_fn, models_dir):
    """
    Trains the autoencoder with early stopping.
    
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
            best_model_path = os.path.join(models_dir, "unet_3d_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

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
        map_ = generate_map()
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
    train_dataset = Grid2DDataset(train_maps, augment=False, pad=args.pad, multiple=args.pad_multiple)
    val_dataset = Grid2DDataset(val_maps, augment=False, pad=args.pad, multiple=args.pad_multiple)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Initialize U-Net model and optimizer
    model = UNet2DAutoencoder(input_channels=1, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)  # Added weight_decay for regularization

    # Start training with early stopping
    best_val_loss = train(model, train_loader, val_loader, optimizer, device, args.epochs, args.patience, voxel_loss, models_dir)

    # Load the best model
    best_model_path = os.path.join(models_dir, "unet_3d_best.pth")
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best model loaded from {best_model_path}.")

    # Save the final model
    final_model_path = os.path.join(models_dir, "unet_3d_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}.")

    # Optional: Visualize some reconstructions
    if args.visualize:
        visualize_reconstruction(model, val_dataset, device, num_samples=args.num_samples)

# ---------------------------
# Argument Parser
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D U-Net Autoencoder for Occupancy Grid Reconstruction")
    parser.add_argument("--width", type=int, default=128, help="Width of the 2D map")
    parser.add_argument("--height", type=int, default=128, help="Height of the 2D map")
    parser.add_argument("--num_obstacles", type=int, default=150, help="Number of obstacles per map")
    parser.add_argument("--min_obstacle_size", type=int, default=4, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int, default=8, help="Maximum size of each obstacle")
    parser.add_argument("--num_maps", type=int, default=200, help="Number of maps to generate for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")  # Increased for 2D
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent space")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize reconstructions after training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize if --visualize is set")
    parser.add_argument("--pad", action='store_true', help="Whether to pad the maps to the nearest multiple")
    parser.add_argument("--pad_multiple", type=int, default=16, help="The multiple to pad to (default: 16)")
    args = parser.parse_args()

    main(args)
