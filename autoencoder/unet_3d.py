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
from dataclasses import dataclass, field

# Ensure CUDA is available
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

# ---------------------------
# Room Data Class
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
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                if opening_size > wall_length:
                    opening_size = wall_length
                opening_start = random.randint(start[1], start[1] + wall_length - opening_size)
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        else:
            wall_length = end[0] - start[0]
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
    def __init__(self, input_channels=1, latent_dim=512):
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
        b = self.bottleneck(F.max_pool2d(e4, 2))  # [B, 256, 100, 100]

        return b  # Returning the feature map instead of flattening

    def decode(self, b):
        # Decoder
        d4 = self.up4(b)  # [B, 256, 200, 200]
        d4 = self.dec4(d4)  # [B, 256, 200, 200]

        d3 = self.up3(d4)  # [B, 128, 400, 400]
        d3 = self.dec3(d3)  # [B, 128, 400, 400]

        d2 = self.up2(d3)  # [B, 64, 800, 800]
        d2 = self.dec2(d2)  # [B, 64, 800, 800]

        d1 = self.up1(d2)  # [B, 32, 1600, 1600]
        d1 = self.dec1(d1)  # [B, 32, 1600, 1600]

        # Final output
        out = self.final(d1)  # [B, 1, 1600, 1600]
        return out  # Removed torch.sigmoid

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def get_latent_vector(self, x):
        """
        Extracts the latent feature map from the input.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W].

        Returns:
            torch.Tensor: Latent feature map of shape [B, 256, H/16, W/16].
        """
        latent = self.encode(x)
        return latent

# ---------------------------
# Loss Function
# ---------------------------

# Using Binary Cross-Entropy with Logits Loss for occupancy
def voxel_loss(recon, target):
    return F.binary_cross_entropy_with_logits(recon, target)

# ---------------------------
# Training and Validation Functions
# ---------------------------

def train_autoencoder(model, train_loader, optimizer, device, loss_fn, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)  # [B, 1, H, W]
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            # Forward pass
            recon_batch = model(batch)  # [B, 1, H, W]

            # Compute loss
            loss = loss_fn(recon_batch, batch)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            axes[0].imshow(original, cmap='Greys', origin='upper')
            axes[0].set_title("Original 2D Map")
            axes[0].axis('off')

            axes[1].imshow(reconstructed > 0.5, cmap='Greys', origin='upper')  # Thresholded for binary visualization
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

    # Learning rate scheduler without verbose to fix deprecation warning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training phase
        train_loss = train_autoencoder(model, train_loader, optimizer, device, loss_fn, scaler)

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

    print(f"Training completed. Best Validation Loss: {best_val_loss:.6f}")
    return best_val_loss

# ---------------------------
# Main Function
# ---------------------------

def main(args):
    # Clear CUDA cache
    torch.cuda.empty_cache()

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
            min_circle_radius=args.min_circle_radius,
            max_circle_radius=args.max_circle_radius,
            obstacle_attempts=args.obstacle_attempts
        )
        # Ensure the map has at least one occupied cell
        if np.any(map_):
            maps.append(map_)
        else:
            print("Generated an empty map. Skipping.")

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # Initialize U-Net model and optimizer
    model = UNet2DAutoencoder(input_channels=1, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)  # Added weight_decay for regularization

    # Start training with early stopping
    best_val_loss = train(model, train_loader, val_loader, optimizer, device, args.epochs, args.patience, voxel_loss, models_dir)

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
        visualize_reconstruction(model, val_dataset, device, num_samples=args.num_samples)

# ---------------------------
# Argument Parser
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D U-Net Autoencoder for Occupancy Grid Reconstruction")
    # Map generation parameters
    parser.add_argument("--width", type=int, default=1600, help="Width of the 2D map")
    parser.add_argument("--height", type=int, default=1600, help="Height of the 2D map")
    parser.add_argument("--min_room_size", type=int, default=300, help="Minimum size of a room")
    parser.add_argument("--max_room_size", type=int, default=600, help="Maximum size of a room")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum recursion depth for splitting rooms")
    parser.add_argument("--wall_thickness", type=int, default=5, help="Thickness of the walls between rooms")
    parser.add_argument("--min_openings", type=int, default=1, help="Minimum number of openings per wall")
    parser.add_argument("--max_openings", type=int, default=2, help="Maximum number of openings per wall")
    parser.add_argument("--min_opening_size", type=int, default=45, help="Minimum size of each opening in units")
    parser.add_argument("--max_opening_size", type=int, default=80, help="Maximum size of each opening in units")
    parser.add_argument("--min_obstacles", type=int, default=1, help="Minimum number of obstacles per room")
    parser.add_argument("--max_obstacles", type=int, default=2, help="Maximum number of obstacles per room")
    parser.add_argument("--min_obstacle_size", type=int, default=100, help="Minimum size of each rectangular obstacle")
    parser.add_argument("--max_obstacle_size", type=int, default=200, help="Maximum size of each rectangular obstacle")
    parser.add_argument("--min_circle_radius", type=int, default=50, help="Minimum radius of circular obstacles")
    parser.add_argument("--max_circle_radius", type=int, default=80, help="Maximum radius of circular obstacles")
    parser.add_argument("--obstacle_attempts", type=int, default=10, help="Number of attempts to place an obstacle without overlap")
    
    # Training parameters
    parser.add_argument("--num_maps", type=int, default=200, help="Number of maps to generate for training")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")  # Reduced from 4 to 2
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent space")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize reconstructions after training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize if --visualize is set")
    parser.add_argument("--augment", action='store_true', help="Whether to apply data augmentation")
    parser.add_argument("--pad", action='store_true', help="Whether to pad the maps to the nearest multiple")
    parser.add_argument("--pad_multiple", type=int, default=16, help="The multiple to pad to (default: 16)")
    args = parser.parse_args()

    main(args)
