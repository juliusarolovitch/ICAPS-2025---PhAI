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
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Utility Functions
# ---------------------------

def generate_map(width, height, depth, num_obstacles, min_obstacle_size, max_obstacle_size):
    """
    Generates a 3D occupancy map with random cubic obstacles.
    """
    map_grid = np.zeros((depth, height, width), dtype=np.float32)
    for _ in range(num_obstacles):
        obstacle_size = np.random.randint(min_obstacle_size, max_obstacle_size + 1)
        x = np.random.randint(0, width - obstacle_size + 1)
        y = np.random.randint(0, height - obstacle_size + 1)
        z = np.random.randint(0, depth - obstacle_size + 1)
        map_grid[z:z+obstacle_size, y:y+obstacle_size, x:x+obstacle_size] = 1
    return map_grid

# ---------------------------
# Dataset Class
# ---------------------------

class VoxelGridDataset(Dataset):
    def __init__(self, maps):
        """
        Initializes the dataset with occupancy grids.
        
        Args:
            maps (list of np.ndarray): List of 3D occupancy maps.
        """
        self.maps = maps
    
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Occupancy grid of shape [1, D, H, W].
        """
        voxel = self.maps[idx]
        # Normalize if needed (already binary)
        voxel = torch.from_numpy(voxel).unsqueeze(0).float()  # [1, D, H, W]
        return voxel

# ---------------------------
# 3D Convolutional Autoencoder
# ---------------------------

class Conv3DAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, input_size=64):
        super(Conv3DAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
            # Input: [B, 1, D, H, W]
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, D/2, H/2, W/2]
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, D/4, H/4, W/4]
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, D/8, H/8, W/8]
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, D/16, H/16, W/16]
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            # Additional layers
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),  # [B, 512, D/32, H/32, W/32]
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1),  # [B, 1024, D/64, H/64, W/64]
            nn.BatchNorm3d(1024),
            nn.ReLU(True),
        )

        # Adjust the bottleneck accordingly
        bottleneck_size = 1024 * (input_size // 64) ** 3  # Update based on new encoder depth
        self.encoder_fc = nn.Sequential(
            nn.Linear(bottleneck_size, latent_dim),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_size),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, D/32, H/32, W/32]
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),   # [B, 256, D/16, H/16, W/16]
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),   # [B, 128, D/8, H/8, W/8]
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),    # [B, 64, D/4, H/4, W/4]
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),     # [B, 32, D/2, H/2, W/2]
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),      # [B, 1, D, H, W]
            nn.Sigmoid()  # For binary occupancy
        )
        
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        latent = self.encoder_fc(x)  # [B, latent_dim]
        
        # Decode
        x = self.decoder_fc(latent)
        x = x.view(-1, 256, self.input_size // 16, self.input_size // 16, self.input_size // 16)  # Adjust based on encoder's output
        x = self.decoder(x)  # [B, 1, D, H, W]
        return x, latent

class Conv3DAutoencoderWithResiduals(nn.Module):
    def __init__(self, latent_dim=512, input_size=64):
        super(Conv3DAutoencoderWithResiduals, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            ResidualBlock3D(32, 64, stride=2),  # Downsample
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 256, stride=2),
            ResidualBlock3D(256, 512, stride=2),
            ResidualBlock3D(512, 1024, stride=2),
            ResidualBlock3D(1024, 1024, stride=2),
        )

        # Calculate bottleneck size
        bottleneck_size = 1024 * (input_size // 64) ** 3
        self.encoder_fc = nn.Sequential(
            nn.Linear(bottleneck_size, latent_dim),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_size),
            nn.ReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            ResidualBlock3D(1024, 1024),
            nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            ResidualBlock3D(512, 512),
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            ResidualBlock3D(256, 256),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            ResidualBlock3D(128, 128),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            ResidualBlock3D(64, 64),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            ResidualBlock3D(32, 32),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.encoder_fc(x)

        # Decode
        x = self.decoder_fc(latent)
        x = x.view(-1, 1024, self.input_size // 64, self.input_size // 64, self.input_size // 64)
        x = self.decoder_conv(x)
        return x, latent

class UNet3DAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, input_size=64):
        super(UNet3DAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)
        self.enc6 = self.conv_block(512, 1024)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(True)
        )

        # Decoder
        self.dec6 = self.deconv_block(1024, 1024)
        self.dec5 = self.deconv_block(2048, 512)  # 1024 from skip connection + 1024 from previous layer
        self.dec4 = self.deconv_block(1024, 256)
        self.dec3 = self.deconv_block(512, 128)
        self.dec2 = self.deconv_block(256, 64)
        self.dec1 = self.deconv_block(128, 32)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)  # [B, 32, D, H, W]
        e2 = self.enc2(F.max_pool3d(e1, 2))  # [B, 64, D/2, H/2, W/2]
        e3 = self.enc3(F.max_pool3d(e2, 2))  # [B, 128, D/4, H/4, W/4]
        e4 = self.enc4(F.max_pool3d(e3, 2))  # [B, 256, D/8, H/8, W/8]
        e5 = self.enc5(F.max_pool3d(e4, 2))  # [B, 512, D/16, H/16, W/16]
        e6 = self.enc6(F.max_pool3d(e5, 2))  # [B, 1024, D/32, H/32, W/32]

        # Bottleneck
        b = self.bottleneck(e6)

        # Decoder with concatenation of skip connections
        d6 = self.dec6(b)  # [B, 1024, D/16, H/16, W/16]
        d5 = self.dec5(torch.cat([d6, e5], dim=1))  # [B, 512, D/8, H/8, W/8]
        d4 = self.dec4(torch.cat([d5, e4], dim=1))  # [B, 256, D/4, H/4, W/4]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # [B, 128, D/2, H/2, W/2]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B, 64, D, H, W]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [B, 32, 2D, 2H, 2W]

        # Final output
        out = self.final(d1)  # [B, 1, D*2, H*2, W*2]
        return out


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


# ---------------------------
# Loss Function
# ---------------------------

# Using Binary Cross-Entropy Loss for voxel occupancy
def voxel_loss(recon, target):
    return F.binary_cross_entropy(recon, target)

# ---------------------------
# Training and Validation Functions
# ---------------------------

def train_autoencoder(model, train_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)  # [B, 1, D, H, W]
        optimizer.zero_grad()

        # Forward pass
        recon_batch, _ = model(batch)  # [B, 1, D, H, W], [B, latent_dim]

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
            batch = batch.to(device)  # [B, 1, D, H, W]

            # Forward pass
            recon_batch, _ = model(batch)  # [B, 1, D, H, W], [B, latent_dim]

            # Compute loss
            loss = loss_fn(recon_batch, batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def visualize_reconstruction(model, dataset, device, num_samples=5):
    """
    Visualizes original and reconstructed voxel grids.
    
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
            original_voxel = dataset[idx].unsqueeze(0).to(device)  # [1, 1, D, H, W]
            reconstructed_voxel, _ = model(original_voxel)  # [1, 1, D, H, W]
            original_voxel = original_voxel.cpu().numpy()[0, 0]
            reconstructed_voxel = reconstructed_voxel.cpu().numpy()[0, 0]

            # Plotting a central slice for visualization
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            mid_slice = original_voxel.shape[0] // 2

            axes[0].imshow(original_voxel[mid_slice], cmap='gray')
            axes[0].set_title("Original Voxel Grid Slice")
            axes[1].imshow(reconstructed_voxel[mid_slice], cmap='gray')
            axes[1].set_title("Reconstructed Voxel Grid Slice")

            plt.savefig(f'results_{idx}.png')

# ---------------------------
# Training Loop with Early Stopping
# ---------------------------

def train(model, train_loader, val_loader, optimizer, device, num_epochs, patience, loss_fn):
    """
    Trains the autoencoder with early stopping.
    
    Args:
        model (nn.Module): Autoencoder model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run on.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        loss_fn (callable): Loss function.
    
    Returns:
        float: Best validation loss.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training phase
        train_loss = train_autoencoder(model, train_loader, optimizer, device, loss_fn)

        # Validation phase
        val_loss = validate_autoencoder(model, val_loader, device, loss_fn)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "conv3d_autoencoder_best.pth")
            print("Best model saved.")
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

    # Generate 3D maps
    print("Generating maps...")
    maps = []
    for _ in tqdm(range(args.num_maps), desc="Generating maps"):
        map_ = generate_map(
            width=args.width,
            height=args.height,
            depth=args.depth,
            num_obstacles=args.num_obstacles,
            min_obstacle_size=args.min_obstacle_size,
            max_obstacle_size=args.max_obstacle_size
        )
        # Ensure the map has at least one occupied voxel
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
    train_dataset = VoxelGridDataset(train_maps)
    val_dataset = VoxelGridDataset(val_maps)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Initialize model, optimizer
    model = Conv3DAutoencoder(latent_dim=args.latent_dim, input_size=args.width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training with early stopping
    best_val_loss = train(model, train_loader, val_loader, optimizer, device, args.num_epochs, args.patience, voxel_loss)

    # Load the best model
    model.load_state_dict(torch.load("conv3d_autoencoder_best.pth"))
    print("Best model loaded.")

    # Save the final model
    torch.save(model.state_dict(), "conv3d_autoencoder_final.pth")
    print("Final model saved as conv3d_autoencoder_final.pth")

    # Optional: Visualize some reconstructions
    if args.visualize:
        visualize_reconstruction(model, val_dataset, device, num_samples=args.num_samples)

# ---------------------------
# Argument Parser
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Convolutional Autoencoder for Occupancy Grid Reconstruction")
    parser.add_argument("--width", type=int, default=64, help="Width of the 3D map")
    parser.add_argument("--height", type=int, default=64, help="Height of the 3D map")
    parser.add_argument("--depth", type=int, default=64, help="Depth of the 3D map")
    parser.add_argument("--num_obstacles", type=int, default=150, help="Number of obstacles per map")
    parser.add_argument("--min_obstacle_size", type=int, default=4, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int, default=8, help="Maximum size of each obstacle")
    parser.add_argument("--num_maps", type=int, default=200, help="Number of maps to generate for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")  # Adjust based on GPU memory
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent space")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize reconstructions after training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize if --visualize is set")
    args = parser.parse_args()

    main(args)
