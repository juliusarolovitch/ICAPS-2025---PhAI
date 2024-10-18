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

def occupancy_to_pointcloud(occupancy_grid):
    indices = np.argwhere(occupancy_grid > 0)  # Indices where occupancy is 1
    # Normalize coordinates to [-0.5, 0.5]
    depth, height, width = occupancy_grid.shape
    coords = indices.astype(np.float32)
    coords[:, 0] = coords[:, 0] / (depth - 1) - 0.5  # z
    coords[:, 1] = coords[:, 1] / (height - 1) - 0.5  # y
    coords[:, 2] = coords[:, 2] / (width - 1) - 0.5   # x
    return coords  # [N, 3]

# ---------------------------
# Dataset Class
# ---------------------------

class PointCloudDataset(Dataset):
    def __init__(self, pointclouds, num_points=2048):
        """
        Initializes the dataset with point clouds.

        Args:
            pointclouds (list of np.ndarray): List of point clouds.
            num_points (int): Number of points to sample from each point cloud.
        """
        self.pointclouds = pointclouds
        self.num_points = num_points

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Point cloud of shape [N, 3].
        """
        pointcloud = self.pointclouds[idx]
        if pointcloud.shape[0] >= self.num_points:
            indices = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
        else:
            # If the point cloud has fewer points, duplicate some points
            indices = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
        pointcloud = pointcloud[indices]
        pointcloud = torch.from_numpy(pointcloud).float()  # [N, 3]
        return pointcloud

# ---------------------------
# PointNet Autoencoder
# ---------------------------

class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points=2048, latent_dim=512):
        super(PointNetAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Conv1d(256, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_points * 3)
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]
        x = self.encoder(x)    # [B, latent_dim, N]
        x = torch.max(x, 2)[0]  # [B, latent_dim]
        latent = x

        # Decoder
        x = self.decoder(x)   # [B, N*3]
        x = x.view(-1, self.num_points, 3)  # [B, N, 3]
        return x, latent

# ---------------------------
# Loss Function
# ---------------------------

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer Distance between two point clouds pc1 and pc2.

    Args:
        pc1: [B, N, 3]
        pc2: [B, N, 3]

    Returns:
        float: Chamfer Distance
    """
    x, y = pc1, pc2
    B, N, _ = x.shape
    xx = x.unsqueeze(2)  # [B, N, 1, 3]
    yy = y.unsqueeze(1)  # [B, 1, N, 3]
    dist = torch.norm(xx - yy, dim=3)  # [B, N, N]
    min_dist_x, _ = torch.min(dist, dim=2)  # [B, N]
    min_dist_y, _ = torch.min(dist, dim=1)  # [B, N]
    loss = torch.mean(min_dist_x) + torch.mean(min_dist_y)
    return loss

# ---------------------------
# Training and Validation Functions
# ---------------------------

def train_autoencoder(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)  # [B, N, 3]
        optimizer.zero_grad()

        # Forward pass
        recon_batch, _ = model(batch)  # [B, N, 3], [B, latent_dim]

        # Compute loss
        loss = chamfer_distance(recon_batch, batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_autoencoder(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            batch = batch.to(device)  # [B, N, 3]

            # Forward pass
            recon_batch, _ = model(batch)  # [B, N, 3], [B, latent_dim]

            # Compute loss
            loss = chamfer_distance(recon_batch, batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def visualize_reconstruction(model, dataset, device, num_samples=5):
    """
    Visualizes original and reconstructed point clouds.

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
            original_pc = dataset[idx].unsqueeze(0).to(device)  # [1, N, 3]
            reconstructed_pc, _ = model(original_pc)  # [1, N, 3]

            original_pc = original_pc.cpu().numpy()[0]  # [N, 3]
            reconstructed_pc = reconstructed_pc.cpu().numpy()[0]  # [N, 3]

            # Plotting the point clouds
            fig = plt.figure(figsize=(10, 5))

            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], s=1)
            ax.set_title("Original Point Cloud")

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(reconstructed_pc[:, 0], reconstructed_pc[:, 1], reconstructed_pc[:, 2], s=1)
            ax.set_title("Reconstructed Point Cloud")

            plt.savefig(f'results_{idx}.png')

# ---------------------------
# Training Loop with Early Stopping
# ---------------------------

def train(model, train_loader, val_loader, optimizer, device, num_epochs, patience):
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

    Returns:
        float: Best validation loss.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training phase
        train_loss = train_autoencoder(model, train_loader, optimizer, device)

        # Validation phase
        val_loss = validate_autoencoder(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "pointnet_autoencoder_best.pth")
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

    # Generate 3D maps and corresponding point clouds
    print("Generating maps and point clouds...")
    maps = []
    pointclouds = []
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
            pointcloud = occupancy_to_pointcloud(map_)
            pointclouds.append(pointcloud)

    if len(pointclouds) == 0:
        print("No valid maps generated. Exiting.")
        return

    # Split into training and validation sets
    split_idx = int(len(pointclouds) * args.train_split)
    train_pointclouds = pointclouds[:split_idx]
    val_pointclouds = pointclouds[split_idx:]

    print(f"Total maps: {len(pointclouds)}, Training: {len(train_pointclouds)}, Validation: {len(val_pointclouds)}")

    # Create datasets
    train_dataset = PointCloudDataset(train_pointclouds, num_points=args.num_points)
    val_dataset = PointCloudDataset(val_pointclouds, num_points=args.num_points)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Initialize model, optimizer
    model = PointNetAutoencoder(num_points=args.num_points, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training with early stopping
    best_val_loss = train(model, train_loader, val_loader, optimizer, device, args.num_epochs, args.patience)

    # Load the best model
    model.load_state_dict(torch.load("pointnet_best.pth"))
    print("Best model loaded.")

    # Save the final model
    torch.save(model.state_dict(), "pointnet_final.pth")
    print("Final model saved as pointnet_final.pth")

    # Optional: Visualize some reconstructions
    if args.visualize:
        visualize_reconstruction(model, val_dataset, device, num_samples=args.num_samples)

# ---------------------------
# Argument Parser
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PointNet Autoencoder for Point Cloud Reconstruction")
    parser.add_argument("--width", type=int, default=64, help="Width of the 3D map")
    parser.add_argument("--height", type=int, default=64, help="Height of the 3D map")
    parser.add_argument("--depth", type=int, default=64, help="Depth of the 3D map")
    parser.add_argument("--num_obstacles", type=int, default=150, help="Number of obstacles per map")
    parser.add_argument("--min_obstacle_size", type=int, default=4, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int, default=8, help="Maximum size of each obstacle")
    parser.add_argument("--num_maps", type=int, default=200, help="Number of maps to generate for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent space")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize reconstructions after training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize if --visualize is set")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample from each point cloud")
    args = parser.parse_args()

    main(args)
