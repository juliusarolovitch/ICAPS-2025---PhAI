import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import argparse


def generate_map(width, height, num_obstacles, min_obstacle_size, max_obstacle_size):
    map_grid = np.zeros((height, width), dtype=np.float32)
    for _ in range(num_obstacles):
        obstacle_size = np.random.randint(
            min_obstacle_size, max_obstacle_size + 1)
        x = np.random.randint(0, width - obstacle_size + 1)
        y = np.random.randint(0, height - obstacle_size + 1)
        map_grid[y:y+obstacle_size, x:x+obstacle_size] = 1
    return map_grid


class MapDataset(Dataset):
    def __init__(self, maps):
        self.maps = maps

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        # Add channel dimension
        return torch.from_numpy(self.maps[idx]).unsqueeze(0)


class ConvAutoencoder(nn.Module):
    def __init__(self, height, width, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Calculate the size after convolutions dynamically
        def conv_output_size(size, kernel_size=3, stride=2, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        # Compute the output height and width after each convolutional layer
        conv1_h, conv1_w = conv_output_size(
            height), conv_output_size(width)  # After first conv
        conv2_h, conv2_w = conv_output_size(
            conv1_h), conv_output_size(conv1_w)  # After second conv
        conv3_h, conv3_w = conv_output_size(
            conv2_h), conv_output_size(conv2_w)  # After third conv
        conv4_h, conv4_w = conv_output_size(
            conv3_h), conv_output_size(conv3_w)  # After fourth conv

        # Encoder
        self.encoder = nn.Sequential(
            # Reduce to conv1_h x conv1_w
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Reduce to conv2_h x conv2_w
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Reduce to conv3_h x conv3_w
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Reduce to conv4_h x conv4_w
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(256 * conv4_h * conv4_w, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * conv4_h * conv4_w),
            nn.ReLU(),
            nn.Unflatten(1, (256, conv4_h, conv4_w)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Upsample
            # Final layer to match 100x100
            nn.Conv2d(1, 1, kernel_size=13, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = nn.BCELoss()(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = nn.BCELoss()(output, batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maps = [generate_map(args.width, args.height, args.num_obstacles, args.min_obstacle_size, args.max_obstacle_size)
            for _ in range(args.num_maps)]

    dataset = MapDataset(maps)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ConvAutoencoder(
        height=args.height, width=args.width, latent_dim=args.latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "autoencoder_2d_100.pth")
            print("New best model saved.")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print("Training completed. Best model saved as best_occupancy_map_autoencoder_100x100.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Convolutional Autoencoder for 100x100 2D Occupancy Map Encoding")
    parser.add_argument("--width", type=int, default=100,
                        help="Width of the map")
    parser.add_argument("--height", type=int, default=100,
                        help="Height of the map")
    parser.add_argument("--num_obstacles", type=int,
                        default=150, help="Number of obstacles per map")
    parser.add_argument("--min_obstacle_size", type=int,
                        default=2, help="Minimum size of each obstacle")
    parser.add_argument("--max_obstacle_size", type=int,
                        default=4, help="Maximum size of each obstacle")
    parser.add_argument("--num_maps", type=int, default=100000,
                        help="Number of maps to generate for training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Dimension of the latent space")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs to wait for improvement before early stopping")
    args = parser.parse_args()

    main(args)
