import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass, field
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Room DataClass
# ---------------------------

@dataclass
class Room:
    x: int
    y: int
    width: int
    height: int
    children: list = field(default_factory=list)

# ---------------------------
# Utility Functions
# ---------------------------

def generate_3d_maze(
    width=100,
    height=100,
    depth=40,
    min_room_size=20,
    max_room_size=30,
    max_depth=5,
    wall_thickness=2,
    min_openings=1,
    max_openings=2,
    min_opening_size=5,
    max_opening_size=10,
    min_walls=2,  # Updated to ensure 2 to 5 walls
    max_walls=5,  # Updated to ensure 2 to 5 walls
    min_wall_length=20,  # Updated for horizontal walls between 20-60
    max_wall_length=60,  # Updated for horizontal walls between 20-60
    wall_attempts=10
):
    """
    Generates a 3D maze by recursively splitting the space into rooms and adding internal horizontal walls.

    Args:
        width (int): Width of the maze (x-axis).
        height (int): Height of the maze (y-axis).
        depth (int): Depth of the maze (z-axis).
        min_room_size (int): Minimum size of a room in x and y.
        max_room_size (int): Maximum size of a room in x and y.
        max_depth (int): Maximum recursion depth for splitting rooms.
        wall_thickness (int): Thickness of the walls between rooms.
        min_openings (int): Minimum number of openings per wall.
        max_openings (int): Maximum number of openings per wall.
        min_opening_size (int): Minimum size of each opening in pixels.
        max_opening_size (int): Maximum size of each opening in pixels.
        min_walls (int): Minimum number of internal horizontal walls per room.
        max_walls (int): Maximum number of internal horizontal walls per room.
        min_wall_length (int): Minimum length of internal walls.
        max_wall_length (int): Maximum length of internal walls.
        wall_attempts (int): Number of attempts to place an internal wall without overlap.

    Returns:
        np.ndarray: 3D occupancy map of shape [height, width, depth].
    """
    # Initialize the 3D map with zeros (free space)
    maze = np.zeros((height, width, depth), dtype=np.uint8)

    # Initialize the root room
    root_room = Room(0, 0, width, height)

    def split_room(room, depth_level):
        if depth_level >= max_depth:
            return
        # Determine if the room can be split horizontally or vertically
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
            # Add horizontal wall (extend vertically across all z)
            maze[split_pos:split_pos + wall_thickness, room.x:room.x + room.width, :] = 1
            # Add openings to the wall
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
            # Add vertical wall (extend vertically across all z)
            maze[room.y:room.y + room.height, split_pos:split_pos + wall_thickness, :] = 1
            # Add openings to the wall
            add_openings((room.y, split_pos), (room.y + room.height, split_pos), orientation='vertical')

        room.children = [child1, child2]
        # Recursively split the child rooms
        split_room(child1, depth_level + 1)
        split_room(child2, depth_level + 1)

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
                maze[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size, :] = 0
        else:
            wall_length = end[0] - start[0]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)  # Ensure opening size doesn't exceed wall
                opening_start = random.randint(start[0], end[0] - opening_size)
                maze[opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness, :] = 0

    def add_internal_walls(room):
        """
        Adds 2 to 5 internal horizontal walls within a room with random lengths between 20 and 60.

        Args:
            room (Room): The room where internal walls will be added.
        """
        num_internal_walls = random.randint(2, 5)  # Ensure 2 to 5 walls
        for _ in range(num_internal_walls):
            for attempt in range(wall_attempts):
                orientation = 'horizontal'  # Only horizontal walls as per requirement
                wall_length = random.randint(min_wall_length, max_wall_length)
                if wall_length >= room.width - 2 * wall_thickness:
                    continue  # Wall too long
                y = random.randint(room.y + wall_thickness, room.y + room.height - wall_thickness)
                x_start = random.randint(room.x + wall_thickness, room.x + room.width - wall_length - wall_thickness)
                x_end = x_start + wall_length
                # Check if the wall area is free
                if np.any(maze[y: y + wall_thickness, x_start:x_end, :] == 1):
                    continue  # Overlaps with existing wall
                # Add the wall
                maze[y: y + wall_thickness, x_start:x_end, :] = 1
                break  # Successfully added

    def collect_leaf_rooms(room, leaf_rooms):
        if not room.children:
            leaf_rooms.append(room)
        else:
            for child in room.children:
                collect_leaf_rooms(child, leaf_rooms)

    # Start splitting from the root room
    split_room(root_room, 0)

    # Collect all leaf rooms
    leaf_rooms = []
    collect_leaf_rooms(root_room, leaf_rooms)

    # Add internal horizontal walls within each leaf room
    for room in leaf_rooms:
        add_internal_walls(room)

    # Add outer boundary walls (extend vertically)
    maze[0:wall_thickness, :, :] = 1  # Top
    maze[-wall_thickness:, :, :] = 1  # Bottom
    maze[:, 0:wall_thickness, :] = 1  # Left
    maze[:, -wall_thickness:, :] = 1  # Right

    return maze

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Import your maze generation code here
# from maze_generator import generate_3d_maze

# ---------------------------
# Point Cloud Conversion
# ---------------------------

def occupancy_grid_to_point_cloud(maze, num_points=2048):
    """
    Converts an occupancy grid into a point cloud by extracting the coordinates of occupied cells.

    Args:
        maze (np.ndarray): 3D occupancy grid of shape [height, width, depth].
        num_points (int): Number of points to sample from the point cloud.

    Returns:
        np.ndarray: Point cloud of shape [num_points, 3].
    """
    occupied_indices = np.argwhere(maze == 1)
    if len(occupied_indices) >= num_points:
        sampled_indices = occupied_indices[np.random.choice(len(occupied_indices), num_points, replace=False)]
    else:
        # If not enough points, duplicate some
        pad_size = num_points - len(occupied_indices)
        pad_indices = occupied_indices[np.random.choice(len(occupied_indices), pad_size, replace=True)]
        sampled_indices = np.concatenate([occupied_indices, pad_indices], axis=0)
    return sampled_indices.astype(np.float32)

# ---------------------------
# Dataset Definition
# ---------------------------

class MazePointCloudDataset(Dataset):
    def __init__(self, num_samples, num_points=2048):
        self.num_samples = num_samples
        self.num_points = num_points
        self.point_clouds = []
        for _ in range(num_samples):
            maze = generate_3d_maze(
                width=100,
                height=100,
                depth=40,
                min_room_size=20,
                max_room_size=30,
                max_depth=5,
                wall_thickness=2,
                min_openings=1,
                max_openings=2,
                min_opening_size=5,
                max_opening_size=10,
                min_walls=2,
                max_walls=5,
                min_wall_length=20,
                max_wall_length=60,
                wall_attempts=10
            )
            point_cloud = occupancy_grid_to_point_cloud(maze, num_points=num_points)
            # Normalize to [-1, 1]
            point_cloud = point_cloud / np.array([[maze.shape[0], maze.shape[1], maze.shape[2]]]) * 2 - 1
            self.point_clouds.append(point_cloud)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        return point_cloud

# ---------------------------
# Chamfer Distance Function
# ---------------------------

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer Distance between two point clouds.

    Args:
        pc1 (torch.Tensor): Point cloud tensor of shape [B, N, 3].
        pc2 (torch.Tensor): Point cloud tensor of shape [B, M, 3].

    Returns:
        torch.Tensor: Chamfer distance.
    """
    x, y = pc1, pc2
    B, N, _ = x.size()
    _, M, _ = y.size()

    x = x.unsqueeze(2)  # [B, N, 1, 3]
    y = y.unsqueeze(1)  # [B, 1, M, 3]
    dist = torch.sum((x - y) ** 2, dim=3)  # [B, N, M]

    dist_x, _ = torch.min(dist, dim=2)  # [B, N]
    dist_y, _ = torch.min(dist, dim=1)  # [B, M]
    loss = torch.mean(dist_x) + torch.mean(dist_y)
    return loss

# ---------------------------
# PointNet Autoencoder
# ---------------------------

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]
        x = torch.max(x, 2)[0]               # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))    # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))    # [B, 256]
        x = self.fc3(x)                      # [B, 9]

        iden = torch.eye(3, requires_grad=True).view(1, 9).repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]
        x = torch.max(x, 2)[0]               # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))    # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))    # [B, 256]
        x = self.fc3(x)                      # [B, k*k]

        iden = torch.eye(self.k, requires_grad=True).view(1, self.k * self.k).repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d()
        self.feature_transform = feature_transform
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x: [B, 3, N]
        batchsize = x.size(0)
        n_pts = x.size(2)

        trans = self.stn(x)  # [B, 3, 3]
        x = x.transpose(2, 1)  # [B, N, 3]
        x = torch.bmm(x, trans)  # [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]

        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

        if self.feature_transform:
            trans_feat = self.fstn(x)  # [B, 64, 64]
            x = x.transpose(2, 1)  # [B, N, 64]
            x = torch.bmm(x, trans_feat)  # [B, N, 64]
            x = x.transpose(2, 1)  # [B, 64, N]
        else:
            trans_feat = None

        pointfeat = x  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = self.bn3(self.conv3(x))          # [B, 1024, N]
        x = torch.max(x, 2)[0]               # [B, 1024]
        return x, trans, trans_feat

class PointNetDecoder(nn.Module):
    def __init__(self, num_points=2048):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points  # Save num_points
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.fc3 = nn.Linear(2048, num_points * 3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))    # [B, 1024]
        x = F.relu(self.bn2(self.fc2(x)))    # [B, 2048]
        x = self.fc3(x)                      # [B, num_points * 3]
        x = x.view(-1, 3, self.num_points)   # Specify num_points explicitly
        return x


class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points=2048, feature_transform=True):
        super(PointNetAutoencoder, self).__init__()
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.decoder = PointNetDecoder(num_points)

    def forward(self, x):
        # x: [B, 3, N]
        x, trans, trans_feat = self.encoder(x)
        recon = self.decoder(x)
        return recon, x, trans_feat

def feature_transform_regularizer(trans):
    """
    Regularization term for the feature transformation matrix.

    Args:
        trans (torch.Tensor): Transformation matrix of shape [B, K, K].

    Returns:
        torch.Tensor: Regularization loss.
    """
    batchsize = trans.size(0)
    K = trans.size(1)
    I = torch.eye(K, requires_grad=True).to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

# ---------------------------
# Training Loop
# ---------------------------

def train():
    # Hyperparameters
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.001
    num_points = 2048
    num_samples = 10000  # Adjust as needed
    feature_transform = True

    # Dataset and DataLoader
    dataset = MazePointCloudDataset(num_samples=num_samples, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetAutoencoder(num_points=num_points, feature_transform=feature_transform).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:
    # Convert data to torch.float32
            data = data.float()  # Add this line
            data = data.permute(0, 2, 1).to(device)  # [B, 3, N]
            optimizer.zero_grad()
            recon, latent, trans_feat = model(data)
            loss = chamfer_distance(data.permute(0, 2, 1), recon.permute(0, 2, 1))
            if feature_transform:
                reg_loss = feature_transform_regularizer(trans_feat) * 0.001
                loss += reg_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), "pointnet_autoencoder.pth")
    print("Model saved as pointnet_autoencoder.pth")

if __name__ == "__main__":
    train()
