# train_model_final.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os
import pickle
import optuna
import random
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt
import heapq
import csv
from matplotlib.colors import ListedColormap
from dataclasses import dataclass, field

from models import MLPModel, UNet2DAutoencoder  # Ensure models are correctly imported

# ---------------------------
# Dataset Classes
# ---------------------------

class FStarDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print(f"Dataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Unpack all seven elements from the dataset
            encoded_map, start, goal, current, g_normalized, h_normalized, target_value = self.data[idx]

            # Normalize positional coordinates by dividing by 127.0 (assuming map size is 128x128)
            start_normalized = np.array(start) / 127.0
            goal_normalized = np.array(goal) / 127.0
            current_normalized = np.array(current) / 127.0

            # Concatenate all components to form the input tensor
            input_tensor = np.concatenate([
                start_normalized,              # 2 elements
                goal_normalized,               # 2 elements
                current_normalized,            # 2 elements
                [g_normalized, h_normalized],  # 2 elements
                encoded_map                    # latent_dim elements
            ])

            # Convert to PyTorch tensors
            input_tensor = torch.from_numpy(input_tensor).float()
            f_star_value_tensor = torch.tensor([target_value]).float()  # Shape: [1]

            # Optional: Add sanity checks
            assert torch.isfinite(input_tensor).all(), f"Non-finite values found in input_tensor at index {idx}"
            assert torch.isfinite(f_star_value_tensor), f"Non-finite value found in f_star_value_tensor at index {idx}"

            return input_tensor, f_star_value_tensor
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise e

# ---------------------------
# Custom Loss Function with Curriculum Learning
# ---------------------------

def custom_loss_function(all_inputs, output, target, lambdas):
    """
    Custom loss function with dynamic lambdas for curriculum learning.

    Args:
        all_inputs (torch.Tensor): Input tensor with gradients enabled.
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth values.
        lambdas (tuple): Tuple containing current values of lambda1, lambda2, lambda3.

    Returns:
        tuple: Total loss and individual loss components for logging.
    """
    lambda1, lambda2, lambda3 = lambdas
    mse_loss = nn.MSELoss()(output, target)
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=all_inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]

    grad_g = gradients[:, 6]
    grad_h = gradients[:, 7]

    # Penalty terms
    penalty_prop1 = torch.mean(torch.relu(-grad_g)) + torch.mean(torch.relu(-grad_h))
    penalty_prop2 = torch.mean(torch.relu(grad_g - grad_h))
    penalty_prop4 = torch.mean(torch.relu(grad_g + grad_h - 2))

    # Total penalty
    total_penalty = lambda1 * penalty_prop1 + lambda2 * penalty_prop2 + lambda3 * penalty_prop4

    # Total loss
    total_loss = mse_loss + total_penalty

    # Return individual penalty terms for logging
    return total_loss, mse_loss.item(), total_penalty.item(), penalty_prop1.item(), penalty_prop2.item(), penalty_prop4.item()

# ---------------------------
# Training Functions with Curriculum Learning
# ---------------------------

def train_model_with_curriculum(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10,
                                model_path="model.pth", criterion=None, lambda_schedule=None):
    """
    Train the model using curriculum learning by dynamically adjusting lambdas.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the training on.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        model_path (str): Path to save the best model.
        criterion (callable): Custom loss function.
        lambda_schedule (dict): Dictionary containing lambda schedules.

    Returns:
        tuple: Trained model and best validation loss.
    """
    print(f"Training model with curriculum learning.")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training loop with curriculum learning...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_penalty_loss = 0.0
        train_penalty_prop1 = 0.0
        train_penalty_prop2 = 0.0
        train_penalty_prop4 = 0.0

        # Get current lambdas from the schedule
        current_lambdas = (
            lambda_schedule['lambda1'](epoch),
            lambda_schedule['lambda2'](epoch),
            lambda_schedule['lambda3'](epoch)
        )

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad_(True)
            optimizer.zero_grad()
            output, all_inputs = model(data)

            loss, mse_loss_value, penalty_loss_value, penalty_prop1_value, penalty_prop2_value, penalty_prop4_value = criterion(
                all_inputs, output, target, current_lambdas)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse_loss += mse_loss_value
            train_penalty_loss += penalty_loss_value
            train_penalty_prop1 += penalty_prop1_value
            train_penalty_prop2 += penalty_prop2_value
            train_penalty_prop4 += penalty_prop4_value

        # Compute averages
        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_penalty_loss /= len(train_loader)
        train_penalty_prop1 /= len(train_loader)
        train_penalty_prop2 /= len(train_loader)
        train_penalty_prop4 /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                loss = nn.MSELoss()(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr}')
        print(f'    Current Lambdas: lambda1={current_lambdas[0]:.4f}, lambda2={current_lambdas[1]:.4f}, lambda3={current_lambdas[2]:.4f}')
        print(f'    Train MSE Loss: {train_mse_loss:.6f}, Train Penalty Loss: {train_penalty_loss:.6f}')
        print(f'    Penalty Prop1: {train_penalty_prop1:.6f}, Penalty Prop2: {train_penalty_prop2:.6f}, Penalty Prop4: {train_penalty_prop4:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_path:
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Best validation loss: {best_val_loss:.6f}")
    return model, best_val_loss

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

def is_valid(pos, map_grid):
    return 0 <= pos[0] < map_grid.shape[0] and 0 <= pos[1] < map_grid.shape[1] and map_grid[pos] == 0

def generate_start_goal(map_grid):
    while True:
        start = (np.random.randint(0, map_grid.shape[0]), np.random.randint(0, map_grid.shape[1]))
        goal = (np.random.randint(0, map_grid.shape[0]), np.random.randint(0, map_grid.shape[1]))
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal:
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

    h = euclidean_distance(current, goal)

    g_normalized = normalize(g, g_min, g_max)
    h_normalized = normalize(h, h_min, h_max)

    start_normalized = np.array(start) / 127.0  # Updated for 128x128 map
    goal_normalized = np.array(goal) / 127.0
    current_normalized = np.array(current) / 127.0
    input_tensor = np.concatenate([
        start_normalized,             # 2 values
        goal_normalized,              # 2 values
        current_normalized,           # 2 values
        [g_normalized, h_normalized], # 2 values
        encoded_map                   # latent_dim values
    ])

    input_tensor = torch.from_numpy(input_tensor).float().to(device)  # Shape: [input_size]
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, input_size]

    model.eval()
    with torch.no_grad():
        f_star_predicted, _ = model(input_tensor)
    f_star_denormalized = denormalize(f_star_predicted.item(), f_star_min, f_star_max)

    return f_star_denormalized

# ---------------------------
# A* Search Implementations
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

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            print(f"Reached maximum expansions ({max_expansions}). Terminating search.")
            return None, expansions

        if current.pos == goal:
            path = []
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], expansions

        closed_set.add(current.pos)

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost

            if next_pos in closed_set and tentative_g >= g_score.get(next_pos, float('inf')):
                continue  # Not a better path

            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                h = euclidean_distance(next_pos, goal)
                next_node = AStarNode(next_pos, g=tentative_g, h=h, parent=current)
                heapq.heappush(open_list, next_node)


class ModelNode:
    def __init__(self, pos, f_star, g, parent=None):
        self.pos = pos
        self.f_star = f_star
        self.g = g
        self.parent = parent

    def __lt__(self, other):
        return self.f_star < other.f_star

def astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, max_expansions=100000):
    open_list = []
    closed_set = set()
    g_score = {start: 0}
    start_f_star = run_inference(map_grid, start, goal, start, 0, encoder, model, normalization_values, device)
    start_node = ModelNode(start, f_star=start_f_star, g=0)
    heapq.heappush(open_list, start_node)
    expansions = 0

    while open_list:
        current = heapq.heappop(open_list)
        expansions += 1

        if expansions > max_expansions:
            print(f"Reached maximum expansions ({max_expansions}). Terminating search.")
            return None, expansions

        if current.pos == goal:
            path = []
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1], expansions

        closed_set.add(current.pos)

        for next_pos, cost in get_neighbors(current.pos):
            if not is_valid(next_pos, map_grid):
                continue

            tentative_g = current.g + cost

            if next_pos in closed_set and tentative_g >= g_score.get(next_pos, float('inf')):
                continue  # Not a better path

            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                f_star = run_inference(map_grid, start, goal, next_pos, tentative_g, encoder, model, normalization_values, device)
                next_node = ModelNode(next_pos, f_star=f_star, g=tentative_g, parent=current)
                heapq.heappush(open_list, next_node)


# ---------------------------
# Evaluation Functions
# ---------------------------

def create_evaluation_data(num_maps=10, num_queries_per_map=5, map_size=128, save_path='evaluation_data.pkl'):
    evaluation_data = []
    for i in range(num_maps):
        map_grid = generate_map(width=map_size, height=map_size)
        for j in range(num_queries_per_map):
            start, goal = generate_start_goal(map_grid)
            evaluation_data.append((map_grid, start, goal))
    # Save evaluation data to a file
    with open(save_path, 'wb') as f:
        pickle.dump(evaluation_data, f)
    print(f"Evaluation data created and saved to {save_path}.")

def compute_path_cost(path):
    if path is None:
        return np.inf
    cost = 0.0
    for i in range(len(path) - 1):
        if abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1]) == 2:
            cost += np.sqrt(2)
        else:
            cost += 1
    return cost

def evaluate_model_on_data(model, evaluation_data, encoder, normalization_values, device, max_expansions=10000):
    total_expansion_reduction = 0.0
    total_additional_path_cost = 0.0
    num_valid_cases = 0

    for map_grid, start, goal in evaluation_data:
        # Run traditional A* with expansion limit
        traditional_result = astar_traditional(start, goal, map_grid, max_expansions=max_expansions)
        if traditional_result is None:
            continue  # Skip if no path found or expansion limit reached
        traditional_path, traditional_expanded = traditional_result
        traditional_path_cost = compute_path_cost(traditional_path)
        print(f"A* path cost: {traditional_path_cost}, expansions: {traditional_expanded}")

        # Run model-based A* with expansion limit
        model_result = astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, max_expansions=max_expansions)
        if model_result is None:
            continue  # Skip if no path found or expansion limit reached
        model_path, model_expanded = model_result
        model_path_cost = compute_path_cost(model_path)
        print(f"Neural A* path cost: {model_path_cost}, expansions: {model_expanded}\n\n")

        # Compute metrics
        expansions_diff = 100 * (traditional_expanded - model_expanded) / traditional_expanded
        cost_diff = 100 * (model_path_cost - traditional_path_cost) / traditional_path_cost

        total_expansion_reduction += expansions_diff
        total_additional_path_cost += cost_diff
        num_valid_cases += 1
    print(f"\nSuccesfully ran {num_valid_cases} queries... moving on\n")

    if num_valid_cases == 0:
        return 0.0, 100.0  # Return worst-case values if no valid cases

    avg_expansion_reduction = total_expansion_reduction / num_valid_cases
    avg_additional_path_cost = total_additional_path_cost / num_valid_cases

    return avg_expansion_reduction, avg_additional_path_cost

# ---------------------------
# Normalization Functions
# ---------------------------

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val
    
# ---------------------------
# Hyperparameter Optimization with Curriculum Learning
# ---------------------------

def optimize_with_curriculum(model, train_loader, val_loader, device, args):
    print("Starting hyperparameter optimization using curriculum learning...")

    # Load evaluation data
    with open(args.evaluation_data_path, 'rb') as f:
        evaluation_data = pickle.load(f)
    print(f"Loaded evaluation data with {len(evaluation_data)} queries.")

    # Load normalization values
    with open(args.norm_path, 'rb') as f:
        normalization_values = pickle.load(f)

    # Initialize Encoder
    encoder = UNet2DAutoencoder(input_channels=1, latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    encoder.eval()
    print(f"Loaded encoder model from {args.encoder_path}")

    def objective(trial):
        # Suggest hyperparameters for lambdas and curriculum schedule
        lambda1_start = trial.suggest_float('lambda1_start', 0.0, 0.1)
        lambda1_end = trial.suggest_float('lambda1_end', 0.0, 1.0)
        lambda2_start = trial.suggest_float('lambda2_start', 0.0, 0.1)
        lambda2_end = trial.suggest_float('lambda2_end', 0.0, 1.0)
        lambda3_start = trial.suggest_float('lambda3_start', 0.0, 0.1)
        lambda3_end = trial.suggest_float('lambda3_end', 0.0, 1.0)
        growth_rate = trial.suggest_float('growth_rate', 0.5, 2.0)
        epochs = trial.suggest_int('epochs', 5, 50)
        print(f"Trial {trial.number}: Lambdas start at ({lambda1_start}, {lambda2_start}, {lambda3_start}), "
              f"end at ({lambda1_end}, {lambda2_end}, {lambda3_end}), growth_rate={growth_rate}, epochs={epochs}")

        # Define lambda schedules
        def create_lambda_schedule(start, end):
            def schedule(epoch):
                progress = epoch / max(epochs - 1, 1)
                return start + (end - start) * (progress ** growth_rate)
            return schedule

        lambda_schedule = {
            'lambda1': create_lambda_schedule(lambda1_start, lambda1_end),
            'lambda2': create_lambda_schedule(lambda2_start, lambda2_end),
            'lambda3': create_lambda_schedule(lambda3_start, lambda3_end)
        }

        # Define custom loss function with dynamic lambdas
        def custom_loss_function_opt(all_inputs, output, target, lambdas):
            return custom_loss_function(all_inputs, output, target, lambdas)

        # Initialize a new model for each trial
        output_size = 1
        input_size = (3 * 2) + args.latent_dim + 2
        trial_model = MLPModel(input_size, output_size).to(device)

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(trial_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2)

        # Training loop with pruning
        for epoch in range(epochs):
            trial_model.train()
            train_loss = 0.0

            # Get current lambdas
            current_lambdas = (
                lambda_schedule['lambda1'](epoch),
                lambda_schedule['lambda2'](epoch),
                lambda_schedule['lambda3'](epoch)
            )

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data.requires_grad_(True)
                optimizer.zero_grad()
                output, all_inputs = trial_model(data)
                loss, _, _, _, _, _ = custom_loss_function_opt(all_inputs, output, target, current_lambdas)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= (batch_idx + 1)

            # Validation loop
            trial_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    output, _ = trial_model(data)
                    loss = nn.MSELoss()(output, target)
                    val_loss += loss.item()

            val_loss /= (batch_idx + 1)

            # Report intermediate validation loss and handle pruning
            trial.report(val_loss, epoch)

            # Prune trial if needed
            if trial.should_prune():
                raise TrialPruned()

            # Adjust learning rate
            scheduler.step(val_loss)

        # Evaluate the model using evaluate_model_on_data
        avg_expansion_reduction, avg_additional_path_cost = evaluate_model_on_data(
            trial_model, evaluation_data, encoder, normalization_values, device, max_expansions=args.max_expansions
        )

        # Compute the objective value
        objective_value = avg_expansion_reduction - 2 * avg_additional_path_cost

        # Report the objective value
        trial.report(objective_value, epoch)

        return objective_value

    # Create an Optuna study with direction='maximize' since we want to maximize the objective value
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    print("Optimization completed.")
    print("Best hyperparameters: ", study.best_params)
    print("Best objective value: ", study.best_value)

    # Retrieve best hyperparameters and retrain the final model
    best_lambda1_start = study.best_params['lambda1_start']
    best_lambda1_end = study.best_params['lambda1_end']
    best_lambda2_start = study.best_params['lambda2_start']
    best_lambda2_end = study.best_params['lambda2_end']
    best_lambda3_start = study.best_params['lambda3_start']
    best_lambda3_end = study.best_params['lambda3_end']
    best_growth_rate = study.best_params['growth_rate']
    best_epochs = study.best_params['epochs']

    # Define lambda schedules with best hyperparameters
    def create_lambda_schedule_final(start, end):
        def schedule(epoch):
            progress = epoch / max(best_epochs - 1, 1)
            return start + (end - start) * (progress ** best_growth_rate)
        return schedule

    lambda_schedule_final = {
        'lambda1': create_lambda_schedule_final(best_lambda1_start, best_lambda1_end),
        'lambda2': create_lambda_schedule_final(best_lambda2_start, best_lambda2_end),
        'lambda3': create_lambda_schedule_final(best_lambda3_start, best_lambda3_end)
    }

    # Define the custom loss function with dynamic lambdas
    def custom_loss_function_best(all_inputs, output, target, lambdas):
        return custom_loss_function(all_inputs, output, target, lambdas)

    # Initialize a new model for final training
    output_size = 1
    input_size = (3 * 2) + args.latent_dim + 2
    final_model = MLPModel(input_size, output_size).to(device)

    # Load the pre-trained weights into the model (if available)
    if args.pretrained_model_path and os.path.isfile(args.pretrained_model_path):
        final_model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
        print(f"Loaded pre-trained model from {args.pretrained_model_path}")
    else:
        print("No pre-trained model provided. Training from scratch.")

    print("Retraining model with best hyperparameters and curriculum learning...")
    trained_model, best_val_loss = train_model_with_curriculum(
        final_model,
        train_loader,
        val_loader,
        device,
        epochs=best_epochs,  # Use the best number of epochs from optimization
        lr=args.lr,
        model_path=args.model_save_path,
        criterion=custom_loss_function_best,
        lambda_schedule=lambda_schedule_final
    )

    print(f"Best model retrained with optimal hyperparameters and saved to {args.model_save_path}")

# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train F* Prediction Model using Curriculum Learning")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the generated dataset pickle file")
    parser.add_argument("--norm_path", type=str, required=True,
                        help="Path to the normalization values pickle file")
    parser.add_argument("--encoder_path", type=str, required=True,
                        help="Path to the pre-trained encoder model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--latent_dim", type=int, default=512,
                        help="Dimension for autoencoder vector")
    parser.add_argument("--model_save_path", type=str,
                        default="model.pth", help="Path to save the trained model")
    parser.add_argument("--learning_type", type=str, choices=[
                        'heuristic', 'priority'], default='priority', help="Choose the type of function that is being learned.")
    parser.add_argument("--optimize_lambdas", action='store_true',
                        help="Whether to perform lambda optimization after training")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of trials for hyperparameter optimization")
    parser.add_argument("--evaluation_data_path", type=str, default='evaluation_data.pkl',
                        help="Path to the saved evaluation data for hyperparameter optimization")
    parser.add_argument("--create_evaluation_data", action='store_true',
                        help="Whether to create evaluation data before training")
    parser.add_argument("--num_eval_maps", type=int, default=10,
                        help="Number of maps to generate for evaluation data")
    parser.add_argument("--num_queries_per_map", type=int, default=5,
                        help="Number of queries per map for evaluation data")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to a pre-trained MLP model to initialize the final model")
    parser.add_argument("--max_expansions", type=int, default=10000,
                        help="Maximum number of node expansions in A* during evaluation")
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------

def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create evaluation data if requested
    if args.create_evaluation_data:
        create_evaluation_data(num_maps=args.num_eval_maps, num_queries_per_map=args.num_queries_per_map, save_path=args.evaluation_data_path)

    # Load normalization values
    if not os.path.isfile(args.norm_path):
        raise FileNotFoundError(f"Normalization values not found at {args.norm_path}")

    with open(args.norm_path, 'rb') as f:
        normalization_values = pickle.load(f)
    print(f"Loaded normalization values from {args.norm_path}")

    # Load dataset
    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset from {args.dataset_path}")

    full_dataset = FStarDataset(dataset)

    # Split into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, num_workers=0)

    # Initialize MLP Model
    output_size = 1
    input_size = (3 * 2) + args.latent_dim + 2  # start, goal, current, g_normalized, h_normalized
    print(f"Calculated input size for MLP: {input_size}")
    model = MLPModel(input_size, output_size).to(device)

    # Initialize Encoder
    encoder = UNet2DAutoencoder(input_channels=1, latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    encoder.eval()
    print(f"Loaded encoder model from {args.encoder_path}")

    # Choose training type based on arguments
    if args.learning_type in ['heuristic', 'priority']:
        print(f"Training MLPModel with learning type: {args.learning_type}")
    else:
        print(f"Invalid learning type: {args.learning_type}. Choose 'heuristic' or 'priority'.")
        return

    # Optionally, perform lambda optimization with curriculum learning
    if args.optimize_lambdas:
        optimize_with_curriculum(model, train_loader, val_loader, device, args)
    else:
        # Define default lambda schedules (no curriculum)
        lambda_schedule = {
            'lambda1': lambda epoch: 0.05,
            'lambda2': lambda epoch: 0.05,
            'lambda3': lambda epoch: 0.05
        }

        # Define the custom loss function
        def custom_loss_function_default(all_inputs, output, target, lambdas):
            return custom_loss_function(all_inputs, output, target, lambdas)

        # Train the MLP Model with curriculum learning
        print("Training with default lambdas and curriculum learning.")
        trained_model, best_val_loss = train_model_with_curriculum(
            model, train_loader, val_loader, device, epochs=args.epochs,
            lr=args.lr, model_path=args.model_save_path, criterion=custom_loss_function_default,
            lambda_schedule=lambda_schedule
        )

        print(f"Training completed. Best model saved as {args.model_save_path}")

if __name__ == '__main__':
    main()
