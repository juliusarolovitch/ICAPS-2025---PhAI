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
from collections import deque  # Added for BFS in path_exists

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
# Custom Loss Function
# ---------------------------

def custom_loss_function(all_inputs, output, target, lambda1=0.05, lambda2=0.05, lambda3=0.05):
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
# Training Functions
# ---------------------------

def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10, model_path="model.pth", loss_fn="mse", criterion=None):
    print(f"Training model with loss function: {loss_fn}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    if criterion is None:
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = custom_loss_function

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training loop...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_penalty_loss = 0.0
        train_penalty_prop1 = 0.0
        train_penalty_prop2 = 0.0
        train_penalty_prop4 = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad_(True)
            optimizer.zero_grad()
            output, all_inputs = model(data)

            if loss_fn == "mse":
                loss = criterion(output, target)
            else:
                loss, mse_loss_value, penalty_loss_value, penalty_prop1_value, penalty_prop2_value, penalty_prop4_value = criterion(all_inputs, output, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if loss_fn != "mse":
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
        # if loss_fn != "mse":
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
    max_depth=3,  # Reduced from 5 to 3 for simpler maps
    wall_thickness=2,
    min_openings=2,  # Increased to have more openings
    max_openings=4,
    opening_size=6,  # Increased to make openings larger
    min_obstacles=1,  # Reduced number of obstacles
    max_obstacles=3,
    min_obstacle_size=5,
    max_obstacle_size=7
):
    """
    Generates a 2D map with randomly sized and positioned rooms separated by walls with random openings.
    Each room contains a random number of smaller obstacles.

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

def path_exists(map_grid, start, goal):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for neighbor, _ in get_neighbors(current):
            if is_valid(neighbor, map_grid) and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

def generate_start_goal(map_grid):
    attempts = 0
    while attempts < 100:
        start = (np.random.randint(0, map_grid.shape[0]), np.random.randint(0, map_grid.shape[1]))
        goal = (np.random.randint(0, map_grid.shape[0]), np.random.randint(0, map_grid.shape[1]))
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal:
            if path_exists(map_grid, start, goal):
                return start, goal
        attempts += 1
    raise Exception("Failed to find valid start and goal positions.")

def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ---------------------------
# Normalization Functions
# ---------------------------

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

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
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue  # Not a better path

            g_score[next_pos] = tentative_g
            f_star = run_inference(map_grid, start, goal, next_pos, tentative_g, encoder, model, normalization_values, device)
            next_node = ModelNode(next_pos, f_star=f_star, g=tentative_g, parent=current)
            heapq.heappush(open_list, next_node)

    return None, expansions


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
            if not is_valid(next_pos, map_grid) or next_pos in closed_set:
                continue

            tentative_g = current.g + cost

            if next_pos in g_score and tentative_g >= g_score[next_pos]:
                continue  # Not a better path

            g_score[next_pos] = tentative_g
            h = euclidean_distance(next_pos, goal)
            next_node = AStarNode(next_pos, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, next_node)

    return None, expansions


# ---------------------------
# Visualization Function
# ---------------------------

def visualize_map(map_grid, path=None, start=None, goal=None, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(map_grid, cmap='gray_r')
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], color='blue', linewidth=2)
    if start:
        plt.plot(start[1], start[0], 'go')  # Green dot for start
    if goal:
        plt.plot(goal[1], goal[0], 'ro')  # Red dot for goal
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# ---------------------------
# Evaluation Functions
# ---------------------------

def create_evaluation_data(num_maps=10, num_queries_per_map=5, map_size=128, save_path='evaluation_data.pkl'):
    evaluation_data = []
    for i in range(num_maps):
        map_grid = generate_map(width=map_size, height=map_size)
        for j in range(num_queries_per_map):
            try:
                start, goal = generate_start_goal(map_grid)
                evaluation_data.append((map_grid, start, goal))
            except Exception as e:
                print(f"Failed to generate start and goal for map {i}, query {j}: {e}")
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

def evaluate_model_on_data(model, evaluation_data, encoder, normalization_values, device, max_expansions=100000):
    total_expansion_reduction = 0.0
    total_additional_path_cost = 0.0
    num_valid_cases = 0

    print("Beginning evaluation...")

    for idx, (map_grid, start, goal) in enumerate(evaluation_data):
        print(f"Evaluating map {idx + 1}/{len(evaluation_data)}")
        print(f"Start: {start}, Goal: {goal}")

        # Run traditional A* with expansion limit
        traditional_result = astar_traditional(start, goal, map_grid, max_expansions=max_expansions)
        if traditional_result is None:
            print(f"No path found by traditional A* for map {idx + 1}")
            save_path = f"traditional_astar_no_path_map_{idx + 1}.png"
            visualize_map(map_grid, start=start, goal=goal, save_path=save_path)
            continue  # Skip if no path found or expansion limit reached
        traditional_path, traditional_expanded = traditional_result
        traditional_path_cost = compute_path_cost(traditional_path)
        print(f"A* path cost: {traditional_path_cost}, expansions: {traditional_expanded}")

        # Run model-based A* with expansion limit
        model_result = astar_with_model(start, goal, map_grid, encoder, model, normalization_values, device, max_expansions=max_expansions)
        if model_result is None:
            print(f"No path found by neural A* for map {idx + 1}")
            save_path = f"neural_astar_no_path_map_{idx + 1}.png"
            visualize_map(map_grid, start=start, goal=goal, save_path=save_path)
            continue  # Skip if no path found or expansion limit reached
        model_path, model_expanded = model_result
        model_path_cost = compute_path_cost(model_path)
        print(f"Neural A* path cost: {model_path_cost}, expansions: {model_expanded}\n\n")

        # Save some maps with paths
        if idx < 5:  # Save the first 5 maps
            save_path_traditional = f"traditional_astar_path_map_{idx + 1}.png"
            save_path_neural = f"neural_astar_path_map_{idx + 1}.png"
            visualize_map(map_grid, path=traditional_path, start=start, goal=goal, save_path=save_path_traditional)
            visualize_map(map_grid, path=model_path, start=start, goal=goal, save_path=save_path_neural)

        # Compute metrics
        expansions_diff = 100 * (traditional_expanded - model_expanded) / traditional_expanded
        cost_diff = 100 * (model_path_cost - traditional_path_cost) / traditional_path_cost

        total_expansion_reduction += expansions_diff
        total_additional_path_cost += cost_diff
        num_valid_cases += 1

    if num_valid_cases == 0:
        return 0.0, 100.0  # Return worst-case values if no valid cases

    avg_expansion_reduction = total_expansion_reduction / num_valid_cases
    avg_additional_path_cost = total_additional_path_cost / num_valid_cases

    return avg_expansion_reduction, avg_additional_path_cost

# ---------------------------
# Hyperparameter Optimization
# ---------------------------

def optimize_lambdas(model, train_loader, val_loader, device, args):
    print("Starting hyperparameter optimization using evaluation metrics...")

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
        # Suggest hyperparameters
        lambda1 = trial.suggest_float('lambda1', 0.0, 1.0)
        lambda2 = trial.suggest_float('lambda2', 0.0, 1.0)
        lambda3 = trial.suggest_float('lambda3', 0.0, 1.0)
        # Suggest number of epochs
        epochs = trial.suggest_int('epochs', 2, 30)
        print(f"Trial {trial.number}: lambda1={lambda1:.4f}, lambda2={lambda2:.4f}, lambda3={lambda3:.4f}, epochs={epochs}")

        # Define custom loss function with suggested hyperparameters
        def custom_loss_function_opt(all_inputs, output, target):
            return custom_loss_function(all_inputs, output, target,
                                        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

        # Initialize a new model for each trial
        output_size = 1
        input_size = (3 * 2) + args.latent_dim + 2
        trial_model = MLPModel(input_size, output_size).to(device)

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(trial_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2)

        # Training loop with pruning
        for epoch in tqdm(range(epochs), desc="Training..."):
            trial_model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data.requires_grad_(True)
                optimizer.zero_grad()
                output, all_inputs = trial_model(data)
                loss, _, _, _, _, _ = custom_loss_function_opt(all_inputs, output, target)
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
    best_lambda1 = study.best_params['lambda1']
    best_lambda2 = study.best_params['lambda2']
    best_lambda3 = study.best_params['lambda3']
    best_epochs = study.best_params['epochs']

    # Define the custom loss function with the best lambdas
    def custom_loss_function_best(all_inputs, output, target):
        return custom_loss_function(all_inputs, output, target,
                                    lambda1=best_lambda1, lambda2=best_lambda2, lambda3=best_lambda3)

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

    print("Retraining model with best hyperparameters...")
    trained_model, best_val_loss = train_model(
        final_model,
        train_loader,
        val_loader,
        device,
        epochs=best_epochs,  # Use the best number of epochs from optimization
        lr=args.lr,
        model_path=args.model_save_path,
        loss_fn="custom",
        criterion=custom_loss_function_best
    )

    print(f"Best model retrained with optimal hyperparameters and saved to {args.model_save_path}")

# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train F* Prediction Model using a Generated Dataset")
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
    parser.add_argument("--loss_function", type=str, choices=[
                        'mse', 'custom'], default='custom', help="Choose between MSE and custom loss function")
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
    parser.add_argument("--max_expansions", type=int, default=100000,
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

    # Optionally, perform lambda optimization if required
    if args.optimize_lambdas:
        optimize_lambdas(model, train_loader, val_loader, device, args)
    else:
        # Train the MLP Model
        if args.loss_function == "custom":
            print("Training with custom loss function.")
            trained_model, best_val_loss = train_model(model, train_loader, val_loader, device, epochs=args.epochs,
                                        lr=args.lr, model_path=args.model_save_path, loss_fn=args.loss_function)
        else:
            print("Training with MSE loss function.")
            trained_model, best_val_loss = train_model(model, train_loader, val_loader, device, epochs=args.epochs,
                                        lr=args.lr, model_path=args.model_save_path, loss_fn=args.loss_function)

        print(f"Training completed. Best model saved as {args.model_save_path}")

if __name__ == '__main__':
    main()
