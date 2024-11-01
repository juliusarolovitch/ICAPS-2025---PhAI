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
from tqdm import tqdm
import math
import random
import time
from itertools import count
import bisect
from collections import defaultdict

# ---------------------------
# Utility Functions for Sliding Tile Puzzle
# ---------------------------

def get_possible_moves(blank_pos, grid_size=5):
    moves = []
    row = blank_pos // grid_size
    col = blank_pos % grid_size
    if row > 0:  # Up
        moves.append(blank_pos - grid_size)
    if row < grid_size - 1:  # Down
        moves.append(blank_pos + grid_size)
    if col > 0:  # Left
        moves.append(blank_pos - 1)
    if col < grid_size - 1:  # Right
        moves.append(blank_pos + 1)
    return moves

def get_neighbors(state, grid_size=5):
    neighbors = []
    blank_pos = state.index(0)
    possible_moves = get_possible_moves(blank_pos, grid_size)
    for move in possible_moves:
        new_state = list(state)
        new_state[blank_pos], new_state[move] = new_state[move], new_state[blank_pos]
        neighbors.append((tuple(new_state), 1))  # Cost is 1 per move
    return neighbors

def manhattan_distance(state, goal_state, grid_size=5):
    distance = 0
    for i in range(1, grid_size * grid_size):  # Tiles numbered from 1 to N-1
        idx_current = state.index(i)
        idx_goal = goal_state.index(i)
        x_current, y_current = idx_current % grid_size, idx_current // grid_size
        x_goal, y_goal = idx_goal % grid_size, idx_goal // grid_size
        distance += abs(x_current - x_goal) + abs(y_current - y_goal)
    return distance

def linear_conflict(state, goal_state, grid_size=5):
    """Calculate the number of linear conflicts in rows and columns."""
    conflict = 0
    state_grid = np.array(state).reshape(grid_size, grid_size)
    goal_grid = np.array(goal_state).reshape(grid_size, grid_size)
    
    # Analyze rows
    for row in range(grid_size):
        state_row = state_grid[row, :]
        goal_row = goal_grid[row, :]
        # Create mapping from tile value to goal position in the row
        goal_positions = {tile: idx for idx, tile in enumerate(goal_row) if tile != 0}
        for i in range(grid_size):
            for j in range(i + 1, grid_size):
                tile_i = state_row[i]
                tile_j = state_row[j]
                if tile_i != 0 and tile_j != 0:
                    if (tile_i in goal_positions and tile_j in goal_positions and
                        goal_positions[tile_i] > goal_positions[tile_j]):
                        conflict += 1

    # Analyze columns
    for col in range(grid_size):
        state_col = state_grid[:, col]
        goal_col = goal_grid[:, col]
        # Create mapping from tile value to goal position in the column
        goal_positions = {tile: idx for idx, tile in enumerate(goal_col) if tile != 0}
        for i in range(grid_size):
            for j in range(i + 1, grid_size):
                tile_i = state_col[i]
                tile_j = state_col[j]
                if tile_i != 0 and tile_j != 0:
                    if (tile_i in goal_positions and tile_j in goal_positions and
                        goal_positions[tile_i] > goal_positions[tile_j]):
                        conflict += 1

    return conflict * 2  # Each conflict adds two moves

def misplaced_tiles(state, goal_state):
    """Calculate the number of misplaced tiles."""
    return sum(1 for s, g in zip(state, goal_state) if s != 0 and s != g)

def is_solvable(state, grid_size=5):
    inversion_count = 0
    state_wo_blank = [tile for tile in state if tile != 0]
    for i in range(len(state_wo_blank)):
        for j in range(i + 1, len(state_wo_blank)):
            if state_wo_blank[i] > state_wo_blank[j]:
                inversion_count += 1

    if grid_size % 2 == 1:
        return inversion_count % 2 == 0
    else:
        blank_pos = state.index(0)
        blank_row_from_bottom = grid_size - (blank_pos // grid_size)
        if blank_row_from_bottom % 2 == 0:
            return inversion_count % 2 == 1
        else:
            return inversion_count % 2 == 0

def generate_random_puzzle_state(num_moves=100, grid_size=5):
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    state = list(goal_state)
    blank_pos = state.index(0)
    for _ in range(num_moves):
        moves = get_possible_moves(blank_pos, grid_size)
        move = random.choice(moves)
        state[blank_pos], state[move] = state[move], state[blank_pos]
        blank_pos = move
    return tuple(state)

# ---------------------------
# Normalization Functions
# ---------------------------

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# ---------------------------
# MLP Model Class
# ---------------------------

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[256, 128, 64]):
        super(MLPModel, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        return output  # Return only output tensor

# ---------------------------
# Custom Loss Function
# ---------------------------

def custom_loss_function(all_inputs, output, target, lambda1=0.0, lambda2=0.0, lambda3=0.0):
    """
    Computes a custom loss combining Mean Squared Error with gradient-based penalties.

    Args:
        all_inputs (torch.Tensor): Input tensor with requires_grad=True.
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth values.
        lambda1 (float): Weight for gradient positivity penalty.
        lambda2 (float): Weight for gradient comparison penalty.
        lambda3 (float): Weight for gradient sum penalty.

    Returns:
        tuple: (total_loss, mse_loss, gradient_loss1, gradient_loss2, gradient_loss3)
    """
    # Compute Mean Squared Error Loss
    mse_loss = nn.MSELoss()(output, target)
    
    # Compute gradients of the output with respect to the inputs
    grad_all_inputs = torch.autograd.grad(
        outputs=output,
        inputs=all_inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]
    
    grad_f_min = grad_all_inputs[:, 0]  # Assuming f_min is the first feature
    grad_g = grad_all_inputs[:, 1]      # g is the second feature
    grad_h = grad_all_inputs[:, 2]      # h is the third feature
    
    # Sufficient conditions to reduce re-expansions
    # 1. grad_f_min should be non-negative
    # 2. grad_g should not exceed grad_h
    # 3. The sum of grad_g and grad_h should be greater than or equal to sqrt(2)
    
    gradient_loss1 = torch.mean(F.relu(-grad_f_min))  # f_min should not decrease
    gradient_loss2 = torch.mean(F.relu(grad_g - grad_h))  # g should not exceed h
    gradient_loss3 = torch.mean(F.relu(math.sqrt(2) - (grad_g + grad_h)))  # g + h >= sqrt(2)
    
    total_loss = mse_loss + lambda1 * gradient_loss1 + lambda2 * gradient_loss2 + lambda3 * gradient_loss3
    return total_loss, mse_loss.item(), gradient_loss1.item(), gradient_loss2.item(), gradient_loss3.item()

# ---------------------------
# Inference Function (Corrected)
# ---------------------------

def run_inference(f_min, g, h, model, normalization_values, device):
    """
    Runs inference using the trained model to predict the priority.

    Args:
        f_min (float): The current minimum f-value in the open list.
        g (float): The cost to reach the current state.
        h (float): The heuristic estimate from the current state to the goal.
        model (nn.Module): The trained MLP model.
        normalization_values (dict): Dictionary containing normalization parameters.
        device (torch.device): The device to run the model on.

    Returns:
        float: The predicted priority value.
    """
    # Normalize inputs
    f_min_norm = normalize(f_min, normalization_values['f_min_min'], normalization_values['f_min_max'])
    g_norm = normalize(g, normalization_values['g_min'], normalization_values['g_max'])
    h_norm = normalize(h, normalization_values['h_min'], normalization_values['h_max'])

    # Prepare input tensor
    input_tensor = torch.tensor([f_min_norm, g_norm, h_norm], dtype=torch.float32).to(device)
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, 3]

    model.eval()
    with torch.no_grad():
        priority_pred = model(input_tensor)
    priority_denorm = denormalize(priority_pred.item(), normalization_values['priority_min'], normalization_values['priority_max'])

    return priority_denorm

# ---------------------------
# Modified Potential Search Function to Collect Data
# ---------------------------

class PotentialNode:
    def __init__(self, state, g, h, flnr, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.flnr = flnr  # Priority function value
        self.parent = parent

    def __lt__(self, other):
        return self.flnr < other.flnr

def potential_search_collect_samples(start_state, goal_state, model, normalization_values, device, data_samples, B=3.0, max_runtime=60, grid_size=5):
    """
    Dynamic Potential Search (DPS) Algorithm modified to collect data samples.

    Parameters:
        start_state (tuple): The initial state of the puzzle.
        goal_state (tuple): The goal state of the puzzle.
        model (nn.Module): The trained MLP model.
        normalization_values (dict): Normalization parameters for the model.
        device (torch.device): The device to run the model on.
        data_samples (list): List to collect data samples as tuples (f_min, g, h, priority).
        B (float): Suboptimality bound.
        max_runtime (int): Maximum allowed runtime in seconds.
        grid_size (int): Size of the puzzle grid.

    Returns:
        tuple: (path, total_cost, expansions, re_expansions, reshuffles)
    """
    import math
    start_time = time.time()
    open_list = []
    g_score = {start_state: 0}

    # Compute h_start using standard heuristic
    h_standard_start = manhattan_distance(start_state, goal_state, grid_size) + linear_conflict(start_state, goal_state, grid_size)

    # Compute initial priority using the model
    f_min = h_standard_start  # Since g=0
    priority_start = h_standard_start / (B * f_min - 0 + 1e-8)  # flnr = h / (C - g)

    # Initialize the starting node
    start_node = PotentialNode(state=start_state, g=0, h=h_standard_start, flnr=priority_start, parent=None)
    heapq.heappush(open_list, start_node)
    expansions = 0
    expansion_counts = defaultdict(int)
    reshuffles = 0

    while open_list:
        # Check for runtime timeout
        if time.time() - start_time > max_runtime:
            re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
            return None, None, expansions, re_expansions, reshuffles  # Timeout

        # Pop the node with the lowest potential
        current_node = heapq.heappop(open_list)
        expansions += 1
        expansion_counts[current_node.state] += 1

        # Check if goal is reached within cost bound
        if current_node.state == goal_state and current_node.g <= B * f_min:
            # Reconstruct path
            path = []
            node = current_node
            while node:
                path.append(node.state)
                node = node.parent
            re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
            return path[::-1], current_node.g, expansions, re_expansions, reshuffles

        # Explore neighbors
        for neighbor_state, cost in get_neighbors(current_node.state, grid_size):
            tentative_g = current_node.g + cost

            # Prune paths that exceed the current cost bound
            if tentative_g > B * f_min:
                continue

            # Compute heuristic
            h_standard = manhattan_distance(neighbor_state, goal_state, grid_size) + linear_conflict(neighbor_state, goal_state, grid_size)

            # Predict priority using the model
            priority = run_inference(f_min, tentative_g, h_standard, model, normalization_values, device)

            # Update f_min if necessary
            if priority < f_min and (f_min - priority) >= math.sqrt(2):
                f_min = priority
                C = B * f_min
                reshuffles += 1

                # Recompute priority for all nodes in open list
                new_open_list = []
                while open_list:
                    node = heapq.heappop(open_list)
                    updated_priority = node.h / (C - node.g + 1e-8)
                    node.flnr = updated_priority
                    heapq.heappush(new_open_list, node)
                open_list = new_open_list

            # If a better path is found, add to open list
            if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                g_score[neighbor_state] = tentative_g
                flnr = h_standard / (B * f_min - tentative_g + 1e-8)
                neighbor_node = PotentialNode(state=neighbor_state, g=tentative_g, h=h_standard, flnr=flnr, parent=current_node)
                heapq.heappush(open_list, neighbor_node)

                # Collect data samples
                if flnr > 0 and not math.isinf(flnr):
                    data_samples.append((f_min, tentative_g, h_standard, flnr))

    re_expansions = sum(1 for count in expansion_counts.values() if count > 1)
    return None, None, expansions, re_expansions, reshuffles  # No solution found

# ---------------------------
# Data Collection Function
# ---------------------------

def collect_data(model, normalization_values, device, grid_size, num_puzzles=100, max_runtime=60, max_samples=100000):
    """
    Collects data samples by running potential search on multiple puzzles.

    Parameters:
        model (nn.Module): The trained MLP model.
        normalization_values (dict): Normalization parameters for the model.
        device (torch.device): The device to run the model on.
        grid_size (int): Size of the puzzle grid.
        num_puzzles (int): Number of puzzles to solve for data collection.
        max_runtime (int): Maximum runtime per puzzle in seconds.
        max_samples (int): Maximum number of samples to collect.

    Returns:
        list: Collected data samples as tuples (f_min, g, h, priority).
    """
    data_samples = []
    puzzles_solved = 0
    attempts = 0

    while puzzles_solved < num_puzzles and len(data_samples) < max_samples:
        attempts += 1
        print(f"\nData Collection: Attempt {attempts} for puzzle {puzzles_solved + 1}/{num_puzzles}")
        
        # Generate a new puzzle
        start_state = generate_random_puzzle_state(num_moves=random.randint(100, 120), grid_size=grid_size)
        while not is_solvable(start_state, grid_size):
            start_state = generate_random_puzzle_state(num_moves=random.randint(100, 120), grid_size=grid_size)
        goal_state = tuple(range(1, grid_size * grid_size)) + (0,)

        # Run potential search and collect data
        path, cost, expansions, re_expansions, reshuffles = potential_search_collect_samples(
            start_state,
            goal_state,
            model,
            normalization_values,
            device,
            data_samples,
            B=3.0,
            max_runtime=max_runtime,
            grid_size=grid_size
        )

        if path is not None:
            puzzles_solved += 1
            print(f"Puzzle {puzzles_solved} solved. Collected {len(data_samples)} samples so far.")
        else:
            print("Puzzle not solved within time limit or other error. Skipping.")

    print(f"\nData Collection Completed. Total Samples Collected: {len(data_samples)}")
    return data_samples

# ---------------------------
# Training Function
# ---------------------------

def train_model(data_samples, input_size, device, num_epochs=20, batch_size=128, learning_rate=0.001, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Trains an MLP model using the collected data samples.

    Parameters:
        data_samples (list): List of data samples as tuples (f_min, g, h, priority).
        input_size (int): Number of input features.
        device (torch.device): The device to run the model on.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        lambda1 (float): Weight for gradient positivity penalty.
        lambda2 (float): Weight for gradient comparison penalty.
        lambda3 (float): Weight for gradient sum penalty.

    Returns:
        nn.Module: The trained MLP model.
        dict: Normalization parameters used for training.
    """
    # Extract features and targets
    f_min_vals = np.array([sample[0] for sample in data_samples], dtype=np.float32)
    g_vals = np.array([sample[1] for sample in data_samples], dtype=np.float32)
    h_vals = np.array([sample[2] for sample in data_samples], dtype=np.float32)
    priority_vals = np.array([sample[3] for sample in data_samples], dtype=np.float32)

    # Normalize features and targets
    f_min_min, f_min_max = f_min_vals.min(), f_min_vals.max()
    g_min, g_max = g_vals.min(), g_vals.max()
    h_min, h_max = h_vals.min(), h_vals.max()
    priority_min, priority_max = priority_vals.min(), priority_vals.max()

    f_min_norm = normalize(f_min_vals, f_min_min, f_min_max)
    g_norm = normalize(g_vals, g_min, g_max)
    h_norm = normalize(h_vals, h_min, h_max)
    priority_norm = normalize(priority_vals, priority_min, priority_max)

    # Prepare input and target tensors
    inputs_normalized = np.stack([f_min_norm, g_norm, h_norm], axis=1)
    targets_normalized = priority_norm.reshape(-1, 1)

    inputs_tensor = torch.from_numpy(inputs_normalized).float().to(device)
    targets_tensor = torch.from_numpy(targets_normalized).float().to(device)

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MLPModel(input_size=input_size, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch_inputs, batch_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_inputs.requires_grad = True  # Enable gradient computation for inputs
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss, mse, grad1, grad2, grad3 = custom_loss_function(
                batch_inputs, outputs, batch_targets,
                lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    # Save the trained model and normalization parameters
    model_save_path = "dps_pr.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    norm_values = {
        'f_min_min': f_min_min,
        'f_min_max': f_min_max,
        'g_min': g_min,
        'g_max': g_max,
        'h_min': h_min,
        'h_max': h_max,
        'priority_min': priority_min,
        'priority_max': priority_max
    }
    norm_save_path = "dps_norm_vals.pkl"
    with open(norm_save_path, 'wb') as f:
        pickle.dump(norm_values, f)
    print(f"Normalization parameters saved to {norm_save_path}")

    return model, norm_values

# ---------------------------
# Generate Puzzles Function
# ---------------------------

def generate_puzzles(num_puzzles, grid_size):
    puzzles = []
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    for _ in range(num_puzzles):
        start_state = generate_random_puzzle_state(num_moves=random.randint(120, 150), grid_size=grid_size)
        while not is_solvable(start_state, grid_size):
            start_state = generate_random_puzzle_state(num_moves=random.randint(120, 150), grid_size=grid_size)
        puzzles.append((start_state, goal_state))
    return puzzles

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect data and train a priority prediction model for Dynamic Potential Search.")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the puzzle grid (e.g., 5 for 5x5 grid)")
    parser.add_argument("--num_puzzles", type=int, default=100, help="Number of puzzles to collect data from")
    parser.add_argument("--max_runtime", type=int, default=60, help="Maximum runtime per puzzle in seconds")
    parser.add_argument("--max_samples", type=int, default=100000, help="Maximum number of samples to collect")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for gradient positivity penalty in loss")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight for gradient comparison penalty in loss")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Weight for gradient sum penalty in loss")
    parser.add_argument("--output_model_path", type=str, default="dps_pr.pth", help="Path to save the trained model")
    parser.add_argument("--output_norm_path", type=str, default="dps_norm_vals.pkl", help="Path to save normalization parameters")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize or load a pre-trained model for data collection
    # For the purpose of data collection, we need an initial model.
    # Here, we initialize a random model. In practice, you might want to start with a pre-trained model.
    input_size = 3  # f_min, g, h
    initial_model = MLPModel(input_size=input_size, output_size=1).to(device)
    print("Initialized initial MLP model for data collection.")

    # Optionally, you can load a pre-trained model here
    # initial_model.load_state_dict(torch.load("path_to_pretrained_model.pth", map_location=device))
    # initial_model.eval()

    # Define normalization values for the initial model
    # Since we're initializing a random model, we need to define reasonable normalization parameters
    # These will be updated based on collected data
    normalization_values = {
        'f_min_min': 0.0,
        'f_min_max': 1000.0,
        'g_min': 0.0,
        'g_max': 1000.0,
        'h_min': 0.0,
        'h_max': 1000.0,
        'priority_min': 0.0,
        'priority_max': 1000.0
    }

    # Collect data samples
    data_samples = collect_data(
        model=initial_model,
        normalization_values=normalization_values,
        device=device,
        grid_size=args.grid_size,
        num_puzzles=args.num_puzzles,
        max_runtime=args.max_runtime,
        max_samples=args.max_samples
    )

    # Train the priority prediction model
    trained_model, trained_norm_values = train_model(
        data_samples=data_samples,
        input_size=input_size,
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3
    )

    print("Training completed successfully.")
