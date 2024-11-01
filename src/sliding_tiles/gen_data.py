import numpy as np
import torch
import random
import heapq
import argparse
import os
import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import signal
import sys

# ---------------------------
# Logging Configuration
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ---------------------------
# Node Class for N-Puzzle
# ---------------------------

class Node:
    def __init__(self, state, g=float('inf'), h=0, parent=None):
        self.state = state  # State is a tuple representing the puzzle configuration
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

# ---------------------------
# Utility Functions
# ---------------------------

def get_possible_moves(blank_pos, grid_size=5):
    """
    Returns a list of positions that the blank tile can move to based on grid size.
    """
    moves = []
    row = blank_pos // grid_size
    col = blank_pos % grid_size
    if row > 0:  # Can move up
        moves.append(blank_pos - grid_size)
    if row < grid_size - 1:  # Can move down
        moves.append(blank_pos + grid_size)
    if col > 0:  # Can move left
        moves.append(blank_pos - 1)
    if col < grid_size - 1:  # Can move right
        moves.append(blank_pos + 1)
    return moves

def get_neighbors(state, grid_size=5):
    """
    Given a state, returns a list of (neighbor_state, cost) tuples.
    """
    neighbors = []
    blank_pos = state.index(0)
    possible_moves = get_possible_moves(blank_pos, grid_size)
    for move in possible_moves:
        new_state = list(state)
        # Swap the blank tile with the adjacent tile
        new_state[blank_pos], new_state[move] = new_state[move], new_state[blank_pos]
        neighbors.append((tuple(new_state), 1))  # Cost is 1 per move
    return neighbors

def manhattan_distance(state, goal_state, grid_size=5):
    """
    Calculates the sum of the Manhattan distances of the tiles from their goal positions.
    """
    distance = 0
    for i in range(1, grid_size * grid_size):  # Tiles numbered from 1 to N
        idx_current = state.index(i)
        idx_goal = goal_state.index(i)
        x_current, y_current = idx_current % grid_size, idx_current // grid_size
        x_goal, y_goal = idx_goal % grid_size, idx_goal // grid_size
        distance += abs(x_current - x_goal) + abs(y_current - y_goal)
    return distance

def linear_conflicts(state, goal_state, grid_size=5):
    """
    Calculates the number of linear conflicts in the state with respect to the goal state.
    Each linear conflict adds 2 to the Manhattan distance.
    """
    conflicts = 0

    # Check rows for linear conflicts
    for row in range(grid_size):
        current_row = [state[row * grid_size + col] for col in range(grid_size)]
        goal_row = [goal_state.index(tile) // grid_size for tile in current_row if tile != 0 and goal_state.index(tile) // grid_size == row]
        conflicts += count_linear_conflicts(current_row, goal_row, grid_size)

    # Check columns for linear conflicts
    for col in range(grid_size):
        current_col = [state[row * grid_size + col] for row in range(grid_size)]
        goal_col = [goal_state.index(tile) % grid_size for tile in current_col if tile != 0 and goal_state.index(tile) % grid_size == col]
        conflicts += count_linear_conflicts(current_col, goal_col, grid_size)

    return conflicts * 2  # Each conflict adds 2 to the heuristic

def count_linear_conflicts(current_line, goal_line, grid_size):
    """
    Counts the number of linear conflicts in a single row or column.
    """
    conflicts = 0
    for i in range(len(goal_line)):
        for j in range(i + 1, len(goal_line)):
            if goal_line[i] > goal_line[j]:
                conflicts += 1
    return conflicts

def manhattan_distance_with_linear_conflicts(state, goal_state, grid_size=5):
    """
    Calculates the Manhattan distance with linear conflicts as an improved heuristic.
    """
    return manhattan_distance(state, goal_state, grid_size) + linear_conflicts(state, goal_state, grid_size)

def is_solvable(state, grid_size=5):
    """
    Checks if a given N-puzzle state is solvable.
    For odd grid sizes, the puzzle is solvable if the number of inversions is even.
    For even grid sizes, solvability depends on the blank row and inversions.
    """
    inversion_count = 0
    state_wo_blank = [tile for tile in state if tile != 0]  # Remove the blank tile
    for i in range(len(state_wo_blank)):
        for j in range(i + 1, len(state_wo_blank)):
            if state_wo_blank[i] > state_wo_blank[j]:
                inversion_count += 1

    if grid_size % 2 == 1:
        # Odd grid size: solvable if inversion count is even
        return inversion_count % 2 == 0
    else:
        # Even grid size: solvable based on blank row and inversions
        blank_pos = state.index(0)
        blank_row_from_bottom = grid_size - (blank_pos // grid_size)
        if blank_row_from_bottom % 2 == 0:
            return inversion_count % 2 == 1
        else:
            return inversion_count % 2 == 0

def generate_random_puzzle_state(num_moves=100, grid_size=5):
    """
    Generate a random puzzle state by applying random moves to the goal state.
    This ensures that the puzzle is solvable.
    """
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    state = list(goal_state)
    blank_pos = state.index(0)
    for _ in range(num_moves):
        moves = get_possible_moves(blank_pos, grid_size)
        move = random.choice(moves)
        # Swap the blank tile with the tile in the new position
        state[blank_pos], state[move] = state[move], state[blank_pos]
        blank_pos = move
    return tuple(state)

def astar(start_state, goal_state, grid_size=5):
    """
    A* search algorithm for the N-puzzle using Manhattan distance with linear conflicts.
    Returns a tuple of (g_values, num_expanded_states).
    """
    start_node = Node(start_state, g=0, h=manhattan_distance_with_linear_conflicts(start_state, goal_state, grid_size))
    open_list = [start_node]
    closed_set = set()
    g_values = {start_state: 0}
    num_expanded = 0  # Initialize counter for expanded states

    while open_list:
        current = heapq.heappop(open_list)
        num_expanded += 1  # Increment for each state popped from open_list
        if num_expanded >= 100000:  # Prevent excessive expansions
            return None, num_expanded

        if current.state == goal_state:
            return g_values, num_expanded

        if current.state in closed_set:
            continue

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            new_g = current.g + cost
            if neighbor_state in g_values and new_g >= g_values[neighbor_state]:
                continue

            g_values[neighbor_state] = new_g
            h = manhattan_distance_with_linear_conflicts(neighbor_state, goal_state, grid_size)
            neighbor_node = Node(neighbor_state, g=new_g, h=h, parent=current)
            heapq.heappush(open_list, neighbor_node)

    return None, num_expanded  # No solution found

def generate_start_goal(grid_size=5):
    """
    Generates a random solvable start and goal state for the puzzle.
    Ensures that the start_state is different from the goal_state and solvable.
    """
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    while True:
        start_state = generate_random_puzzle_state(num_moves=random.randint(100,130), grid_size=grid_size)
        if start_state != goal_state and is_solvable(start_state, grid_size):
            return start_state, goal_state

# ---------------------------
# Single Puzzle Processing Function
# ---------------------------

def process_single_puzzle(args):
    (puzzle_idx, grid_size, puzzle_save_dir, track_expansions, max_samples) = args
    try:
        # Generate start and goal states
        start_state, goal_state = generate_start_goal(grid_size=grid_size)

        # Save the puzzle data as .npz
        puzzle_file_path = os.path.join(puzzle_save_dir, f"puzzle_{grid_size}x{grid_size}_{puzzle_idx+1}.npz")
        np.savez(puzzle_file_path, start_state=start_state, goal_state=goal_state)
        logger.info(f"Puzzle {puzzle_idx + 1}: Start and goal states saved.")

        # Encode the start and goal states
        # Here, encoding is simply flattening the state. Replace with actual encoding if needed.
        encoded_start = np.array(start_state, dtype=np.float32)
        encoded_goal = np.array(goal_state, dtype=np.float32)

        # Run A* search from start to goal and from goal to start
        forward_g_values, forward_num_expanded = astar(start_state, goal_state, grid_size=grid_size)
        backward_g_values, backward_num_expanded = astar(goal_state, start_state, grid_size=grid_size)

        if forward_g_values is None or backward_g_values is None:
            logger.warning(f"Puzzle {puzzle_idx + 1}: A* did not find a solution.")
            return None

        # Combine forward and backward g-values to compute f*
        f_star_values = {}
        for state in forward_g_values:
            if state in backward_g_values:
                f_star = forward_g_values[state] + backward_g_values[state]
                f_star_values[state] = f_star

        # Prepare different versions of f*
        # Vanilla: No penalty
        vanilla_f_star_values = f_star_values.copy()

        # Exponential penalty
        exp_f_star_values = {}
        c_star = forward_g_values[goal_state]  # Optimal path cost
        if c_star == 0:
            logger.warning(f"Puzzle {puzzle_idx + 1}: c_star is zero, skipping penalties.")
            return None

        for state, f_star in f_star_values.items():
            penalty = np.exp((f_star - c_star) / c_star)
            exp_f_star_values[state] = f_star * penalty

        # Multiplicative penalty
        mult_f_star_values = {}
        for state, f_star in f_star_values.items():
            penalty = 1 + ((f_star - c_star) / c_star)
            mult_f_star_values[state] = f_star * penalty

        # Dictionary to iterate through each dataset type
        f_star_versions = {
            'vanilla': vanilla_f_star_values,
            'exp': exp_f_star_values,
            'mult': mult_f_star_values
        }

        # Prepare data to return
        puzzle_data = {
            'vanilla': [],
            'exp': [],
            'mult': []
        }
        all_g_values = {
            'vanilla': [],
            'exp': [],
            'mult': []
        }
        all_h_values = {
            'vanilla': [],
            'exp': [],
            'mult': []
        }
        all_f_star_values = {
            'vanilla': [],
            'exp': [],
            'mult': []
        }

        for dataset_type, modified_f_star_values in f_star_versions.items():
            sample_count = 0
            for state, f_star_value in modified_f_star_values.items():
                if state == goal_state:
                    continue  # Skip the goal state to avoid c_star=0

                g_star = forward_g_values[state]
                h = manhattan_distance_with_linear_conflicts(state, goal_state, grid_size=grid_size)

                # Collect data
                puzzle_data[dataset_type].append((encoded_start, encoded_goal, state, g_star, h, f_star_value))
                all_g_values[dataset_type].append(g_star)
                all_h_values[dataset_type].append(h)
                all_f_star_values[dataset_type].append(f_star_value)

                sample_count += 1
                if sample_count >= max_samples:
                    break  # Limit samples per query

            logger.info(f"Puzzle {puzzle_idx + 1}: Collected {sample_count} samples for '{dataset_type}' dataset.")

        # Optionally, collect the number of states expanded
        total_expanded = forward_num_expanded + backward_num_expanded

        # Print the number of states expanded for this puzzle if tracking is enabled
        if track_expansions:
            logger.info(f"Puzzle {puzzle_idx + 1}: {total_expanded} states expanded.")

        return puzzle_data, all_g_values, all_h_values, all_f_star_values

    except Exception as e:
        logger.error(f"Puzzle {puzzle_idx + 1}: Error during processing: {e}")
        return None

# ---------------------------
# Data Generation Function
# ---------------------------

def generate_dataset(num_puzzles, puzzle_save_dir="puzzles", grid_size=5, track_expansions=False, max_samples=5000):
    # Initialize separate datasets
    datasets = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    all_g_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    all_h_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }
    all_f_star_values = {
        'vanilla': [],
        'exp': [],
        'mult': []
    }

    if not os.path.exists(puzzle_save_dir):
        os.makedirs(puzzle_save_dir)
        logger.info(f"Created puzzle save directory: {puzzle_save_dir}")

    # Prepare arguments for multiprocessing
    args_list = []
    for puzzle_idx in range(num_puzzles):
        args_list.append((puzzle_idx, grid_size, puzzle_save_dir, track_expansions, max_samples))

    # Initialize multiprocessing pool
    pool_size = min(40, cpu_count())
    with Pool(processes=pool_size) as pool:
        try:
            logger.info(f"Starting multiprocessing pool with {pool_size} processes.")
            results = list(tqdm(pool.imap_unordered(process_single_puzzle, args_list), total=num_puzzles, desc="Processing Puzzles"))
            logger.info("Multiprocessing pool completed.")
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Terminating workers.")
            pool.terminate()
            pool.join()
            sys.exit(1)
        except Exception as e:
            logger.error(f"An exception occurred during multiprocessing: {e}")
            pool.terminate()
            pool.join()
            sys.exit(1)

    # Now, combine the results
    for result in results:
        if result is None:
            continue  # Skip puzzles that were skipped
        puzzle_data, g_values, h_values, f_star_vals = result
        for dataset_type in ['vanilla', 'exp', 'mult']:
            datasets[dataset_type].extend(puzzle_data[dataset_type])
            all_g_values[dataset_type].extend(g_values[dataset_type])
            all_h_values[dataset_type].extend(h_values[dataset_type])
            all_f_star_values[dataset_type].extend(f_star_vals[dataset_type])

    normalized_datasets = {}
    normalization_values = {}
    for dataset_type in ['vanilla', 'exp', 'mult']:
        if not all_g_values[dataset_type]:  # Avoid empty datasets
            logger.warning(f"No data collected for '{dataset_type}' dataset. Skipping normalization.")
            continue

        logger.info(f"Normalizing '{dataset_type}' dataset...")
        g_min, g_max = np.min(all_g_values[dataset_type]), np.max(all_g_values[dataset_type])
        h_min, h_max = np.min(all_h_values[dataset_type]), np.max(all_h_values[dataset_type])
        f_star_min, f_star_max = np.min(all_f_star_values[dataset_type]), np.max(all_f_star_values[dataset_type])

        normalization_values[dataset_type] = {
            'f_star_min': f_star_min, 'f_star_max': f_star_max,
            'g_min': g_min, 'g_max': g_max,
            'h_min': h_min, 'h_max': h_max
        }

    for dataset_type in ['vanilla', 'exp', 'mult']:
        dataset = datasets.get(dataset_type, [])
        norm_values = normalization_values.get(dataset_type, {})

        if not dataset or not norm_values:
            continue

        f_star_min, f_star_max = norm_values['f_star_min'], norm_values['f_star_max']
        g_min, g_max = norm_values['g_min'], norm_values['g_max']
        h_min, h_max = norm_values['h_min'], norm_values['h_max']

        normalized_dataset = []
        for encoded_start, encoded_goal, state, g_star, h, f_star_value in dataset:
            g_normalized = (g_star - g_min) / (g_max - g_min + 1e-8)
            h_normalized = (h - h_min) / (h_max - h_min + 1e-8)
            target_normalized = (f_star_value - f_star_min) / (f_star_max - f_star_min + 1e-8)

            if np.isfinite(g_normalized) and np.isfinite(h_normalized) and np.isfinite(target_normalized):
                normalized_dataset.append(
                    (encoded_start, encoded_goal, state, g_normalized, h_normalized, target_normalized)
                )

        normalized_datasets[dataset_type] = normalized_dataset
        logger.info(f"Normalized '{dataset_type}' dataset with {len(normalized_dataset)} samples.")

    return normalized_datasets, normalization_values

# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create Datasets for F* Prediction Model for N-Puzzle with Multiprocessing and Sample Limiting"
    )
    parser.add_argument("--num_puzzles", type=int, default=100,
                        help="Number of puzzles to generate")
    parser.add_argument("--grid_size", type=int, default=5,
                        help="Size of the puzzle grid (e.g., 5 for 5x5 puzzle)")
    parser.add_argument("--puzzle_save_dir", type=str, default="puzzles",
                        help="Directory to save the generated puzzles")
    parser.add_argument("--save_dataset_dir", type=str, default="datasets",
                        help="Directory to save the generated datasets")
    parser.add_argument("--norm_save_dir", type=str, default="normalization_values",
                        help="Directory to save the normalization values")
    parser.add_argument("--track_expansions", action='store_true',
                        help="Enable tracking and logging of state expansions per puzzle")
    parser.add_argument("--max_samples_per_puzzle", type=int, default=5000,
                        help="Maximum number of samples to collect per puzzle")
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------

def main():
    args = parse_arguments()

    # Create necessary directories
    directories = [
        args.puzzle_save_dir,
        args.save_dataset_dir,
        args.norm_save_dir
    ]
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

    logger.info(f"Generating {args.num_puzzles} puzzles of size {args.grid_size}x{args.grid_size} and calculating normalization values")

    dataset_dict, normalization_values_dict = generate_dataset(
        num_puzzles=args.num_puzzles,
        puzzle_save_dir=args.puzzle_save_dir,
        grid_size=args.grid_size,
        track_expansions=args.track_expansions,
        max_samples=args.max_samples_per_puzzle
    )

    # Include grid size in the filenames
    grid_size_str = f"{args.grid_size}x{args.grid_size}"

    # Save each dataset and its normalization values separately
    for dataset_type in ['vanilla', 'exp', 'mult']:
        dataset = dataset_dict.get(dataset_type, [])
        normalization_values = normalization_values_dict.get(dataset_type, {})

        if not dataset:
            logger.warning(f"No data to save for '{dataset_type}' dataset.")
            continue

        # Save dataset with grid size in filename
        dataset_save_path = os.path.join(args.save_dataset_dir, f"{dataset_type}_dataset_{grid_size_str}.pkl")
        try:
            with open(dataset_save_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"'{dataset_type.capitalize()}' dataset saved to {dataset_save_path}")
        except Exception as e:
            logger.error(f"Failed to save '{dataset_type}' dataset: {e}")
            continue

        # Save normalization values with grid size in filename
        norm_save_path = os.path.join(args.norm_save_dir, f"{dataset_type}_normalization_values_{grid_size_str}.pkl")
        try:
            with open(norm_save_path, 'wb') as f:
                pickle.dump(normalization_values, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"'{dataset_type.capitalize()}' normalization values saved to {norm_save_path}")
        except Exception as e:
            logger.error(f"Failed to save '{dataset_type}' normalization values: {e}")
            continue

    logger.info("All datasets created and saved successfully.")

if __name__ == '__main__':
    main()
