import numpy as np
import torch
import random
import argparse
import os
import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import sys

# ---------------------------
# Additional Imports for Rubik's Cube
# ---------------------------
from collections import deque
from copy import deepcopy

# Import Rubik's Cube libraries
import pycuber as pc  # For cube representation and manipulation
import kociemba  # For solving the cube

# ---------------------------
# Logging Configuration
# ---------------------------

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Uncomment the following line to also log to a file
        # logging.FileHandler("rubiks_cube_data_generation.log")
    ]
)

logger = logging.getLogger(__name__)

# ---------------------------
# Utility Functions for Rubik's Cube
# ---------------------------

def generate_random_cube_state(scramble_length=20):
    """
    Generates a random cube state by applying a random scramble.
    """
    moves = [
        'U', "U'", 'U2', 'D', "D'", 'D2',
        'F', "F'", 'F2', 'B', "B'", 'B2',
        'R', "R'", 'R2', 'L', "L'", 'L2'
    ]
    scramble = [random.choice(moves) for _ in range(scramble_length)]
    scramble_sequence = ' '.join(scramble)
    cube_state = pc.Cube()
    try:
        cube_state(scramble_sequence)  # Correct usage: positional argument
    except Exception as e:
        logger.error(f"Error applying scramble sequence '{scramble_sequence}': {e}")
        return None, None
    return cube_state, scramble_sequence

def get_inverse_scramble(scramble):
    """
    Returns the inverse of a scramble sequence.
    """
    inverse_moves = {
        'U': "U'", "U'": 'U', 'U2': 'U2',
        'D': "D'", "D'": 'D', 'D2': 'D2',
        'F': "F'", "F'": 'F', 'F2': 'F2',
        'B': "B'", "B'": 'B', 'B2': 'B2',
        'R': "R'", "R'": 'R', 'R2': 'R2',
        'L': "L'", "L'": 'L', 'L2': 'L2'
    }
    inverse_scramble = [inverse_moves[move] for move in reversed(scramble.split())]
    return ' '.join(inverse_scramble)

def cube_state_to_string(cube_state):
    """
    Converts a cube state to a string representation compatible with Kociemba's solver.
    Kociemba expects a 54-character string representing the cube's facelets in the following order:
    U (Up), R (Right), F (Front), D (Down), L (Left), B (Back)
    Each face is represented in row-major order (top-left to bottom-right).
    """
    # Define the order of faces for Kociemba
    face_order = ['U', 'R', 'F', 'D', 'L', 'B']
    
    # Define the color mapping from PyCuber to Kociemba
    # PyCuber uses single letters in brackets: [w], [r], [g], [y], [o], [b]
    color_map = {
        '[w]': 'U',  # white -> Up
        '[r]': 'R',  # red -> Right
        '[g]': 'F',  # green -> Front
        '[y]': 'D',  # yellow -> Down
        '[o]': 'L',  # orange -> Left
        '[b]': 'B'   # blue -> Back
    }
    
    # Initialize a list to hold all facelet letters
    facelet_letters = []
    for face in face_order:
        face_obj = cube_state.get_face(face)
        for row in face_obj:
            for cubie in row:
                # Convert Square object to string - it will be in format '[color]'
                color = str(cubie).lower()
                if color not in color_map:
                    logger.error(f"Unknown color '{color}' on face '{face}'.")
                    return None
                facelet_letters.append(color_map[color])
    
    state_str = ''.join(facelet_letters)
    
    if len(state_str) != 54:
        logger.error(f"Invalid cube string length: {len(state_str)} instead of 54.")
        return None
    
    return state_str

def test_cube_mapping():
    """
    Tests the cube_state_to_string function with a solved cube.
    """
    solved_cube = pc.Cube()
    solved_str = cube_state_to_string(solved_cube)
    expected_str = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    if solved_str == expected_str:
        logger.info("Test Passed: Solved cube string matches expected Kociemba format.")
    else:
        logger.error("Test Failed: Solved cube string does not match expected Kociemba format.")
        logger.error(f"Expected: {expected_str}")
        logger.error(f"Got:      {solved_str}")
        
def validate_cube_string(state_str):
    """
    Validates the cube string to ensure each color appears exactly 9 times.
    """
    from collections import Counter
    if state_str is None:
        return False
    if len(state_str) != 54:
        logger.error(f"Cube string length is {len(state_str)}, expected 54.")
        return False
    count = Counter(state_str)
    expected_colors = ['U', 'R', 'F', 'D', 'L', 'B']
    for color in expected_colors:
        if count[color] != 9:
            logger.error(f"Color '{color}' appears {count[color]} times instead of 9.")
            return False
    return True

def solve_cube(cube_state):
    """
    Solves the cube using Kociemba's algorithm.
    Returns the solution moves and the number of moves (g value).
    """
    state_str = cube_state_to_string(cube_state)
    if state_str is None:
        return None, None
    try:
        solution = kociemba.solve(state_str)
        solution_moves = solution.split()
        g_value = len(solution_moves)
        return solution_moves, g_value
    except Exception as e:
        logger.error(f"Error solving cube: {e}")
        return None, None

def test_cube_mapping():
    """
    Tests the cube_state_to_string function with a solved cube.
    """
    solved_cube = pc.Cube()
    solved_str = cube_state_to_string(solved_cube)
    expected_str = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    if solved_str == expected_str:
        logger.info("Test Passed: Solved cube string matches expected Kociemba format.")
    else:
        logger.error("Test Failed: Solved cube string does not match expected Kociemba format.")
        logger.error(f"Expected: {expected_str}")
        logger.error(f"Got:      {solved_str}")
def apply_move_sequence(cube_state, moves):
    """
    Applies a sequence of moves to a cube state and returns the new state.
    """
    new_state = cube_state.copy()
    try:
        new_state(' '.join(moves))
        return new_state
    except Exception as e:
        logger.error(f"Error applying moves: {e}")
        return None
def get_intermediate_states(cube_state, solution_moves):
    """
    Generates intermediate states and their g-values from a solution sequence.
    Returns a list of (state_encoding, g_value) tuples.
    """
    states = []
    current_state = cube_state.copy()
    
    # Add initial state
    encoded_state = cube_state_to_string(current_state)
    if encoded_state:
        states.append((encoded_state, 0))
    
    # Generate each intermediate state
    current_moves = []
    for move in solution_moves:
        current_moves.append(move)
        next_state = current_state.copy()
        try:
            next_state(' '.join([move]))  # Apply just the current move
            current_state = next_state  # Update current state
            encoded_state = cube_state_to_string(next_state)
            if encoded_state:
                states.append((encoded_state, len(current_moves)))
        except Exception as e:
            logger.error(f"Error applying move {move}: {e}")
            continue
    
    return states

def process_single_cube(args):
    (cube_idx, puzzle_save_dir, max_samples) = args
    try:
        # Generate a random cube state with a reasonable scramble length
        scramble_length = random.randint(10, 20)
        scrambled_cube, scramble_sequence = generate_random_cube_state(scramble_length=scramble_length)
        
        if scrambled_cube is None:
            logger.warning(f"Cube {cube_idx + 1}: Scramble generation failed.")
            return None

        # Save the scramble sequence
        puzzle_file_path = os.path.join(puzzle_save_dir, f"cube_{cube_idx+1}.pkl")
        with open(puzzle_file_path, 'wb') as f:
            pickle.dump({'scramble': scramble_sequence}, f)
        logger.info(f"Cube {cube_idx + 1}: Scramble sequence saved.")

        # Get optimal solution from scrambled to solved (forward)
        forward_solution_moves, _ = solve_cube(scrambled_cube)
        if forward_solution_moves is None:
            logger.warning(f"Cube {cube_idx + 1}: Could not solve from start to goal.")
            return None
            
        # Get intermediate states from scrambled to solved
        forward_states = get_intermediate_states(scrambled_cube, forward_solution_moves)
        
        # Get optimal solution from solved to scrambled (backward)
        inverse_scramble = get_inverse_scramble(scramble_sequence)
        scrambled_from_solved = pc.Cube()
        try:
            scrambled_from_solved(inverse_scramble)
        except Exception as e:
            logger.error(f"Cube {cube_idx + 1}: Error applying inverse scramble: {e}")
            return None
            
        backward_solution_moves, _ = solve_cube(scrambled_from_solved)
        if backward_solution_moves is None:
            logger.warning(f"Cube {cube_idx + 1}: Could not solve from goal to start.")
            return None
            
        # Get intermediate states from solved to scrambled
        backward_states = get_intermediate_states(scrambled_from_solved, backward_solution_moves)

        # Create a mapping of encoded states to their backward g-values
        backward_g_values = {state: g for state, g in backward_states}

        # Prepare dataset entries for all intermediate states
        dataset_entries = []
        total_backward_moves = len(backward_solution_moves)
        
        for state, forward_g in forward_states:
            entry = {
                'cube_idx': cube_idx + 1,
                'state': state,
                'forward_g_value': forward_g,
                'backward_g_value': total_backward_moves - backward_g_values.get(state, 0)
            }
            dataset_entries.append(entry)

        logger.info(f"Cube {cube_idx + 1}: Generated {len(dataset_entries)} intermediate states.")
        return dataset_entries

    except Exception as e:
        logger.error(f"Cube {cube_idx + 1}: Error during processing: {e}")
        return None

def generate_dataset(num_cubes, puzzle_save_dir="cubes", max_samples=1000):
    all_entries = []

    if not os.path.exists(puzzle_save_dir):
        os.makedirs(puzzle_save_dir)
        logger.info(f"Created cube save directory: {puzzle_save_dir}")

    # Prepare arguments for multiprocessing
    args_list = [(cube_idx, puzzle_save_dir, max_samples) for cube_idx in range(num_cubes)]

    # Initialize multiprocessing pool
    pool_size = min(40, cpu_count())
    with Pool(processes=pool_size) as pool:
        try:
            logger.info(f"Starting multiprocessing pool with {pool_size} processes.")
            results = list(tqdm(pool.imap_unordered(process_single_cube, args_list), 
                              total=num_cubes, desc="Processing Cubes"))
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

    # Collect all entries from all cubes
    for result in results:
        if result is not None:
            all_entries.extend(result)

    logger.info(f"Collected {len(all_entries)} total intermediate states.")
    return all_entries

# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create Datasets for Rubik's Cube F* Prediction Model"
    )
    parser.add_argument("--num_cubes", type=int, default=100,
                        help="Number of cubes to generate")
    parser.add_argument("--puzzle_save_dir", type=str, default="cubes",
                        help="Directory to save the generated cube scrambles")
    parser.add_argument("--save_dataset_dir", type=str, default="datasets",
                        help="Directory to save the generated datasets")
    parser.add_argument("--max_samples_per_cube", type=int, default=1000,
                        help="Maximum number of samples to collect per cube")
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------

def main():
    args = parse_arguments()

    # Create necessary directories
    directories = [
        args.puzzle_save_dir,
        args.save_dataset_dir
    ]
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

    logger.info(f"Generating {args.num_cubes} Rubik's Cubes and collecting data.")

    # Run the test to verify mapping
    test_cube_mapping()

    dataset = generate_dataset(
        num_cubes=args.num_cubes,
        puzzle_save_dir=args.puzzle_save_dir,
        max_samples=args.max_samples_per_cube
    )

    # Save the dataset
    dataset_save_path = os.path.join(args.save_dataset_dir, f"rubiks_cube_dataset.pkl")
    try:
        with open(dataset_save_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Dataset saved to {dataset_save_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

    logger.info("Dataset creation and saving completed.")

if __name__ == '__main__':
    main()
