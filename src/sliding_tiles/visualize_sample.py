import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# Node Class for Sliding Tile Puzzle
# ---------------------------

class Node:
    def __init__(self, state, g=float('inf'), h=0, f=float('inf')):
        self.state = state  # State is a tuple representing the puzzle configuration
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost estimate to goal
        self.f = f  # Total estimated cost

    def __lt__(self, other):
        return self.f < other.f

# ---------------------------
# Utility Functions
# ---------------------------

def get_possible_moves(blank_pos, grid_size=4):
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

def get_neighbors(state, grid_size=4):
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

def manhattan_distance(state, goal_state, grid_size=4):
    """
    Calculates the sum of the Manhattan distances of the tiles from their goal positions.
    """
    distance = 0
    for i in range(1, grid_size * grid_size):  # Tiles numbered from 1 to N-1
        idx_current = state.index(i)
        idx_goal = goal_state.index(i)
        x_current, y_current = idx_current % grid_size, idx_current // grid_size
        x_goal, y_goal = idx_goal % grid_size, idx_goal // grid_size
        distance += abs(x_current - x_goal) + abs(y_current - y_goal)
    return distance

def is_solvable(state, grid_size=4):
    """
    Checks if a given sliding tile puzzle state is solvable.
    For odd grid sizes (like 3x3), the puzzle is solvable if the number of inversions is even.
    For even grid sizes, the solvability depends on the position of the blank tile.
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
        blank_row = blank_pos // grid_size
        blank_row_from_bottom = grid_size - blank_row
        if blank_row_from_bottom % 2 == 0:
            return inversion_count % 2 == 1
        else:
            return inversion_count % 2 == 0

def generate_random_puzzle_state(num_moves=50, grid_size=4):
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

def astar(start_state, goal_state, grid_size=4):
    """
    A* search algorithm for the sliding tile puzzle.
    Returns a tuple of (path, num_expanded_states).
    """
    start_node = Node(start_state, g=0, h=manhattan_distance(start_state, goal_state, grid_size))
    start_node.f = start_node.g + start_node.h
    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()
    num_expanded = 0  # Initialize counter for expanded states

    came_from = {}  # For reconstructing the path
    g_score = {start_state: 0}

    while open_list:
        current = heapq.heappop(open_list)
        num_expanded += 1

        if current.state == goal_state:
            # Reconstruct the path
            path = []
            state = current.state
            while state in came_from:
                path.append(state)
                state = came_from[state]
            path.append(start_state)
            path.reverse()
            return path, num_expanded

        if current.state in closed_set:
            continue

        closed_set.add(current.state)

        for neighbor_state, cost in get_neighbors(current.state, grid_size):
            if neighbor_state in closed_set:
                continue

            tentative_g_score = g_score[current.state] + cost

            if tentative_g_score < g_score.get(neighbor_state, float('inf')):
                came_from[neighbor_state] = current.state
                g_score[neighbor_state] = tentative_g_score
                h = manhattan_distance(neighbor_state, goal_state, grid_size)
                f = tentative_g_score + h
                neighbor_node = Node(neighbor_state, g=tentative_g_score, h=h, f=f)
                heapq.heappush(open_list, neighbor_node)

    return None, num_expanded  # No solution found

def plot_puzzle_state(state, grid_size=4, ax=None):
    """
    Display the puzzle state as a grid.
    """
    puzzle = np.array(state).reshape((grid_size, grid_size))
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = None

    ax.imshow(np.zeros_like(puzzle), cmap='gray', extent=[0, grid_size, grid_size, 0])

    for (i, j), val in np.ndenumerate(puzzle):
        if val != 0:
            ax.text(j + 0.5, i + 0.5, str(int(val)), va='center', ha='center', fontsize=16,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))
        else:
            ax.text(j + 0.5, i + 0.5, '', va='center', ha='center', fontsize=16)  # Blank tile

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sliding Tile Puzzle")

    if fig is not None:
        plt.show()

def animate_puzzle_solution(path, grid_size=4, gif_filename='solution.gif', interval=500):
    """
    Animate the puzzle solution given a path of states and save as a GIF.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    ax.set_title("Sliding Tile Puzzle Solution")

    # Initialize the plot with the first state
    puzzle = np.array(path[0]).reshape((grid_size, grid_size))
    im = ax.imshow(np.zeros_like(puzzle), cmap='gray', extent=[0, grid_size, grid_size, 0])

    texts = []
    for (i, j), val in np.ndenumerate(puzzle):
        if val != 0:
            text = ax.text(j + 0.5, i + 0.5, str(int(val)), va='center', ha='center', fontsize=16,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))
            texts.append(text)
        else:
            text = ax.text(j + 0.5, i + 0.5, '', va='center', ha='center', fontsize=16)  # Blank tile
            texts.append(text)

    def init():
        return texts

    def animate(k):
        puzzle = np.array(path[k]).reshape((grid_size, grid_size))
        for idx, (i, j) in enumerate(np.ndindex(puzzle.shape)):
            val = puzzle[i, j]
            if val != 0:
                texts[idx].set_text(str(int(val)))
            else:
                texts[idx].set_text('')
        return texts

    ani = animation.FuncAnimation(fig, animate, frames=len(path),
                                  init_func=init, interval=interval, blit=False, repeat=False)

    # Save the animation as a GIF
    ani.save(gif_filename, writer='pillow')
    plt.close(fig)  # Close the figure to prevent it from displaying in some environments
    print(f"Animation saved as {gif_filename}")

# ---------------------------
# Main Function
# ---------------------------

def main():
    grid_size = 4  # 4x4 grid for the 15-Puzzle
    # Generate start and goal states
    goal_state = tuple(range(1, grid_size * grid_size)) + (0,)
    start_state = generate_random_puzzle_state(num_moves=200, grid_size=grid_size)

    # Ensure the puzzle is solvable
    while not is_solvable(start_state, grid_size):
        start_state = generate_random_puzzle_state(num_moves=50, grid_size=grid_size)

    print("Start State:")
    plot_puzzle_state(start_state, grid_size)

    print("Solving the puzzle...")
    # Run A* search to find the solution path
    path, num_expanded = astar(start_state, goal_state, grid_size=grid_size)

    if path is None:
        print("No solution found.")
        return

    print("Here is the found path:")
    for i in path:
        print(i[0:4])
        print(i[4:8])
        print(i[8:12])
        print(i[12:16])
        print("\n\n")

    print(f"Solution found in {len(path) - 1} moves.")
    print(f"Number of states expanded: {num_expanded}")

    # Animate the solution and save as GIF
    gif_filename = 'solution.gif'
    animate_puzzle_solution(path, grid_size=grid_size, gif_filename=gif_filename, interval=500)

    print("Animation completed.")

if __name__ == '__main__':
    main()
