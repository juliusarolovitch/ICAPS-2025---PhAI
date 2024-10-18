import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

# ---------------------------
# Node Class for Pathfinding
# ---------------------------

class Node:
    def __init__(self, pos, g=float('inf'), h=0, parent=None):
        self.pos = pos  # (x, y)
        self.g = g      # Cost from start to current node
        self.h = h      # Heuristic cost estimate to goal
        self.f = g + h  # Total estimated cost
        self.parent = parent  # Parent node in the path

    def __lt__(self, other):
        return self.f < other.f

# ---------------------------
# Utility Functions
# ---------------------------

def generate_empty_map(map_size=100):
    """
    Generates a 2D empty map.

    Args:
        map_size (int): Size of the map (map_size x map_size).

    Returns:
        np.ndarray: 2D occupancy map where 1 represents free space.
    """
    map_grid = np.ones((map_size, map_size), dtype=np.float32)
    return map_grid

def is_valid(pos, map_grid):
    """
    Checks if a position is valid (within bounds and not an obstacle).

    Args:
        pos (tuple): (x, y)
        map_grid (np.ndarray): 2D occupancy map.

    Returns:
        bool: True if valid, False otherwise.
    """
    x, y = pos
    return 0 <= x < map_grid.shape[1] and 0 <= y < map_grid.shape[0] and map_grid[y, x] == 1

def generate_start_goal(map_grid, min_distance=80):
    """
    Generates random start and goal positions that are free and sufficiently apart.

    Args:
        map_grid (np.ndarray): 2D occupancy map.
        min_distance (float): Minimum Euclidean distance between start and goal.

    Returns:
        tuple: (start, goal) positions as (x, y)
    """
    map_size = map_grid.shape[0]
    while True:
        start = (random.randint(0, map_size - 1), random.randint(0, map_size - 1))
        goal = (random.randint(0, map_size - 1), random.randint(0, map_size - 1))
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal:
            distance = euclidean_distance(start, goal)
            if distance >= min_distance:
                return start, goal

def euclidean_distance(a, b):
    """
    Computes Euclidean distance between two points.

    Args:
        a (tuple): (x1, y1)
        b (tuple): (x2, y2)

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(start, goal, map_grid):
    """
    Performs A* search to find the shortest path from start to goal.

    Args:
        start (tuple): (x, y) start position.
        goal (tuple): (x, y) goal position.
        map_grid (np.ndarray): 2D occupancy map.

    Returns:
        list: Path as a list of (x, y) tuples from start to goal, or None if no path found.
    """
    start_node = Node(start, g=0, h=euclidean_distance(start, goal))
    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()
    g_values = np.full(map_grid.shape, float('inf'))
    g_values[start[1], start[0]] = 0

    while open_list:
        current = heapq.heappop(open_list)

        if current.pos == goal:
            path = []
            while current is not None:
                path.append(current.pos)
                current = current.parent
            return path[::-1]  # Return reversed path

        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)

        for neighbor, cost in get_neighbors(current.pos, map_grid):
            if not is_valid(neighbor, map_grid):
                continue

            new_g = current.g + cost

            if new_g < g_values[neighbor[1], neighbor[0]]:
                g_values[neighbor[1], neighbor[0]] = new_g
                new_node = Node(neighbor, g=new_g, h=euclidean_distance(neighbor, goal), parent=current)
                heapq.heappush(open_list, new_node)

    return None  # No path found

def get_neighbors(pos, map_grid):
    """
    Retrieves valid neighboring positions and their movement costs.

    Args:
        pos (tuple): (x, y) current position.
        map_grid (np.ndarray): 2D occupancy map.

    Returns:
        list: List of tuples [(neighbor_pos, cost), ...]
    """
    x, y = pos
    neighbors = [
        ((x-1, y), 1), ((x+1, y), 1), ((x, y-1), 1), ((x, y+1), 1),
        ((x-1, y-1), np.sqrt(2)), ((x-1, y+1), np.sqrt(2)),
        ((x+1, y-1), np.sqrt(2)), ((x+1, y+1), np.sqrt(2))
    ]
    return [(neighbor, cost) for neighbor, cost in neighbors if is_valid(neighbor, map_grid)]

def plot_trajectory(map_grid, path, save_path="trajectory.png"):
    """
    Plots the map and the trajectory with thicker dark blue lines and adds circles at start and goal.

    Args:
        map_grid (np.ndarray): 2D occupancy map.
        path (list): List of (x, y) tuples representing the path.
        save_path (str): Path to save the image.
    """
    plt.figure(figsize=(8, 8))
    
    # Display the map with white background
    plt.imshow(map_grid, cmap='binary', origin='upper')

    if not path:
        print("Empty path. Nothing to plot.")
        return

    # Extract x and y coordinates from the path
    x_coords, y_coords = zip(*path)

    # Plot the trajectory in dark blue with a thicker line
    plt.plot(x_coords, y_coords, color='darkblue', linewidth=3)

    # Add small dark blue circles at start and goal
    plt.scatter(x_coords[0], y_coords[0], color='darkblue', s=100, zorder=5)  # Start
    plt.scatter(x_coords[-1], y_coords[-1], color='darkblue', s=100, zorder=5)  # Goal

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Trajectory image saved as {save_path}")

def main():
    while True:
        # Generate a 100x100 empty map
        map_grid = generate_empty_map(map_size=100)

        # Generate start and goal positions with at least 80 units apart
        start, goal = generate_start_goal(map_grid, min_distance=80)

        # Run A* to find the trajectory
        path = astar(start, goal, map_grid)

        if path is None:
            print("No path found. This should not happen in an empty map. Regenerating...")
            continue  # Try again

        # Calculate path length
        path_length = sum(euclidean_distance(path[i-1], path[i]) for i in range(1, len(path)))

        print(f"Path length: {path_length:.2f}")

        if path_length >= 80:
            print("Valid path found.")
            break
        else:
            print("Path too short. Regenerating...")

    # Plot and save the trajectory
    plot_trajectory(map_grid, path, save_path="trajectory.png")

if __name__ == "__main__":
    main()
