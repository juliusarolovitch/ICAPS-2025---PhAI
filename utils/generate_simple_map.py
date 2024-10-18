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

def generate_map(map_size=128, num_obstacles=20, obstacle_size=15):
    """
    Generates a 2D map with specified obstacles.

    Args:
        map_size (int): Size of the map (map_size x map_size).
        num_obstacles (int): Number of square obstacles.
        obstacle_size (int): Size of each square obstacle.

    Returns:
        np.ndarray: 2D occupancy map where 1 represents free space and 0 represents obstacles.
    """
    # Initialize map with free space
    map_grid = np.ones((map_size, map_size), dtype=np.float32)

    obstacles = []

    attempts = 0
    max_attempts = 1000  # Prevent infinite loop

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        # Random top-left corner for the obstacle
        x = random.randint(0, map_size - obstacle_size)
        y = random.randint(0, map_size - obstacle_size)

        new_obstacle = (x, y, obstacle_size, obstacle_size)  # (x, y, width, height)

        # Check for overlap
        overlap = False
        for obs in obstacles:
            if rectangles_overlap(new_obstacle, obs):
                overlap = True
                break

        if not overlap:
            obstacles.append(new_obstacle)
            # Place obstacle on the map
            map_grid[y:y+obstacle_size, x:x+obstacle_size] = 0  # 0 for obstacle

    if len(obstacles) < num_obstacles:
        raise ValueError("Could not place all obstacles without overlap. Try reducing the number or size of obstacles.")

    return map_grid

def rectangles_overlap(rect1, rect2):
    """
    Checks if two rectangles overlap.

    Args:
        rect1 (tuple): (x1, y1, w1, h1)
        rect2 (tuple): (x2, y2, w2, h2)

    Returns:
        bool: True if rectangles overlap, False otherwise.
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if (x1 + w1 <= x2) or (x2 + w2 <= x1):
        return False
    if (y1 + h1 <= y2) or (y2 + h2 <= y1):
        return False
    return True

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
    if 0 <= x < map_grid.shape[1] and 0 <= y < map_grid.shape[0]:
        return map_grid[y, x] == 1
    return False

def generate_start_goal(map_grid):
    """
    Generates random start and goal positions that are free and distinct.

    Args:
        map_grid (np.ndarray): 2D occupancy map.

    Returns:
        tuple: (start, goal) positions as (x, y)
    """
    map_size = map_grid.shape[0]
    while True:
        start = (random.randint(0, map_size - 1), random.randint(0, map_size - 1))
        goal = (random.randint(0, map_size - 1), random.randint(0, map_size - 1))
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal:
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
            # Reconstruct path
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
        ((x-1, y), 1),
        ((x+1, y), 1),
        ((x, y-1), 1),
        ((x, y+1), 1),
        ((x-1, y-1), np.sqrt(2)),
        ((x-1, y+1), np.sqrt(2)),
        ((x+1, y-1), np.sqrt(2)),
        ((x+1, y+1), np.sqrt(2))
    ]
    valid_neighbors = []
    for neighbor, cost in neighbors:
        if is_valid(neighbor, map_grid):
            valid_neighbors.append((neighbor, cost))
    return valid_neighbors

def plot_trajectory(map_grid, path, save_path="trajectory.png"):
    """
    Plots the map and the trajectory with thicker dark blue lines and adds circles at start and goal.

    Args:
        map_grid (np.ndarray): 2D occupancy map.
        path (list): List of (x, y) tuples representing the path.
        save_path (str): Path to save the image.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(map_grid, cmap='gray', origin='upper')

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
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Trajectory image saved as {save_path}")

def main():
    while True:
        # Generate a 100x100 map with 6 randomly placed 20x20 obstacles
        try:
            map_grid = generate_map()
        except ValueError as e:
            print(e)
            return

        # Generate start and goal positions
        start, goal = generate_start_goal(map_grid)

        # Run A* to find the trajectory
        path = astar(start, goal, map_grid)

        if path is None:
            print("No path found. Regenerating map...")
            continue  # Try again

        # Calculate path length
        path_length = sum(euclidean_distance(path[i-1], path[i]) for i in range(1, len(path)))

        print(f"Path length: {path_length:.2f}")

        if path_length >= 130:
            print("Valid path found.")
            break
        else:
            print("Path too short. Regenerating...")

    # Plot and save the trajectory
    plot_trajectory(map_grid, path, save_path="trajectory.png")

if __name__ == "__main__":
    main()
