import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from dataclasses import dataclass, field

# ---------------------------
# Node Class for Pathfinding
# ---------------------------

class Node:
    def __init__(self, pos, g=float('inf'), h=0, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

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

    Returns:
        np.ndarray: 2D occupancy map of shape [height, width].
    """
    # Initialize map_grid as 1 (free space)
    map_grid = np.ones((height, width), dtype=np.float32)
    
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
            # Horizontal split
            split_min = room.y + min_room_size
            split_max = room.y + room.height - min_room_size - wall_thickness
            if split_max <= split_min:
                return
            split_pos = random.randint(split_min, split_max)
            # Create child rooms
            child1 = Room(room.x, room.y, room.width, split_pos - room.y)
            child2 = Room(room.x, split_pos + wall_thickness, room.width, room.y + room.height - split_pos - wall_thickness)
            # Add horizontal wall
            map_grid[split_pos:split_pos + wall_thickness, room.x:room.x + room.width] = 0
            # Add openings
            add_openings((split_pos, room.x), (split_pos, room.x + room.width), orientation='horizontal')
        else:
            # Vertical split
            split_min = room.x + min_room_size
            split_max = room.x + room.width - min_room_size - wall_thickness
            if split_max <= split_min:
                return
            split_pos = random.randint(split_min, split_max)
            # Create child rooms
            child1 = Room(room.x, room.y, split_pos - room.x, room.height)
            child2 = Room(split_pos + wall_thickness, room.y, room.x + room.width - split_pos - wall_thickness, room.height)
            # Add vertical wall
            map_grid[room.y:room.y + room.height, split_pos:split_pos + wall_thickness] = 0
            # Add openings
            add_openings((room.y, split_pos), (room.y + room.height, split_pos), orientation='vertical')
        
        room.children = [child1, child2]
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
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 1
        else:
            wall_length = end[0] - start[0]
            possible_positions = wall_length - opening_size
            if possible_positions <= 0:
                return
            for _ in range(num_openings):
                opening_start = random.randint(start[0], start[0] + possible_positions)
                map_grid[opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness] = 1
    
    split_room(root_room, 0)
    
    leaf_rooms = []
    def collect_leaf_rooms(room):
        if not room.children:
            leaf_rooms.append(room)
        else:
            for child in room.children:
                collect_leaf_rooms(child)
    
    collect_leaf_rooms(root_room)
    
    for room in leaf_rooms:
        num_obstacles = random.randint(min_obstacles, max_obstacles)
        for _ in range(num_obstacles):
            obstacle_w = random.randint(min_obstacle_size, max_obstacle_size)
            obstacle_h = random.randint(min_obstacle_size, max_obstacle_size)
            if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
                continue
            obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
            obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
            map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 0
    
    map_grid[0:wall_thickness, :] = 0
    map_grid[-wall_thickness:, :] = 0
    map_grid[:, 0:wall_thickness] = 0
    map_grid[:, -wall_thickness:] = 0
    
    return map_grid

def is_valid(pos, map_grid):
    return (0 <= pos[0] < map_grid.shape[0] and
            0 <= pos[1] < map_grid.shape[1] and
            map_grid[pos] == 1)

def generate_start_goal(map_grid):
    while True:
        start = (random.randint(0, map_grid.shape[0]-1), random.randint(0, map_grid.shape[1]-1))
        goal = (random.randint(0, map_grid.shape[0]-1), random.randint(0, map_grid.shape[1]-1))
        if is_valid(start, map_grid) and is_valid(goal, map_grid) and start != goal:
            return start, goal

def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(start, goal, map_grid):
    start_node = Node(start, g=0, h=euclidean_distance(start, goal))
    open_list = [start_node]
    closed_set = set()
    g_values = np.full(map_grid.shape, float('inf'))
    g_values[start] = 0

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

        for next_pos, cost in get_neighbors(current.pos, map_grid):
            if not is_valid(next_pos, map_grid):
                continue

            new_g = current.g + cost

            if new_g < g_values[next_pos]:
                g_values[next_pos] = new_g
                new_node = Node(next_pos, g=new_g, h=euclidean_distance(next_pos, goal), parent=current)
                heapq.heappush(open_list, new_node)

    return None

def get_neighbors(pos, map_grid):
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
    plt.figure(figsize=(8, 8))
    plt.imshow(map_grid, cmap='gray', origin='upper')

    # Extract x and y coordinates from the path
    y_coords, x_coords = zip(*path)

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
        map_grid = generate_map()

        start, goal = generate_start_goal(map_grid)

        path = astar(start, goal, map_grid)

        if path is None:
            print("No path found. Regenerating map...")
            continue

        path_length = sum(euclidean_distance(path[i-1], path[i]) for i in range(1, len(path)))

        print(f"Path length: {path_length:.2f}")

        if path_length >= 130:
            print("Valid path found.")
            break
        else:
            print("Path too short. Regenerating...")

    plot_trajectory(map_grid, path, save_path="trajectory.png")

if __name__ == "__main__":
    main()
