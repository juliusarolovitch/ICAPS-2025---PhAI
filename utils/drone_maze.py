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

def plot_3d_maze(maze, filename='3d_maze.png'):
    """
    Plots the 3D maze using matplotlib's voxel plotting with partially translucent walls and saves the plot.

    Args:
        maze (np.ndarray): 3D occupancy map.
        filename (str): Filename to save the plot.
    """
    # Identify wall voxels
    wall_voxels = maze == 1

    # Create a color array with RGBA (including alpha for transparency)
    # Initialize with all zeros
    colors = np.zeros(wall_voxels.shape + (4,), dtype=np.float32)
    # Set color with partial transparency (alpha=0.3) for walls
    colors[wall_voxels] = [0.5, 0.5, 0.5, 0.3]  # Gray color with 30% opacity

    # Create the figure and axis
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot voxels
    ax.voxels(wall_voxels, facecolors=colors, edgecolor='k', linewidth=0.05)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Room-Based Maze for Drone Navigation')

    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=45)

    # Improve aspect ratio
    ax.set_box_aspect([np.ptp(a) for a in [range(maze.shape[0]), range(maze.shape[1]), range(maze.shape[2])]])

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"3D maze plot saved as {filename}")

def main():
    # Define maze dimensions
    width, height, depth = 100, 100, 40  # x, y, z dimensions

    print("Generating 3D maze...")
    maze = generate_3d_maze(
        width=width,
        height=height,
        depth=depth,
        min_room_size=20,
        max_room_size=30,
        max_depth=5,
        wall_thickness=2,
        min_openings=1,
        max_openings=2,
        min_opening_size=5,
        max_opening_size=10,
        min_walls=2,  # Updated to 2-5 walls
        max_walls=5,  # Updated to 2-5 walls
        min_wall_length=20,  # Updated to 20-60
        max_wall_length=60,  # Updated to 20-60
        wall_attempts=10
    )

    print("Plotting and saving the maze...")
    plot_3d_maze(maze, filename='3d_maze.png')

if __name__ == "__main__":
    main()
