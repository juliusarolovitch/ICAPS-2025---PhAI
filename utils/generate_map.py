import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import random

@dataclass
class Room:
    x: int
    y: int
    width: int
    height: int
    children: list = field(default_factory=list)

def generate_random_map(
    width=512,  # Reduced from 1600 to 512
    height=512,  # Reduced from 1600 to 512
    min_room_size=60,  # Adjusted for smaller maps
    max_room_size=120,  # Adjusted for smaller maps
    max_depth=5,  # Reduced depth for smaller maps
    wall_thickness=5,
    min_openings=1,
    max_openings=2,
    min_opening_size=10,  # Adjusted for smaller maps
    max_opening_size=20,  # Adjusted for smaller maps
    min_obstacles=4,
    max_obstacles=20,
    min_obstacle_size=10,
    max_obstacle_size=30,
    obstacle_attempts=10,
    trap_probability=0.4
):
    """
    Generates a 2D map with rooms and walls with openings.
    Adds rectangular obstacles and concave traps without overlapping.

    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        min_room_size (int): Minimum size of a room.
        max_room_size (int): Maximum size of a room.
        max_depth (int): Maximum recursion depth for splitting rooms.
        wall_thickness (int): Thickness of the walls between rooms.
        min_openings (int): Minimum number of openings per wall.
        max_openings (int): Maximum number of openings per wall.
        min_opening_size (int): Minimum size of each opening in pixels.
        max_opening_size (int): Maximum size of each opening in pixels.
        min_obstacles (int): Minimum number of obstacles per room.
        max_obstacles (int): Maximum number of obstacles per room.
        min_obstacle_size (int): Minimum size (width/height) of each rectangular obstacle.
        max_obstacle_size (int): Maximum size (width/height) of each rectangular obstacle.
        obstacle_attempts (int): Number of attempts to place an obstacle without overlap.
        trap_probability (float): Probability of placing a concave trap instead of a regular obstacle.

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
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)  # Ensure opening size doesn't exceed wall
                opening_start = random.randint(start[1], end[1] - opening_size)
                map_grid[start[0]:start[0] + wall_thickness, opening_start:opening_start + opening_size] = 0
        else:
            wall_length = end[0] - start[0]
            if wall_length <= min_opening_size:
                return
            for _ in range(num_openings):
                opening_size = random.randint(min_opening_size, max_opening_size)
                opening_size = min(opening_size, wall_length)  # Ensure opening size doesn't exceed wall
                opening_start = random.randint(start[0], end[0] - opening_size)
                map_grid[opening_start:opening_start + opening_size, start[1]:start[1] + wall_thickness] = 0
    
    def place_concave_trap(room):
        """
        Places a concave trap within the given room. The trap can be L-shaped or triangular.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        trap_type = random.choice(['L', 'triangle'])
        if trap_type == 'L':
            return place_L_shaped_trap(room)
        else:
            return place_triangular_trap(room)
    
    def place_L_shaped_trap(room):
        """
        Places an L-shaped concave trap within the given room.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        trap_size = random.randint(min_obstacle_size, max_obstacle_size)
        trap_thickness = wall_thickness  # Thickness of the trap arms

        # Ensure the trap fits within the room
        if (trap_size * 2 + wall_thickness) > room.width or (trap_size * 2 + wall_thickness) > room.height:
            return False  # Trap too big for the room

        # Choose a position for the corner of the L-shape
        corner_x = random.randint(room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
        corner_y = random.randint(room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)

        # Randomly decide the orientation of the L-shape
        orientation = random.choice(['left', 'right', 'up', 'down'])

        if orientation == 'left':
            # Horizontal arm to the left, vertical arm upwards
            arm1 = ((corner_y, corner_x - trap_size), (trap_size, trap_thickness))
            arm2 = ((corner_y - trap_size, corner_x), (trap_thickness, trap_size))
        elif orientation == 'right':
            # Horizontal arm to the right, vertical arm upwards
            arm1 = ((corner_y, corner_x), (trap_size, trap_thickness))
            arm2 = ((corner_y - trap_size, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))
        elif orientation == 'up':
            # Vertical arm upwards, horizontal arm to the left
            arm1 = ((corner_y - trap_size, corner_x), (trap_thickness, trap_size))
            arm2 = ((corner_y - trap_size, corner_x - trap_size), (trap_size, trap_thickness))
        else:  # 'down'
            # Vertical arm downwards, horizontal arm to the right
            arm1 = ((corner_y, corner_x), (trap_thickness, trap_size))
            arm2 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_size, trap_thickness))
        
        # Check if arms are within bounds
        (y1, x1), (h1, w1) = arm1
        (y2, x2), (h2, w2) = arm2

        if (x1 < 0 or y1 < 0 or x1 + w1 > width or y1 + h1 > height or
            x2 < 0 or y2 < 0 or x2 + w2 > width or y2 + h2 > height):
            return False  # Out of bounds

        # Check for overlap with existing obstacles
        if (np.any(map_grid[y1:y1 + h1, x1:x1 + w1] == 1) or
            np.any(map_grid[y2:y2 + h2, x2:x2 + w2] == 1)):
            return False  # Overlaps with existing obstacle

        # Place the L-shaped trap
        map_grid[y1:y1 + h1, x1:x1 + w1] = 1
        map_grid[y2:y2 + h2, x2:x2 + w2] = 1

        return True  # Successfully placed

    def place_triangular_trap(room):
        """
        Places a triangular concave trap within the given room.

        Args:
            room (Room): The room where the trap will be placed.

        Returns:
            bool: True if the trap was successfully placed, False otherwise.
        """
        trap_size = random.randint(min_obstacle_size, max_obstacle_size)
        trap_thickness = wall_thickness  # Thickness of the trap lines

        # Define the three lines of the triangle (right-angled for simplicity)
        # Choose a position for the right angle
        corner_x = random.randint(room.x + wall_thickness, room.x + room.width - trap_size - wall_thickness)
        corner_y = random.randint(room.y + wall_thickness, room.y + room.height - trap_size - wall_thickness)

        # Randomly decide the orientation of the triangle
        orientation = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])

        if orientation == 'top-left':
            arm1 = ((corner_y, corner_x), (trap_thickness, trap_size))          # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))          # Horizontal
            arm3 = ((corner_y, corner_x), (trap_thickness, trap_thickness))    # Diagonal
        elif orientation == 'top-right':
            arm1 = ((corner_y, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))  # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))                              # Horizontal
            arm3 = ((corner_y, corner_x + trap_size - trap_thickness), (trap_thickness, trap_thickness))  # Diagonal
        elif orientation == 'bottom-left':
            arm1 = ((corner_y + trap_size - trap_thickness, corner_x), (trap_thickness, trap_size))  # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))                              # Horizontal
            arm3 = ((corner_y + trap_size - trap_thickness, corner_x), (trap_thickness, trap_thickness))  # Diagonal
        else:  # 'bottom-right'
            arm1 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_thickness, trap_size))  # Vertical
            arm2 = ((corner_y, corner_x), (trap_size, trap_thickness))                              # Horizontal
            arm3 = ((corner_y + trap_size - trap_thickness, corner_x + trap_size - trap_thickness), (trap_thickness, trap_thickness))  # Diagonal

        # Check if arms are within bounds
        arms = [arm1, arm2, arm3]
        for (y, x), (h, w) in arms:
            if (x < 0 or y < 0 or x + w > width or y + h > height):
                return False  # Out of bounds

        # Check for overlap with existing obstacles
        for (y, x), (h, w) in arms:
            if np.any(map_grid[y:y + h, x:x + w] == 1):
                return False  # Overlaps with existing obstacle

        # Place the triangular trap
        for (y, x), (h, w) in arms:
            map_grid[y:y + h, x:x + w] = 1

        return True  # Successfully placed

    # Start splitting from the root room
    split_room = locals()['split_room']  # To ensure recursive references work
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
            if random.random() < trap_probability:
                # Attempt to place a concave trap
                placed = False
                for attempt in range(obstacle_attempts):
                    if place_concave_trap(room):
                        placed = True
                        break  # Successfully placed
                if not placed:
                    print(f"Could not place a concave trap in room at ({room.x}, {room.y}) after {obstacle_attempts} attempts.")
                continue  # Move to next obstacle
            else:
                # Place a regular rectangular obstacle
                placed = False
                for attempt in range(obstacle_attempts):
                    obstacle_w = random.randint(min_obstacle_size, max_obstacle_size)
                    obstacle_h = random.randint(min_obstacle_size, max_obstacle_size)
                    # Ensure obstacle fits within the room with some padding
                    if obstacle_w >= room.width - 2 * wall_thickness or obstacle_h >= room.height - 2 * wall_thickness:
                        continue  # Skip if obstacle is too big for the room
                    obstacle_x = random.randint(room.x + wall_thickness, room.x + room.width - obstacle_w - wall_thickness)
                    obstacle_y = random.randint(room.y + wall_thickness, room.y + room.height - obstacle_h - wall_thickness)
                    # Check for overlap
                    if np.any(map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] == 1):
                        continue  # Overlaps with existing obstacle
                    # Place the rectangular obstacle
                    map_grid[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 1
                    placed = True
                    break  # Successfully placed
                if not placed:
                    print(f"Could not place a rectangular obstacle in room at ({room.x}, {room.y}) after {obstacle_attempts} attempts.")
                    continue  # Skip if unable to place after attempts

    # Add outer boundary walls
    # Top and bottom
    map_grid[0:wall_thickness, :] = 1
    map_grid[-wall_thickness:, :] = 1
    # Left and right
    map_grid[:, 0:wall_thickness] = 1
    map_grid[:, -wall_thickness:] = 1
    
    return map_grid

# Example usage and visualization
if __name__ == "__main__":
    generated_map = generate_random_map(
        width=512,
        height=512,
        min_room_size=60,        # Adjusted parameter
        max_room_size=120,       # Adjusted parameter
        max_depth=5,             # Adjusted parameter
        trap_probability=0.4     # Increased probability to include traps
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(generated_map, cmap='Greys', origin='upper')
    plt.title("Generated 2D Map with Concave Traps")
    plt.axis('off')
    plt.savefig('map_with_traps.png')
    plt.show()
