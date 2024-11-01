import numpy as np
import matplotlib.pyplot as plt
import random


def generate_random_walls_map(
    width=2000,          # Width of the map
    height=400,          # Height of the map
    wall_thickness=5,    # Thickness of the walls
    num_walls=100,       # Number of random walls to place
    min_wall_length=50,  # Minimum length of a wall
    max_wall_length=300,  # Maximum length of a wall
    wall_gap=10,         # Minimum gap between walls
):
    """
    Generates a 2D map with randomly placed horizontal and vertical walls.

    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        wall_thickness (int): Thickness of the walls.
        num_walls (int): Number of random walls to place.
        min_wall_length (int): Minimum length of a wall.
        max_wall_length (int): Maximum length of a wall.
        wall_gap (int): Minimum gap between walls to prevent overcrowding.

    Returns:
        np.ndarray: 2D occupancy map of shape [height, width].
    """
    map_grid = np.zeros((height, width), dtype=np.float32)

    # Add outer boundary walls
    map_grid[0:wall_thickness, :] = 1  # Top
    map_grid[-wall_thickness:, :] = 1  # Bottom
    map_grid[:, 0:wall_thickness] = 1  # Left
    map_grid[:, -wall_thickness:] = 1  # Right

    for i in range(num_walls):
        # Randomly decide wall orientation
        orientation = random.choice(['horizontal', 'vertical'])

        if orientation == 'horizontal':
            # Random y position ensuring walls stay within boundaries
            y = random.randint(wall_thickness + wall_gap,
                               height - wall_thickness - wall_gap)
            # Random x position and wall length
            wall_length = random.randint(min_wall_length, max_wall_length)
            x = random.randint(wall_gap, width - wall_length - wall_gap)

            # Check for overlap and place the wall
            if np.all(map_grid[y:y + wall_thickness, x:x + wall_length] == 0):
                map_grid[y:y + wall_thickness, x:x + wall_length] = 1
            else:
                # Overlaps, skip or retry
                continue

        else:  # vertical
            # Random x position ensuring walls stay within boundaries
            x = random.randint(wall_thickness + wall_gap,
                               width - wall_thickness - wall_gap)
            # Random y position and wall length
            wall_length = random.randint(min_wall_length, max_wall_length)
            y = random.randint(wall_gap, height - wall_length - wall_gap)

            # Check for overlap and place the wall
            if np.all(map_grid[y:y + wall_length, x:x + wall_thickness] == 0):
                map_grid[y:y + wall_length, x:x + wall_thickness] = 1
            else:
                # Overlaps, skip or retry
                continue

    return map_grid


# Example usage and visualization
if __name__ == "__main__":
    generated_map = generate_random_walls_map(
        width=2000,
        height=400,
        wall_thickness=5,
        num_walls=50,           # Increased number for more complexity
        min_wall_length=50,
        max_wall_length=300,
        wall_gap=20              # Increased gap to prevent overcrowding
    )
    print("Created map")
    plt.figure(figsize=(20, 4))  # Adjusted figure size for rectangular map
    plt.imshow(generated_map, cmap='Greys', origin='upper')
    plt.title("Generated 2D Map with Random Walls")
    plt.axis('off')
    plt.savefig('xyt_map.png', bbox_inches='tight', pad_inches=0, dpi=20)
