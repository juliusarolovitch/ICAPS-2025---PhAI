import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from dataclasses import dataclass
from typing import List
import os
import json


@dataclass
class WarehouseZone:
    type: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class Obstacle:
    type: str
    x: float
    y: float
    width: float
    height: float


def create_warehouse_layout(width=50, height=30, num_aisles=4, num_docks=3, num_stations=3, num_obstacles=5,
                            num_pillars=10, seed=None):
    """
    Create a 2D warehouse layout with configurable elements, obstacles, and pillars.

    Parameters:
    width (int): Width of the warehouse in meters
    height (int): Height of the warehouse in meters
    num_aisles (int): Number of storage aisles
    num_docks (int): Number of loading docks
    num_stations (int): Number of packing stations
    num_obstacles (int): Number of obstacles to add
    num_pillars (int): Number of pillars to add
    seed (int): Random seed for reproducibility

    Returns:
    tuple: Figure, axis, layout, obstacles, and pillars
    """
    if seed is not None:
        np.random.seed(seed)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the warehouse boundaries
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Define colors
    colors = ['#4B5563', '#60A5FA', '#34D399', '#FCD34D', '#F87171', '#9CA3AF']

    # Add loading docks
    dock_width, dock_height = 20, 8
    loading_docks = []
    for i in range(num_docks):
        x = 2 + i * (width - 4) / (num_docks - 1)
        y = height - dock_height - 2
        loading_docks.append(WarehouseZone('loading_dock', x, y, dock_width, dock_height))
        ax.add_patch(Rectangle((x, y), dock_width, dock_height, facecolor=colors[0], label='Loading Dock'))

    # Add storage aisles
    aisle_width, aisle_height = 10, 80
    storage_aisles = []
    for i in range(num_aisles):
        x = 10 + i * (width - 20) / (num_aisles - 1)
        y = 5
        storage_aisles.append(WarehouseZone('storage_aisle', x, y, aisle_width, aisle_height))
        ax.add_patch(Rectangle((x, y), aisle_width, aisle_height, facecolor=colors[1], label='Storage Aisle'))

    # Add packing stations
    station_width, station_height = 40, 15 
    packing_stations = []
    for i in range(num_stations):
        x = width - station_width - 5
        y = 5 + i * (height - 10) / (num_stations - 1)
        packing_stations.append(WarehouseZone('packing_station', x, y, station_width, station_height))
        ax.add_patch(Rectangle((x, y), station_width, station_height, facecolor=colors[2], label='Packing Station'))

    # Add obstacles
    obstacles: List[Obstacle] = []
    for _ in range(num_obstacles):
        obstacle_width = np.random.uniform(10, 40)
        obstacle_height = np.random.uniform(10, 40)
        x = np.random.uniform(5, width - obstacle_width - 5)
        y = np.random.uniform(aisle_height + 10, height - obstacle_height - 5)
        obstacles.append(Obstacle('rectangle', x, y, obstacle_width, obstacle_height))
        ax.add_patch(Rectangle((x, y), obstacle_width, obstacle_height, facecolor=colors[3], label='Obstacle'))

    # Add pillars
    pillars: List[Obstacle] = []
    for _ in range(num_pillars):
        pillar_width = np.random.uniform(10, 20)
        pillar_height = np.random.uniform(10, 20)
        x = np.random.uniform(5, width - pillar_width - 5)
        y = np.random.uniform(5, height - pillar_height - 5)
        pillars.append(Obstacle('pillar', x, y, pillar_width, pillar_height))
        ax.add_patch(Rectangle((x, y), pillar_width, pillar_height, facecolor=colors[5], label='Pillar'))

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Set title and labels
    plt.title('Warehouse Layout')
    plt.xlabel('Width (meters)')
    plt.ylabel('Height (meters)')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    layout = loading_docks + storage_aisles + packing_stations
    return fig, ax, layout, obstacles, pillars


def generate_dataset(dataset_dir, num_samples=100, seed_start=0):
    """
    Generate and save a dataset of warehouse layouts to the specified directory.

    Parameters:
    dataset_dir (str): Path to the directory where the dataset will be saved
    num_samples (int): Number of warehouse layouts to generate
    seed_start (int): Starting seed value for reproducibility
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for i in range(num_samples):
        width = 512
        height = 256
        num_aisles = np.random.randint(10, 25)
        num_docks = np.random.randint(10, 15)
        num_stations = np.random.randint(4, 10)
        num_obstacles = np.random.randint(10, 25)
        num_pillars = np.random.randint(20, 40)

        fig, ax, layout, obstacles, pillars = create_warehouse_layout(width=width, height=height, num_aisles=num_aisles,
                                                                      num_docks=num_docks, num_stations=num_stations,
                                                                      num_obstacles=num_obstacles,
                                                                      num_pillars=num_pillars, seed=seed_start + i)

        if not os.path.exists(os.path.join(dataset_dir, f'{i + 1}')):
            os.makedirs(os.path.join(dataset_dir, f'{i + 1}/'))
        sample_path = os.path.join(dataset_dir, f'{i + 1}/layout_{i + 1}.json')
        fig_path = os.path.join(dataset_dir, f'{i + 1}/layout_{i + 1}.png')

        with open(sample_path, 'w') as f:
            json.dump({
                'layout': [zone.__dict__ for zone in layout],
                'obstacles': [obstacle.__dict__ for obstacle in obstacles],
                'pillars': [pillar.__dict__ for pillar in pillars]
            }, f, indent=2)

        plt.savefig(fig_path)
        plt.close(fig)


def main():
    generate_dataset(args.dataset_dir, num_samples=args.num_samples, seed_start=args.seed_start)

if __name__ == "__main__":
    # parse arguments

    import argparse
    parser = argparse.ArgumentParser(description='Generate 2D warehouse layouts.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of warehouse layouts to generate')
    parser.add_argument('--seed_start', type=int, default=0, help='Starting seed value for reproducibility')
    parser.add_argument('--dataset_dir', type=str, default='warehouse_dataset', help='Path to the dataset directory')
    args = parser.parse_args()

    main()