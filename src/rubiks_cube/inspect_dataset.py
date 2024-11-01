import pickle
import pprint
from collections import defaultdict

# Load the dataset
with open('datasets/rubiks_cube_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"\nTotal number of state pairs in dataset: {len(dataset)}")
print(f"Number of unique cubes: {len(set(entry['cube_idx'] for entry in dataset))}")

# Get statistics about number of states per cube
states_per_cube = defaultdict(int)
for entry in dataset:
    states_per_cube[entry['cube_idx']] += 1

print(f"\nAverage states per cube: {sum(states_per_cube.values()) / len(states_per_cube):.2f}")

print("\nFirst 3 entries in the dataset:")
pp = pprint.PrettyPrinter(indent=2)
for i, entry in enumerate(dataset[:100]):
    print(f"\nEntry {i+1}:")
    pp.pprint(entry)

# Print g-value statistics
forward_g_values = [entry['forward_g_value'] for entry in dataset]
backward_g_values = [entry['backward_g_value'] for entry in dataset]

print("\nG-value statistics:")
print(f"Forward g-values  - Min: {min(forward_g_values)}, Max: {max(forward_g_values)}, Avg: {sum(forward_g_values)/len(forward_g_values):.2f}")
print(f"Backward g-values - Min: {min(backward_g_values)}, Max: {max(backward_g_values)}, Avg: {sum(backward_g_values)/len(backward_g_values):.2f}")