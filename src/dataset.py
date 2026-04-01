"""
Handles data loading, ground truth parsing, and RGB-D synchronization.
"""
import numpy as np

def read_file_list(filename):
    """Reads a text file containing timestamps and file paths."""
    with open(filename, 'r') as file:
        data = file.read()

    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list_data = [[v.strip() for v in line.split(" ") if v.strip() != ""]
                 for line in lines if len(line) > 0 and line[0] != "#"]
    list_data = [(float(l[0]), l[1:]) for l in list_data if len(l) > 1]

    return dict(list_data)

def associate_data(first_list, second_list, offset=0.0, max_difference=0.02):
    """Associates data streams by finding pairs with closely matching timestamps."""
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())

    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()

    matches = []
    first_keys_set = set(first_keys)
    second_keys_set = set(second_keys)

    for diff, a, b in potential_matches:
        if a in first_keys_set and b in second_keys_set:
            first_keys_set.remove(a)
            second_keys_set.remove(b)
            matches.append((a, first_list[a][0], b, second_list[b][0]))

    matches.sort()
    return matches

def load_ground_truth(filename):
    """Loads ground truth trajectory data safely. Returns empty dict if missing."""
    try:
        with open(filename, 'r') as file:
            data = file.read()

        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list_data = [[v.strip() for v in line.split(" ") if v.strip() != ""]
                     for line in lines if len(line) > 0 and line[0] != "#"]

        gt_dict = {}
        for l in list_data:
            if len(l) > 7:
                timestamp = float(l[0])
                xyz = np.array([float(l[1]), float(l[2]), float(l[3])])
                gt_dict[timestamp] = xyz

        return gt_dict
    except FileNotFoundError:
        print(f"Notice: Ground truth file '{filename}' not found. Running visualization only.")
        return {}