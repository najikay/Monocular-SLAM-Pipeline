"""
Data loading and synchronization module.
Handles reading dataset files, parsing ground truth,
and associating RGB and depth images based on timestamps.
"""
import numpy as np


def read_file_list(filename):
    """
    Reads a text file containing timestamps and file paths.

    Args:
        filename (str): Path to the text file.
    Returns:
        dict: A dictionary mapping timestamps (float) to file paths (str).
    """
    with open(filename, 'r') as file:
        data = file.read()

    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list_data = [[v.strip() for v in line.split(" ") if v.strip() != ""]
                 for line in lines if len(line) > 0 and line[0] != "#"]
    list_data = [(float(l[0]), l[1:]) for l in list_data if len(l) > 1]

    return dict(list_data)


def associate_data(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associates two synchronized data streams (RGB and depth)
    by finding pairs with closely matching timestamps.

    Args:
        first_list (dict): First dictionary of timestamp-to-data.
        second_list (dict): Second dictionary of timestamp-to-data.
        offset (float): Optional time offset between the two streams.
        max_difference (float): Maximum allowed time difference for a match.
    Returns:
        list: A sorted list of matched tuples (time1, data1, time2, data2).
    """
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
    """
    Loads ground truth trajectory data safely.

    Args:
        filename (str): Path to the ground truth text file.
    Returns:
        dict: A dictionary mapping timestamps to 3D position vectors (np.array),
              or an empty dict if the file is missing.
    """
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
        print(f"Warning: Ground truth file {filename} not found. Running without evaluation.")
        return {}