#!/usr/bin/env python3
"""build_test_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~
Build a new test dataset from ModelNet40 HDF5 files by extracting 25 samples
per class and saving them as normalized .npy files. This test set is used to
evaluate pretrained PointNet models.

The output directory will contain 40 folders (one per class), each with 25 samples
named <class_name>_000.npy to <class_name>_024.npy.

Requirements:
-------------
• HDF5 input files: test0.h5 and test1.h5 from modelnet40_hdf5_2048/
• Class names from shape_names.txt
• Output directory: test_dataset/

Each output sample is:
- Shape: (1024, 3)
- Centered at origin
- Scaled such that max point norm ≤ 1

Usage examples
--------------
# Run from project root to generate test_dataset/ with 1000 samples
$ python build_test_dataset.py

Dependencies: numpy, h5py (install via `pip install numpy h5py`)
"""

import os
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration parameters
H5_FILES = ["modelnet40_hdf5_2048/test0.h5", "modelnet40_hdf5_2048/test1.h5"]  # Paths to input HDF5 files
SHAPE_NAMES_FILE = "modelnet40_hdf5_2048/shape_names.txt"  # Path to the file containing class names
OUTPUT_DIR = "test_dataset"  # Output directory for saving processed point cloud samples
SAMPLES_PER_CLASS = 25       # Number of samples to extract per class
POINT_DIM = 3                # Each point has 3 dimensions (x, y, z)
POINT_NUM = 1024             # Number of points per sample

def load_h5_file(path):
    """Load point cloud data and labels from a single HDF5 file."""
    with h5py.File(path, 'r') as f:
        data = f['data'][:]   # Shape: (N, 1024, 3)
        label = f['label'][:] # Shape: (N, 1)
    return data, label.squeeze()  # Return data and flattened labels

def normalize_pointcloud(pc):
    """Normalize a point cloud:
    - Center it at the origin
    - Scale so that the furthest point has norm ≤ 1
    """
    pc = pc - np.mean(pc, axis=0)  # Center the point cloud at the origin
    scale = np.max(np.linalg.norm(pc, axis=1))  # Compute the maximum distance from the origin
    return pc / scale  # Scale all points by this distance


def main():
    # Load class names from the shape_names file
    with open(SHAPE_NAMES_FILE, "r") as f:
        shape_names = [line.strip() for line in f.readlines()]

    # Load and merge all test data and labels from HDF5 files
    all_data, all_labels = [], []
    for file in H5_FILES:
        data, labels = load_h5_file(file)
        all_data.append(data)
        all_labels.append(labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Group sample indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        class_to_indices[label].append(idx)

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving {SAMPLES_PER_CLASS} samples for each of {len(shape_names)} classes...")

    # Process each class and save selected samples
    for class_id, class_name in enumerate(shape_names):
        indices = class_to_indices[class_id]
        np.random.shuffle(indices)  # Shuffle indices randomly
        selected = indices[:SAMPLES_PER_CLASS]  # Select the first N samples

        class_dir = Path(OUTPUT_DIR) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(selected):
            pc = normalize_pointcloud(all_data[idx])  # Normalize the point cloud
            out_path = class_dir / f"{class_name}_{i:03d}.npy"  # Output file path
            np.save(out_path, pc)

    print(f"✅ Done! Saved 1000 samples in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
