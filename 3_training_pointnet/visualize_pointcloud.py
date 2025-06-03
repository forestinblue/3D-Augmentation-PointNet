#!/usr/bin/env python3
"""visualize_pointcloud.py

Quick 3D preview of a single point cloud sample stored as a .npy file.

Usage
-----
$ python visualize_pointcloud.py --exp 1 --cls airplane --idx 3 [--root ./]
$ python visualize_pointcloud.py -e 3 -c 17 -i 0          # class by numeric id

Arguments
---------
--exp / -e  int      Experiment number 1‑6 (maps to exp1_baseline, ..., exp6_all_combined)
--cls / -c  str|int  Class name (e.g., "chair") or zero‑based class id (based on sorted folders)
--idx / -i  int      Zero‑based sample index within the class folder (e.g., 0–24)
--root      path     Root directory where expX folders are located (default is current dir)
--save      path     Optional: Save PNG instead of showing interactive window
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)
import numpy as np

# Mapping from experiment index to directory name
EXP_DIR_MAP = {
    1: "exp1_baseline",
    2: "exp2_traditional",
    3: "exp3_pointE",
    4: "exp4_combined_traditional",
    5: "exp5_combined_pointE",
    6: "exp6_all_combined",
}


def parse_args():
    """
    Parse command-line arguments for visualizing a point cloud sample.

    Returns:
        argparse.Namespace: Parsed arguments object
    """
    p = argparse.ArgumentParser(description="Visualize a point cloud .npy sample")
    p.add_argument("--exp", "-e", type=int, required=True, choices=range(1, 7),
                   help="Experiment number (1‑6)")
    p.add_argument("--cls", "-c", required=True,
                   help="Class name (e.g. 'airplane') or zero‑based class ID")
    p.add_argument("--idx", "-i", type=int, required=True,
                   help="Sample index within the class (zero-based)")
    p.add_argument("--root", type=str, default=".",
                   help="Root directory containing experiment folders")
    p.add_argument("--save", type=str, default=None,
                   help="If specified, save the output as a PNG to the given path")
    return p.parse_args()


def class_name_from_arg(cls_arg, exp_path):
    """
    Convert user input for class (either name or index) into a valid class folder name.

    Args:
        cls_arg (str|int): Class name or numeric ID provided by the user
        exp_path (Path): Path to the experiment folder

    Returns:
        str: Valid class folder name

    Raises:
        SystemExit: If the class name or index is invalid
    """
    class_dirs = sorted([p.name for p in exp_path.iterdir() if p.is_dir()])
    try:
        # Try interpreting the class argument as an integer index
        cls_id = int(cls_arg)
        if not (0 <= cls_id < len(class_dirs)):
            raise ValueError
        return class_dirs[cls_id]
    except ValueError:
        # If not an integer, check if it matches a valid class name
        if cls_arg not in class_dirs:
            raise SystemExit(
                f"Class '{cls_arg}' not found in {exp_path}. Available: {class_dirs}"
            )
        return cls_arg



def ensure_equal_aspect(ax):
    """
    Force equal aspect ratio for 3D scatter plot so the object doesn't appear distorted.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D axis object
    """
    lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    spans = lims[:, 1] - lims[:, 0]              # Get range for each axis
    centers = lims[:, 0] + spans / 2             # Compute center of each axis
    max_span = max(spans)                        # Use the largest span to set uniform limits

    # Set all axes to have equal range centered around their midpoints
    ax.set_xlim(centers[0] - max_span / 2, centers[0] + max_span / 2)
    ax.set_ylim(centers[1] - max_span / 2, centers[1] + max_span / 2)
    ax.set_zlim(centers[2] - max_span / 2, centers[2] + max_span / 2)


def main():
    """
    Entry point for script execution. Parses arguments, loads the target .npy point cloud,
    and visualizes or saves it as a 3D scatter plot.
    """
    args = parse_args()

    # Map experiment number to folder path
    exp_dir_name = EXP_DIR_MAP[args.exp]
    exp_path = Path(args.root).expanduser().resolve() / exp_dir_name
    if not exp_path.is_dir():
        sys.exit(f"Experiment directory '{exp_path}' not found.")

    # Resolve class folder name
    cls_name = class_name_from_arg(args.cls, exp_path)
    class_path = exp_path / cls_name
    npy_files = sorted(class_path.glob("*.npy"))
    if not npy_files:
        sys.exit(f"No .npy files in {class_path}")
    if not (0 <= args.idx < len(npy_files)):
        sys.exit(f"Sample index {args.idx} out of range 0‑{len(npy_files) - 1}")

    # Load the selected .npy point cloud sample
    sample_path = npy_files[args.idx]
    pts = np.load(sample_path)
    if pts.shape[1] != 3:
        sys.exit(f"Expected N×3 points, got shape {pts.shape}")

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, depthshade=True)
    ax.set_title(f"{exp_dir_name} | {cls_name} | idx {args.idx}\n{sample_path.name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ensure_equal_aspect(ax)  # Keep uniform scaling
    ax.grid(False)

    # Show the plot or save it as a PNG
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
