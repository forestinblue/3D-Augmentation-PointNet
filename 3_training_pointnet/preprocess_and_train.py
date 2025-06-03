# -*- coding: utf-8 -*-
"""
preprocess_and_train.py
~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end pipeline for Elvin's 3D point cloud classification experiments.

Features
--------
1. Pre-process three datasets (Original, Traditionally Augmented, Point·E) so that each
   sample becomes a **(1024, 3) float32** point cloud centered at the origin with max ‖p‖ ≤ 1.
2. Build six experiment folders (exp1 – exp6) exactly as required.
3. Generate *train_files.txt* and *expX_failed_samples.txt* for each experiment.
4. Fine-tune a pretrained **PointNet** on exp1–exp3 using defined hyperparameters,
   saving the model weights and training logs for each epoch.

Run Examples
------------
# To build data folders only
python preprocess_and_train.py --stage preprocess --base_dir .

# To run training after data preparation
python preprocess_and_train.py --stage train --base_dir .

# To run the full pipeline in one step
python preprocess_and_train.py --stage all --base_dir .

Dependencies
------------
- numpy, pandas, torch, plyfile (install with: pip install plyfile), tqdm
- (optional) scikit-learn for confusion matrix plotting
"""

# Standard library imports
from __future__ import annotations
import argparse
import csv
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional dependency for reading PLY files
try:
    from plyfile import PlyData
except ImportError:  # Will be skipped in test coverage
    PlyData = None

# Required dependency: PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print("PyTorch is required to run training: pip install torch", file=sys.stderr)
    raise e

################################################################################
#                            CONFIGURATION SETTINGS                            #
################################################################################

# Experiment ID to folder name mapping
EXP_MAP = {
    1: "exp1_baseline",
    2: "exp2_traditional",
    3: "exp3_pointE",
    4: "exp4_combined_traditional",
    5: "exp5_combined_pointE",
    6: "exp6_all_combined",
}

# List of 40 class names from ModelNet40
CLASS_NAMES: Tuple[str, ...] = (
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
)

# Mapping from label ID to class name
LABEL_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

# Global settings
NUM_POINTS = 1024       # Number of points per sample
SEED = 42               # Random seed for reproducibility
EPOCHS = 50             # Number of training epochs
BATCH_SIZE = 32         # Training batch size
LR = 1e-3               # Learning rate

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

################################################################################
#                               UTILITY FUNCTIONS                              #
################################################################################

def ensure_dir(path: Path):
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def resample_points(points: np.ndarray, num_points: int = NUM_POINTS) -> np.ndarray:
    """Uniformly sample points from a point cloud.

    If the input point cloud has fewer than `num_points` points,
    sampling is done with replacement. Otherwise, without replacement.
    """
    if points.shape[0] < num_points:
        choice = np.random.choice(points.shape[0], num_points, replace=True)
    else:
        choice = np.random.choice(points.shape[0], num_points, replace=False)
    return points[choice]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center the point cloud to the origin and scale it so that the maximum point norm is ≤ 1."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    if furthest_distance == 0:
        return points  # Avoid division by zero while preserving float32 type
    points /= furthest_distance
    return points


def preprocess_sample(points: np.ndarray, *, num_points: int = NUM_POINTS) -> np.ndarray:
    """Resample and normalize a point cloud into a (1024, 3) shape."""
    points = resample_points(points, num_points).astype(np.float32)
    points = normalize_points(points)
    return points


################################################################################
#                               DATA INGESTION                                #
################################################################################

def load_ply_points(path: Path) -> np.ndarray:
    """Load 3D point cloud coordinates from a .ply file."""
    if PlyData is None:
        raise RuntimeError("plyfile is not installed. Please run: pip install plyfile")
    plydata = PlyData.read(str(path))
    data = plydata["vertex"]
    pts = np.vstack([data["x"], data["y"], data["z"]]).T.astype(np.float32)
    return pts


################################################################################
#                        EXPERIMENT FOLDER BUILDERS                           #
################################################################################

def process_original_and_augmented(src_dir: Path, data_fname: str, label_fname: str, dest_exp: Path):
    """Preprocess the original or traditionally augmented dataset and save it per class."""
    data = np.load(src_dir / data_fname)  # shape: (N, 2048, 3)
    labels = np.load(src_dir / label_fname)  # shape: (N,)

    assert data.shape[0] == labels.shape[0], "Mismatch between number of data and labels"
    for idx in tqdm(range(data.shape[0]), desc=f"Preprocessing {dest_exp.name}"):
        cls = LABEL_TO_CLASS[int(labels[idx])]
        dest_cls = dest_exp / cls
        ensure_dir(dest_cls)
        filename = f"{cls}_{idx:04d}.npy"
        out_path = dest_cls / filename
        processed = preprocess_sample(data[idx])
        np.save(out_path, processed)


def process_point_e(src_root: Path, dest_exp: Path):
    """Preprocess all Point·E-generated point clouds and store them class-wise."""
    classes = [d for d in src_root.iterdir() if d.is_dir()]
    for cls_dir in tqdm(classes, desc="Preprocessing Point·E dataset"):
        cls = cls_dir.name
        ply_files = list(cls_dir.glob("*.ply"))
        for ply_path in ply_files:
            try:
                points = load_ply_points(ply_path)
                processed = preprocess_sample(points)
            except Exception as e:
                print(f"[Warning] Could not read {ply_path}: {e}")
                continue
            dest_cls = dest_exp / cls
            ensure_dir(dest_cls)
            out_name = ply_path.stem + ".npy"
            np.save(dest_cls / out_name, processed)


def copy_tree_with_tag(src: Path, dst: Path, tag: str):
    """Copy all .npy files from `src` to `dst`, adding a `tag` prefix to each filename.

    This prevents overwriting and ensures correct sample counts in merged datasets.
    """
    for cls_dir in src.iterdir():
        if not cls_dir.is_dir():
            continue
        dest_cls = dst / cls_dir.name
        ensure_dir(dest_cls)
        for f in cls_dir.glob("*.npy"):
            new_name = f"{tag}_{f.name}"
            shutil.copy2(f, dest_cls / new_name)


################################################################################
#                              VALIDATION                                     #
################################################################################

def validate_exp_dir(exp_dir: Path):
    """Check all samples in an experiment directory for validity.

    Logs any sample that:
    - Cannot be loaded
    - Has incorrect shape
    - Contains NaNs or Infs
    - Exceeds the norm threshold
    """
    failed_log = exp_dir / f"{exp_dir.name}_failed_samples.txt"
    with open(failed_log, "w", newline="\n") as fh:
        for npy_path in exp_dir.rglob("*.npy"):
            rel = npy_path.relative_to(exp_dir)
            try:
                pts = np.load(npy_path)
            except Exception as e:
                fh.write(f"{rel} – cannot read ({e})\n")
                continue

            reason = None
            if pts.shape != (NUM_POINTS, 3):
                reason = f"Invalid shape {pts.shape}"
            elif not np.isfinite(pts).all():
                reason = "Contains NaN or Inf values"
            else:
                max_norm = np.max(np.linalg.norm(pts, axis=1))
                if max_norm > 1.05:
                    reason = f"Max norm > {max_norm:.3f}"
            if reason:
                fh.write(f"{rel} – {reason}\n")
        # Empty log file means all samples passed


################################################################################
#                          TRAIN FILE GENERATION                              #
################################################################################

def generate_train_list(exp_dir: Path):
    """Create a text file listing all training file paths (relative to `exp_dir`)."""
    paths: List[str] = []
    for cls in sorted([d for d in exp_dir.iterdir() if d.is_dir()]):
        files = sorted(cls.glob("*.npy"))
        paths.extend(str(p.relative_to(exp_dir)) for p in files)
    txt_path = exp_dir / "train_files.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))


################################################################################
#                               DATASET                                        #
################################################################################

class NpyDataset(Dataset):
    """Custom PyTorch Dataset for loading .npy point cloud files."""

    def __init__(self, root: Path):
        self.samples: List[Tuple[Path, int]] = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            for f in (root / cls_name).glob("*.npy"):
                self.samples.append((f, cls_idx))
        if not self.samples:
            raise RuntimeError(f"No samples found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pts = np.load(path).astype(np.float32)
        pts = torch.from_numpy(pts).transpose(0, 1)  # shape becomes (3, N)
        return pts, label


################################################################################
#                               POINTNET                                       #
################################################################################

class TNet(nn.Module):
    """
    Mini T-Net module used inside PointNet for transformation estimation.
    It learns an affine transformation matrix to align the input features
    or intermediate features into a canonical space.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),  # Output transformation matrix
        )
        self.register_buffer("iden", torch.eye(k).float())  # Identity matrix for stability

    def forward(self, x):  # x shape: (B, k, N)
        B = x.size(0)
        feat = self.conv(x).max(dim=2)[0]  # Global feature vector (B, 1024)
        matrix = self.fc(feat).view(B, self.k, self.k) + self.iden[:self.k, :self.k]
        return matrix


class PointNetCls(nn.Module):
    """
    PointNet Classification Network.

    - Applies input transformation (TNet).
    - Extracts local and global features.
    - Optionally applies feature transformation (TNet).
    - Outputs class logits using a fully connected MLP.
    """

    def __init__(self, k: int = 40, feat_transform: bool = True):
        super().__init__()
        self.input_transform = TNet(k=3)
        self.feat_transform = TNet(k=64) if feat_transform else None

        # First shared MLP block (64-dimensional features)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )

        # Second shared MLP block (64 → 128 → 1024)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, k),  # Output logits for k classes
        )

    def forward(self, x):  # x shape: (B, 3, N)
        # Input TNet transformation
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)  # Align input using transformation

        # Feature extraction (shared MLP)
        x = self.mlp1(x)

        # Optional feature transformation TNet
        if self.feat_transform:
            trans_feat = self.feat_transform(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None

        # Second MLP block and global max pooling
        x = self.mlp2(x)
        x = x.max(dim=2)[0]  # Global feature: shape (B, 1024)

        # Classification
        logits = self.fc(x)
        return logits


################################################################################
#                               TRAINING                                       #
################################################################################

def train_one_exp(exp_dir: Path, model_out: Path, log_csv: Path, device: str, *, pretrained: Path | None = None):
    """
    Train PointNet on a given experiment directory.

    Args:
        exp_dir (Path): Directory containing processed .npy training samples.
        model_out (Path): Path to save trained model weights (.pth).
        log_csv (Path): Path to save training log as a CSV file.
        device (str): 'cuda' or 'cpu'.
        pretrained (Path | None): Optional path to pretrained weights for fine-tuning.
    """
    # Load dataset and initialize data loader
    dataset = NpyDataset(exp_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # Initialize PointNet model
    model = PointNetCls(k=len(CLASS_NAMES)).to(device)

    # Load pretrained weights if specified (used for fine-tuning exp2–exp6)
    if pretrained is not None:
        print(f"→ Load pretrained weights from {pretrained.relative_to(exp_dir.parent)}")
        state = torch.load(pretrained, map_location=device)
        model.load_state_dict(state)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Open CSV logger
    headers = ["epoch", "loss", "acc"]
    with open(log_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)

        # Training loop
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = correct = total = 0.0
            for pts, labels in loader:
                pts, labels = pts.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(pts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * pts.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += pts.size(0)

            # Log epoch performance
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            writer.writerow([epoch, epoch_loss, epoch_acc])
            print(f"[Exp {exp_dir.name}] Epoch {epoch:02d}/50 – Loss {epoch_loss:.4f} – Acc {epoch_acc:.4f}")

    # Save final model weights
    torch.save(model.state_dict(), model_out)


################################################################################
#                               ORCHESTRATOR                                   #
################################################################################

def build_all_experiments(base_dir: Path):
    """
    Build and preprocess all six experiments: exp1–exp6.

    This function:
    - Preprocesses original, augmented, and Point·E datasets (exp1–3)
    - Combines datasets for extended experiments (exp4–6)
    - Validates data and generates train_files.txt
    """
    task_pkg = base_dir / "Elvin_Task_Package"
    pointe_root = base_dir / "ModelNet40_PointE" / "ModelNet40_PointE"  # skip __MACOSX folders if present

    # Preprocess datasets for exp1–exp3
    exp1 = base_dir / "exp1_baseline";
    ensure_dir(exp1)
    exp2 = base_dir / "exp2_traditional";
    ensure_dir(exp2)
    exp3 = base_dir / "exp3_pointE";
    ensure_dir(exp3)

    process_original_and_augmented(task_pkg, "modelnet40_sampled_data.npy", "modelnet40_sampled_labels.npy", exp1)
    process_original_and_augmented(task_pkg, "modelnet40_sampled_data_augmented.npy",
                                   "modelnet40_sampled_labels_augmented.npy", exp2)
    process_point_e(pointe_root, exp3)

    # Create combined experiments (exp4–exp6)
    exp4 = base_dir / "exp4_combined_traditional";
    ensure_dir(exp4)
    exp5 = base_dir / "exp5_combined_pointE";
    ensure_dir(exp5)
    exp6 = base_dir / "exp6_all_combined";
    ensure_dir(exp6)

    copy_tree_with_tag(exp1, exp4, tag="ori")
    copy_tree_with_tag(exp2, exp4, tag="trad")

    copy_tree_with_tag(exp1, exp5, tag="ori")
    copy_tree_with_tag(exp3, exp5, tag="pe")

    copy_tree_with_tag(exp1, exp6, tag="ori")
    copy_tree_with_tag(exp2, exp6, tag="trad")
    copy_tree_with_tag(exp3, exp6, tag="pe")

    # Validate and create train_files.txt for each experiment
    for exp in (exp1, exp2, exp3, exp4, exp5, exp6):
        validate_exp_dir(exp)
        generate_train_list(exp)


################################################################################
#                                   MAIN                                       #
################################################################################

def main():
    """
    Main entry point for running preprocessing and/or training.

    Command line arguments:
    --stage [preprocess/train/all]: which stage to run
    --base_dir: root directory where data and outputs are stored
    --device: 'cuda' or 'cpu'
    """
    parser = argparse.ArgumentParser(description="Pipeline for PointNet experiments")
    parser.add_argument("--stage", choices=["preprocess", "train", "all"], default="all",
                        help="Which stage to run")
    parser.add_argument("--base_dir", default=".", help="Base directory containing raw data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()

    if args.stage in ("preprocess", "all"):
        print("==========  PREPROCESS  ==========")
        build_all_experiments(base_dir)

    if args.stage in ("train", "all"):
        exp1_name = EXP_MAP[1]
        m1 = base_dir / exp1_name / f"model_{exp1_name}.pth"

        print("==========   TRAINING (pretrain → finetune)   ==========")
        for exp_id, exp_name in EXP_MAP.items():
            exp_dir = base_dir / exp_name
            model_out = exp_dir / f"model_{exp_name}.pth"
            log_csv = exp_dir / f"train_log_exp{exp_id}.csv"
            if exp_id == 1:
                # Train from scratch
                train_one_exp(exp_dir, model_out, log_csv, device=args.device)
            else:
                # Fine-tune from pretrained exp1 model
                train_one_exp(exp_dir, model_out, log_csv, device=args.device, pretrained=m1)


if __name__ == "__main__":
    main()
