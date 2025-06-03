#!/usr/bin/env python3
"""evaluate_model.py
~~~~~~~~~~~~~~~~~~~~
Evaluate pretrained PointNet models (Exp1–6) on the new test_dataset/.

Each model is loaded and tested against 1000 standardized samples from
test_dataset/, organized into 40 class folders. The script computes
Test Accuracy and F1-score and outputs a CSV summary for each model.

Dependencies: torch, numpy, scikit-learn
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from tqdm import tqdm

# ------------------------
# Define the original PointNet model
# ------------------------

class TNet(nn.Module):
    def __init__(self, k=3):
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
            nn.Linear(256, k * k),
        )
        # Register an identity matrix as a buffer (used in transformation matrix initialization)
        self.register_buffer("iden", torch.eye(k).float())

    def forward(self, x):
        B = x.size(0)
        feat = self.conv(x).max(dim=2)[0]  # Global max pooling
        matrix = self.fc(feat).view(B, self.k, self.k) + self.iden[:self.k, :self.k]  # Add identity for stability
        return matrix

class PointNetCls(nn.Module):
    def __init__(self, k=40, feat_transform=True):
        super().__init__()
        # Transformation network for aligning the input point cloud
        self.input_transform = TNet(k=3)
        # Optional feature transformation network
        self.feat_transform = TNet(k=64) if feat_transform else None

        # First MLP block: transforms (3, N) input to intermediate features
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )

        # Second MLP block: deep feature extraction
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )

        # Fully connected classification head
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, k),
        )

    def forward(self, x):  # x shape: (B, 3, N)
        # Apply input spatial transform
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)

        # Apply first MLP block
        x = self.mlp1(x)

        # Optional feature transformation
        if self.feat_transform:
            trans_feat = self.feat_transform(x)
            x = torch.bmm(trans_feat, x)

        # Apply second MLP block
        x = self.mlp2(x)

        # Global max pooling to get global feature vector
        x = x.max(dim=2)[0]

        # Classification head
        logits = self.fc(x)
        return logits

# ------------------------
# Main logic of inference script
# ------------------------


EXPERIMENTS = {
    1: "XuJiaHeng/exp1_baseline/model_exp1_baseline.pth",
    2: "XuJiaHeng/exp2_traditional/model_exp2_traditional.pth",
    3: "XuJiaHeng/exp3_pointE/model_exp3_pointE.pth",
    4: "XuJiaHeng/exp4_combined_traditional/model_exp4_combined_traditional.pth",
    5: "XuJiaHeng/exp5_combined_pointE/model_exp5_combined_pointE.pth",
    6: "XuJiaHeng/exp6_all_combined/model_exp6_all_combined.pth",
}

TEST_DATASET_DIR = "test_dataset"
OUTPUT_DIR = "evaluation"
NUM_CLASSES = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data():
    """
    Load all point cloud test samples and their corresponding labels from the test dataset directory.
    Returns:
        samples: np.ndarray of shape (num_samples, 1024, 3)
        labels: np.ndarray of shape (num_samples,)
        class_names: list of class folder names
    """
    samples, labels, class_names = [], [], sorted(os.listdir(TEST_DATASET_DIR))
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(TEST_DATASET_DIR) / class_name
        for file in sorted(class_dir.glob("*.npy")):
            pc = np.load(file)
            samples.append(pc)
            labels.append(class_idx)
    return np.array(samples), np.array(labels), class_names


def evaluate(model_path, experiment_id):
    """
    Evaluate a pretrained PointNet model on the test dataset.

    Args:
        model_path: Path to the pretrained .pth model
        experiment_id: Identifier for the experiment (1–6)
    """
    print(f"\n▶️ Evaluating Exp{experiment_id} model...")

    # Load pretrained model
    model = PointNetCls(k=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load test data
    X_test, y_true, _ = load_test_data()
    y_pred = []

    # Perform inference
    with torch.no_grad():
        for pc in tqdm(X_test, desc="Inferencing"):
            pc_tensor = torch.tensor(pc, dtype=torch.float32).T.unsqueeze(0).to(DEVICE)  # Shape: (1, 3, 1024)
            logits = model(pc_tensor)
            pred = logits.argmax(dim=1).item()
            y_pred.append(pred)

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"✅ Exp{experiment_id} Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Save detailed results to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_file = Path(OUTPUT_DIR) / f"test_result_exp{experiment_id}.csv"
    with open(result_file, "w") as f:
        f.write("id,true_label,pred_label\n")
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            f.write(f"{i},{t},{p}\n")
        f.write(f"\nAccuracy,{acc:.4f}\nF1_score,{f1:.4f}\n")


def main():
    """
    Iterate over all experiments and evaluate the corresponding pretrained models.
    Skips any experiments for which the model file is missing.
    """
    for exp_id, model_path in EXPERIMENTS.items():
        if not Path(model_path).exists():
            print(f"⚠️  Model file not found: {model_path}, skipping.")
            continue
        evaluate(model_path, exp_id)


if __name__ == "__main__":
    main()
