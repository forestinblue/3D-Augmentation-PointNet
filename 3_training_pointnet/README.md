# 📦 3D Point Cloud Classification Pipeline with PointNet

This project provides an end-to-end pipeline for **preprocessing**, **training**, and **visualizing** a 3D point cloud classification system using **PointNet**. It includes:

- `preprocess_and_train.py`: Dataset preparation & model training
- `visualize_pointcloud.py`: 3D preview tool for `.npy` samples
- `visualize_training_logs.py`: Plotting training curves from CSV logs

---

## 📁 Project Structure

```
project_root/
├── exp1_baseline/ # Generated experiment folder (after preprocess)
│ ├── airplane/
│ │ └── airplane_0000.npy
│ ├── train_log_exp1.csv # Training log
│ └── model_exp1_baseline.pth # Trained PointNet weights
├── preprocess_and_train.py # Main training pipeline
├── visualize_pointcloud.py # Script for rendering 3D point clouds
├── visualize_training_logs.py # Script for plotting training metrics
└── README.md
```
---

## 🔧 Dependencies

Install all required packages:

```bash
pip install numpy pandas torch plyfile tqdm matplotlib
```
## 🚀 Script 1: preprocess_and_train.py

### 🧭 Purpose
Preprocesses three datasets: Original, Traditional-Augmented, and Point-E
Builds 6 experiment directories: baseline, augmented, Point-E, and combinations
Trains PointNet on each dataset
Saves logs and trained weights
### ▶️ Usage
```bash
# Preprocess only
python preprocess_and_train.py --stage preprocess --base_dir .

# Train only (after preprocessing)
python preprocess_and_train.py --stage train --base_dir .

# Full pipeline (preprocess + train)
python preprocess_and_train.py --stage all --base_dir .
```
## 👁️ Script 2: visualize_pointcloud.py

### 🧭 Purpose
Visualizes a single .npy file (normalized 3D point cloud)
Supports both interactive display and saving to PNG
### ▶️ Usage
```bash
# Visualize sample from exp3, class 'airplane', index 0
python visualize_pointcloud.py --exp 3 --cls airplane --idx 0

# Visualize and save to PNG
python visualize_pointcloud.py --exp 2 --cls 5 --idx 4 --save chair5.png
```
## 📊 Script 3: visualize_training_logs.py

### 🧭 Purpose
Plots training loss and accuracy curves from CSV logs
### ▶️ Usage
```bash
# Plot all available training logs
python visualize_training_logs.py

# Plot accuracy only for exp1, exp2, exp3
python visualize_training_logs.py --exp 1 2 3 --metric acc

# Save both metrics to image
python visualize_training_logs.py --metric both --save logs.png
```
## 📝 Notes

All experiments are based on ModelNet40 (40-class classification).
Each sample is a .npy file containing a (1024, 3) float32 point cloud.
All training logs are saved as train_log_expX.csv in the respective experiment folder.
✅ Example Outputs

Trained model weights: exp1_baseline/model_exp1_baseline.pth
Visualizations: chair5.png
Training logs: train_log_exp1.csv
For evaluation & analysis tools (build_test_dataset.py, evaluate_model.py, analyze_and_plot_results.py), see the extended README_eval.md (or ask for a combined version).
