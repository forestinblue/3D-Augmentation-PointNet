[README.md](README.md)
# 🧠 Text-to-3D Point Cloud Generation using Point-E (ModelNet40)

This project uses **OpenAI's Point-E** diffusion model to generate **3D point clouds** from **text prompts**, customized for the [ModelNet40](https://modelnet.cs.princeton.edu/) object classes.

---

## 🚀 Overview

- 🔡 **Input**: Class names from ModelNet40 (e.g., `chair`, `car`, `bathtub`, …)
- ✏️ **Prompt Templates**: 10 prompt variations per class (e.g., `"a 3D point cloud of a chair"`)
- 🌀 **Generation**:
  - 25 samples per class
  - Uses **base model** + **upsampler model**
- 💾 **Output**: `.ply` point cloud files with RGB color

---

## 📁 Project Structure

```

📦 modelnet\_point\_e\_generator/
├── modelnet40\_point\_e\_generate.py      # ⬅️ Main generation script
├── ModelNet40\_sampled/                 # Input data (used for class names only)
│   ├── chair/
│   │   └── chair\_0001.off
│   └── ...
└── ModelNet40\_generated\_v2/            # Output point clouds (.ply)
├── chair/
│   └── chair\_0000.ply
└── ...

````

---

## 🔧 Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended)
- Install dependencies:

```bash
pip install torch numpy tqdm
pip install git+https://github.com/openai/point-e.git
````

---

## 🧩 Key Features

| Component                  | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `prompt_templates`         | 10 customizable natural language templates           |
| `PointCloudSampler`        | Combines base + upsampler models for high-res output |
| `generate_class_samples()` | Saves 25 `.ply` files per class                      |
| `load_modelnet40()`        | Parses `.off` files to retrieve class list           |

---

## 📜 Example Prompts

For class `"car"`:

```text
- a 3D point cloud of a car
- a realistic 3D scan of a car
- a point cloud of a car rendered in a virtual scene
...
```

Each class generates 10 prompts × {2, 3} samples = 25 files.

---

## ✅ How to Run

1. Clone and install Point-E:

```bash
git clone https://github.com/openai/point-e
cd point-e
pip install -e .
```

2. Place a sampled ModelNet40 dataset in `ModelNet40_sampled/` for class name extraction.

3. Run the script:

```bash
python modelnet40_point_e_generate.py
```

4. Generated `.ply` files will appear in `ModelNet40_generated_v2/{class_name}/`.

---

## 📊 Output Sample Naming

| Sample Name      | Prompt Group | Sample Count |
| ---------------- | ------------ | ------------ |
| `chair_0000.ply` | G1           | 1st of 10    |
| `chair_0015.ply` | G2           | 1st of 15    |

Each sample is saved with the class name + index.

---

## 🖼️ Visualizing Output

You can visualize the `.ply` files using:

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("chair/chair_0000.ply")
o3d.visualization.draw_geometries([pcd])
```

---

## 🧠 Acknowledgements

* [OpenAI Point-E](https://github.com/openai/point-e)
* [ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)
* [Open3D for Visualization](http://www.open3d.org/)

---

## 📌 Notes

* You can adjust `prompt_templates`, number of samples, or guidance scale to experiment.
* This script is designed for class-level generation, not per-instance reconstruction.

---

## 📂 Optional Extensions

* 🔁 Integrate CLIP for prompt filtering
* 🧪 Add SSIM/LPIPS metrics to compare with original shapes
* 🌐 Upload generated point clouds to [Sketchfab](https://sketchfab.com/) or 3D viewers

```

