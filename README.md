
# 📦 Comparative Study of Traditional vs. AIGC-based 3D Point Cloud Augmentation

This repository contains a course project conducted at **CUHK-Shenzhen** for the **"Emerging Topics in AI"** course (Spring 2025).  
The project compares **traditional geometric augmentation** and **generative augmentation** (via Point-E) on point cloud classification tasks using PointNet.

---

## 🧠 Project Goal

To evaluate the effectiveness of **generative data augmentation (Point-E)** compared to **traditional augmentation techniques**  
for improving the performance of a 3D point cloud classifier on the **ModelNet40** dataset.

---

## 🧪 Experiment Overview

| Experiment | Augmentation Strategy        | Accuracy (%) | F1-Score (%) |
|------------|------------------------------|---------------|--------------|
| Exp1       | No Augmentation (Baseline)   | 87.4          | 86.7         |
| Exp2       | Traditional Only             | 90.2          | 89.4         |
| Exp3       | Point-E Only                 | 88.6          | 87.9         |
| Exp4       | Baseline + Traditional       | 91.3          | 90.6         |
| Exp5       | Baseline + Point-E           | 90.8          | 90.1         |
| Exp6       | All Combined                 | **92.6**      | **91.9**     |

> 🔎 *Note*: Generalization tests were limited to ModelNet40. Cross-domain evaluation on unseen datasets (e.g., ScanObjectNN) is suggested for future work.

---

## 📁 Folder Structure

| Folder | Description |
|--------|-------------|
| `1_traditional_augmentation/` | Jupyter notebook for traditional augmentation (rotation, jittering, etc.) |
| `2_aigc_point_e_generation/` | Scripts and test samples generated using OpenAI's Point-E |
| `3_training_pointnet/`       | PointNet implementation, preprocessing, training & visualization |
| `4_evaluation_ablation/`     | Evaluation pipeline, metrics, ablation studies & plots |
| `Data/`                      | Folder with 6 experimental datasets (partial) |
| `report/`                    | Final PDF report submitted for course credit |
| `requirements.txt`           | Python package dependencies |

---

## 🔧 Setup & Usage

### 1. Install Requirements
```bash
pip install -r requirements.txt
````

### 2. Run Augmentations

* Traditional: `1_traditional_augmentation/Traditional_Augmentation.ipynb`
* Point-E: `2_aigc_point_e_generation/Generating_Image_PointE.py`

### 3. Train the Classifier

```bash
cd 3_training_pointnet/
python preprocess_and_train.py
```

### 4. Evaluate Results

```bash
cd 4_evaluation_ablation/
python evaluate_model.py
python analyze_and_plot_results.py
```

---

## 🙋‍♂️ Author Contribution

**Kim Junseok** (Team Leader)

* Led the **overall project planning and coordination**
* Designed and implemented the **AIGC-based augmentation pipeline using Point-E**
* Filtered and refined synthetic point clouds for semantic quality
* Proposed and executed **hybrid augmentation experiments** (Exp4–6)
* Led the ablation analysis, visualization, and contributed to the final report

> **Team Members**: Junseok Kim (Leader), Yekai, Xujiaheng, Gaohe
> **Instructor**: [Prof. LIU Zhen (刘震教授)](https://sds.cuhk.edu.cn/en/teacher/1951)
> **Institution**: School of Data Science, The Chinese University of Hong Kong, Shenzhen

---

## 📄 License

MIT License — for educational and research use.

