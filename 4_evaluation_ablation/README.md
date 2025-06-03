
README.py
=========
This README explains the full evaluation pipeline for the 3D PointNet classification experiments.
It includes 3 major components:

1. ✅ build_test_dataset.py — Generate a standardized test dataset.
2. ✅ evaluate_model.py — Run evaluation for pretrained PointNet models (Exp1~Exp6).
3. ✅ analyze_and_plot_results.py — Summarize and visualize the evaluation results.

---------------------------------------------------------------------------------------
1. Build Test Dataset (from ModelNet40 HDF5)
---------------------------------------------------------------------------------------

📄 Script: build_test_dataset.py

• Loads `test0.h5` and `test1.h5` from the ModelNet40 HDF5 dataset
• Reads class labels from `shape_names.txt`
• Extracts 25 samples per class → 40 classes → 1000 total test samples
• Saves each point cloud as `.npy` (normalized, centered, scaled) under:

    test_dataset/
        ├── airplane/
        │   ├── airplane_000.npy
        │   └── ...
        └── xbox/
            └── xbox_024.npy

📌 Output shape per sample: (1024, 3), dtype float32

🧪 Usage:
```bash
$ python build_test_dataset.py
````

---

2. Evaluate Trained Models on the Test Dataset

---

📄 Script: evaluate\_model.py

• Loads pretrained models (Exp1\~Exp6) from paths like:

* `XuJiaHeng/exp1_baseline/model_exp1_baseline.pth`
* ...

• For each experiment:
\- Loads the model
\- Runs inference on 1000 test samples
\- Computes overall Accuracy and Macro-F1
\- Saves predictions and metrics as a CSV file:

```
evaluation/test_result_exp{N}.csv
    ├── id,true_label,pred_label
    ├── ...
    └── Accuracy,0.92
        F1_score,0.90
```

🧪 Usage:

```bash
$ python evaluate_model.py
```

---

3. Visualize Accuracy & F1-score across Experiments

---

📄 Script: analyze\_and\_plot\_results.py

• Loads the `test_result_exp*.csv` files under `evaluation/`
• Extracts Accuracy and F1-score per experiment
• Draws bar chart comparing performance across Exp1\~Exp6
• Saves chart as `acc_f1_bar_chart.png`
• Prints performance summary as a table

🧪 Usage:

```bash
$ python analyze_and_plot_results.py
```

📊 Example Output:

```
       Accuracy  F1-score
Exp1     0.9020     0.8901
Exp2     0.9155     0.9023
Exp3     0.9287     0.9152
...
```

---

## 🧩 Dependencies

* numpy
* pandas
* matplotlib
* h5py
* torch
* tqdm
* scikit-learn (for evaluation metrics)

📦 Install with:

```bash
pip install numpy pandas matplotlib h5py torch tqdm scikit-learn
```

---
## 📁 Folder Structure

```
4_Evaluate_Ablation_Study/
├── evaluation/
│   ├── test_result_exp1.csv
│   ├── test_result_exp2.csv
│   ├── test_result_exp3.csv
│   ├── test_result_exp4.csv
│   ├── test_result_exp5.csv
│   ├── test_result_exp6.csv
│   └── acc_f1_bar_chart.png                # Bar chart comparing accuracy and F1-score
├── analyze_and_plot_results.py             # Plot accuracy/F1 results from evaluation CSVs
├── build_test_dataset.py                   # Generate test dataset from ModelNet40 HDF5
├── evaluate_model.py                       # Run inference and export evaluation results
├── Exp1-6--Loss&Accuracy.png               # Training curves for all experiments
├── pointnet_evaluation_report.pdf          # Final report document (PDF)
├── READ.ME.py                              # Python-style documentation of the project
└── readme.txt                              # Plain text version of README
```

---

## 📌 Notes

* The test dataset generation is random (sampling 25 per class), but repeatable if you fix seed.
* You can modify the number of samples per class or evaluation metrics as needed.
* Ensure all .pth model files exist before running evaluation.

"""

```

