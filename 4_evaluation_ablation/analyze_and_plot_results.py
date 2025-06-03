#!/usr/bin/env python3
"""
analyze_and_plot_results.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load evaluation CSVs from experiments Exp1–Exp6, extract Accuracy and F1-score,
generate a bar chart comparison, and print a summary table.

Usage:
$ python analyze_and_plot_results.py

Dependencies:
- pandas
- matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Configuration for file paths
# -----------------------------
base_dir = Path("./evaluation")                  # Directory where evaluation CSVs are stored
csv_template = "test_result_exp{}.csv"           # Filename format for each experiment
plot_output = "acc_f1_bar_chart.png"             # Output image filename

# -----------------------------
# Read evaluation results from CSV files
# -----------------------------
results = {}
for i in range(1, 7):  # Iterate over Exp1 to Exp6
    path = base_dir / csv_template.format(i)
    try:
        df = pd.read_csv(path, skip_blank_lines=False)
        acc = float(df.iloc[-2, 1])              # Accuracy: second-to-last row, second column
        f1 = float(df.iloc[-1, 1])               # F1-score: last row, second column
        results[f"Exp{i}"] = {"Accuracy": acc, "F1-score": f1}
    except Exception as e:
        print(f"⚠️ Failed to process Exp{i}: {e}")
        continue

# -----------------------------
# Create summary table and plot bar chart
# -----------------------------
df_results = pd.DataFrame(results).T             # Transpose for experiment-wise rows
df_results = df_results.sort_index()             # Sort by experiment number

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
df_results.plot(kind="bar", ax=ax)
ax.set_title("Test Accuracy and F1-score Across Experiments")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(True, axis="y")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(plot_output)
plt.show()

print(f"\n✅ Plot saved as: {plot_output}")

# -----------------------------
# Print summary table in terminal
# -----------------------------
print("\n📊 Summary Table:")
print(df_results.to_string(float_format="%.4f"))
