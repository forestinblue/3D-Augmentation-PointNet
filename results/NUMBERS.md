# Numbers (single source of truth)

All numbers below are recomputed from the per-sample prediction files in
`4_evaluation_ablation/test_result_csv/test_result_exp{1..6}.csv` by
`results/extract_numbers.py` (output: `results/numbers.json`). They match the
`Accuracy` / `F1_score` footer lines of those CSVs and
`4_evaluation_ablation/pointnet_evaluation_report.pdf` to 4 decimals.

## Setup

- **Data:** ModelNet40, 40 classes, 1024 points per shape.
- **Train:** 25 samples per class per source (1,000 per source). Sources: real
  (ModelNet40 train), geometric-augmented (Y-rotation U[-15°, 15°], scale U[0.9, 1.1],
  Gaussian noise σ = 0.02, random 10% point dropout; parameters from
  `1_traditional_augmentation/Traditional_Augmentation.ipynb`), Point-E text-to-3D
  (10 prompts per class, 25 samples per class, all kept — no quality filter).
- **Test:** 910 samples from the ModelNet40 test split (20 or 25 per class;
  classes with only 20 test shapes cap at 20). *The report and scripts say 1,000; the CSVs contain 910.*
- **Model/training:** PointNet, 50 epochs, Adam lr 1e-3, batch 32, seed 42.
  Exp1 is trained from scratch; **Exp2–6 fine-tune from the Exp1 checkpoint**
  for another 50 epochs (`3_training_pointnet/preprocess_and_train.py:534-540`).
- **Runs:** 1 run per configuration, 1 seed → no variance estimate.

## Results table (regenerated from CSVs)

| Exp | Training data (N train) | Test acc (%) | Macro-F1 (%) | Δ acc vs Exp1 (pts) | Mean per-class acc (%) | Final train acc (%) |
|---|---|---|---|---|---|---|
| 1 | Real only (1,000) | **66.6** | 65.7 | — | 66.8 | 87.1 |
| 2 | Geometric only (1,000) | **71.9** | 71.2 | +5.3 | 71.2 | 94.6 |
| 3 | Point-E only (1,000) | **9.1** | 6.8 | −57.5 | 8.9 | 90.3 |
| 4 | Real + geometric (2,000) | **73.1** | 72.0 | +6.5 | 72.7 | 94.1 |
| 5 | Real + Point-E (2,000) | **65.1** | 63.0 | −1.5 | 64.4 | 84.1 |
| 6 | Real + geometric + Point-E (3,000) | **69.0** | 68.6 | +2.4 | 68.7 | 89.0 |

Test N = 910 for every row. Final train acc = epoch-50 row of
`Data/exp*/train_log_exp*.csv`.

Classes improved / worsened / unchanged vs Exp1 (per-class test accuracy):
Exp2 22/16/2, Exp3 2/38/0, Exp4 23/10/7, Exp5 16/18/6, Exp6 22/16/2.

### Headline comparisons (geometric vs generative)

| Comparison | Geometric | Point-E | Gap (pts) |
|---|---|---|---|
| Alone (Exp2 vs Exp3) | 71.9 | 9.1 | +62.8 for geometric |
| Added to real data (Exp4 vs Exp5) | 73.1 | 65.1 | +8.0 for geometric |
| Baseline for reference (Exp1) | 66.6 | 66.6 | — |

Caveat for Exp3: it is the Exp1 model fine-tuned on synthetic data only, so its
collapse mixes Point-E's domain gap with forgetting of the real-data features.

## Per-class test accuracy (%)

N test = samples of that class in the 910-sample test set.

| Class | N test | Exp1 | Exp2 | Exp3 | Exp4 | Exp5 | Exp6 |
|---|---|---|---|---|---|---|---|
| airplane | 25 | 76 | 84 | 0 | 84 | 80 | 88 |
| bathtub | 25 | 68 | 76 | 0 | 88 | 80 | 64 |
| bed | 25 | 96 | 80 | 0 | 100 | 96 | 72 |
| bench | 20 | 85 | 60 | 25 | 70 | 35 | 60 |
| bookshelf | 25 | 12 | 92 | 0 | 72 | 96 | 68 |
| bottle | 25 | 92 | 88 | 0 | 84 | 64 | 72 |
| bowl | 20 | 100 | 70 | 0 | 100 | 100 | 80 |
| car | 25 | 44 | 92 | 0 | 100 | 88 | 80 |
| chair | 25 | 72 | 88 | 0 | 84 | 88 | 84 |
| cone | 20 | 90 | 90 | 20 | 80 | 90 | 85 |
| cup | 20 | 80 | 60 | 0 | 50 | 85 | 50 |
| curtain | 20 | 45 | 65 | 5 | 65 | 20 | 40 |
| desk | 25 | 52 | 76 | 4 | 56 | 32 | 52 |
| door | 20 | 75 | 70 | 0 | 85 | 85 | 95 |
| dresser | 25 | 36 | 76 | 60 | 56 | 76 | 56 |
| flower_pot | 20 | 60 | 15 | 10 | 45 | 45 | 30 |
| glass_box | 25 | 92 | 48 | 0 | 68 | 56 | 56 |
| guitar | 25 | 80 | 88 | 88 | 96 | 88 | 96 |
| keyboard | 20 | 75 | 95 | 0 | 95 | 90 | 95 |
| lamp | 20 | 45 | 60 | 0 | 45 | 50 | 55 |
| laptop | 20 | 95 | 90 | 5 | 95 | 85 | 100 |
| mantel | 25 | 100 | 84 | 0 | 100 | 88 | 76 |
| monitor | 25 | 84 | 80 | 40 | 72 | 88 | 92 |
| night_stand | 25 | 60 | 48 | 8 | 36 | 44 | 64 |
| person | 20 | 85 | 75 | 0 | 90 | 45 | 95 |
| piano | 25 | 40 | 56 | 20 | 52 | 32 | 80 |
| plant | 25 | 28 | 48 | 12 | 40 | 12 | 48 |
| radio | 20 | 35 | 50 | 20 | 45 | 45 | 50 |
| range_hood | 25 | 100 | 96 | 0 | 100 | 96 | 88 |
| sink | 20 | 40 | 55 | 0 | 60 | 55 | 50 |
| sofa | 25 | 76 | 76 | 0 | 92 | 72 | 84 |
| stairs | 20 | 50 | 40 | 30 | 35 | 25 | 55 |
| stool | 20 | 75 | 80 | 0 | 75 | 75 | 70 |
| table | 25 | 44 | 80 | 0 | 52 | 68 | 44 |
| tent | 20 | 65 | 75 | 0 | 75 | 20 | 45 |
| toilet | 25 | 96 | 100 | 0 | 96 | 96 | 100 |
| tv_stand | 25 | 56 | 64 | 0 | 80 | 56 | 44 |
| vase | 25 | 36 | 60 | 0 | 60 | 44 | 68 |
| wardrobe | 20 | 70 | 50 | 0 | 65 | 50 | 50 |
| xbox | 20 | 60 | 70 | 10 | 65 | 35 | 65 |

With 20–25 test samples per class, one shape is 4–5 pts; single-class
differences are within noise, so the per-class table is descriptive only.

## Superseded numbers

The accuracy table in the README of commit `7714863` (87.4–92.6 %) is not
reproducible from any evaluation output in this repo and is superseded by the
table above.

## Figure

`results/accuracy_plot.png` ← `results/plot.py` (reads `results/numbers.json`).

`assets/hero.png` (README Figure 1) is exported from `assets/fig_src/hero.drawio`; its
thumbnails and result images come from `scripts/render_fig_assets.py`.
