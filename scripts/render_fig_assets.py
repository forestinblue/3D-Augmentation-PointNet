"""
Source images for the hero figure (assets/fig_src/hero.drawio).

  python scripts/render_fig_assets.py            # all
  python scripts/render_fig_assets.py samples    # point-cloud thumbnails only
  python scripts/render_fig_assets.py results    # results_bars.png + results_table.png only

Inputs are only files already in the repo:
  - Data/exp1_baseline/airplane/airplane_000{0..3}.npy     real ModelNet40 (after preprocessing)
  - Data/exp2_traditional/airplane/airplane_0000.npy        stored geometric augmentation of airplane_0000
  - Data/exp3_pointE/airplane/airplane_00{12,00}.npy        stored Point-E samples (no filter was used)
  - results/NUMBERS.md                                      accuracies ("Results table")
Real samples are the first files by index. Point-E #12 (airplane-shaped) and #0
(not airplane-shaped) were chosen by eye as one success and one failure; about 6 of
the 25 stored Point-E airplanes are not airplane-shaped. Only the airplane class is
stored locally, so every thumbnail is an airplane.

Sizes are in draw.io canvas pixels (canvas 1400 px wide). Images are written at 3x
(thumbnails) or 2x (results) so that 1 matplotlib point == 1 canvas pixel.
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "fig_src")

GREY_LIGHT = "#BFBFBF"
GREY = "#7F7F7F"
ACCENT = "#1F5FAD"
INK = "#333333"
POINT = "#4D4D4D"

THUMB_PX = 110  # thumbnail cell size on the canvas
POINT_SIZE = 1.2  # scatter marker size (pt^2 at 3x), same for every thumbnail
ELEV, AZIM = 22, -58  # same camera for every thumbnail


def set_font():
    for name in ("Helvetica", "Arial"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            return name
        except ValueError:
            continue
    raise RuntimeError("Neither Helvetica nor Arial is available")


# ---------------------------------------------------------------- thumbnails
def render_cloud(pts, path):
    # Stored clouds are y-up (ModelNet40 convention); plot y on the vertical axis.
    # The same mapping is used for Point-E samples, so their orientation is shown as stored.
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig = plt.figure(figsize=(THUMB_PX / 72, THUMB_PX / 72), dpi=216)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.scatter(x, z, y, s=POINT_SIZE, c=POINT, depthshade=False, linewidths=0)
    ax.set_proj_type("ortho")
    ax.set_xlim(-0.72, 0.72), ax.set_ylim(-0.72, 0.72), ax.set_zlim(-0.72, 0.72)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def make_samples():
    load = lambda p: np.load(os.path.join(ROOT, p)).astype(np.float32)  # noqa: E731
    jobs = {f"real_{i}.png": f"Data/exp1_baseline/airplane/airplane_{i:04d}.npy" for i in range(4)}
    jobs["geo_0.png"] = "Data/exp2_traditional/airplane/airplane_0000.npy"
    # Point-E: one airplane-shaped (#12) and one failed (#0) sample, chosen by eye from all 25
    jobs.update({f"pointe_{i}.png": f"Data/exp3_pointE/airplane/airplane_{i:04d}.npy" for i in (12, 0)})
    for name, src in jobs.items():
        pts = load(src)
        assert pts.shape == (1024, 3), (src, pts.shape)
        render_cloud(pts, os.path.join(OUT, name))
        print(f"{name:14s} <- {src}")


# ---------------------------------------------------------------- results
def read_numbers():
    """Rows of the 'Results table' in results/NUMBERS.md -> {exp: (data, n_train, acc)}."""
    text = open(os.path.join(ROOT, "results", "NUMBERS.md")).read()
    rows = {}
    for m in re.finditer(r"^\| (\d) \| (.+?) \((\d,\d{3})\) \| \*\*([\d.]+)\*\* \|", text, re.M):
        rows[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
    assert sorted(rows) == [1, 2, 3, 4, 5, 6], rows
    return rows


def make_bars(rows, w=380, h=280):
    bars = [(1, "Real only", GREY_LIGHT), (4, "+ Geometric", GREY), (5, "+ Point-E", ACCENT)]
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=144)
    ax = fig.add_axes([0.04, 0.2, 0.92, 0.78])
    for i, (exp, label, color) in enumerate(bars):
        acc = float(rows[exp][2])
        ax.bar(i, acc, width=0.62, color=color)
        ax.text(i, acc + 1.5, rows[exp][2], ha="center", va="bottom", fontsize=18, fontweight="bold", color=INK)
        ax.text(i, -4, f"{label}\n(Exp{exp})", ha="center", va="top", fontsize=15, color=INK, linespacing=1.15)
    ax.set_ylim(0, 84)
    ax.set_xlim(-0.55, 2.55)
    ax.axhline(0, color=GREY, lw=1.5)
    ax.set_axis_off()
    path = os.path.join(OUT, "results_bars.png")
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print(f"results_bars.png  {[rows[e][2] for e, _, _ in bars]}")


def make_table(rows, w=380, h=200):
    names = {1: "Real", 2: "Geometric", 3: "Point-E", 4: "Real + geometric",
             5: "Real + Point-E", 6: "Real + geo. + Point-E"}
    for e, n in names.items():  # names must describe the NUMBERS.md row
        src = rows[e][0].lower()
        assert all(k in src for k in n.lower().replace("geo.", "geometric").replace(" only", "").split(" + ")), (e, src)
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=144)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.set_axis_off()
    cols = [(4, "left", "Exp"), (48, "left", "Training data"), (298, "right", "N"), (376, "right", "Acc.")]
    row_h = (h - 4) / 7
    for x, ha, t in cols:
        ax.text(x, row_h * 0.5, t, ha=ha, va="center", fontsize=15, fontweight="bold", color=INK)
    ax.plot([0, w], [row_h, row_h], color=GREY, lw=1.5)
    for r, e in enumerate(range(1, 7), start=1):
        y = row_h * (r + 0.5)
        shown = e in (1, 4, 5)  # rows that appear as bars
        weight = "bold" if shown else "normal"
        color = ACCENT if e == 5 else INK
        for (x, ha, _), t in zip(cols, [str(e), names[e], rows[e][1], rows[e][2]]):
            ax.text(x, y, t, ha=ha, va="center", fontsize=15, fontweight=weight, color=color)
    path = os.path.join(OUT, "results_table.png")
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("results_table.png " + ", ".join(f"Exp{e}={rows[e][2]}" for e in range(1, 7)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "samples", "results"])
    what = ap.parse_args().what
    os.makedirs(OUT, exist_ok=True)
    print("font:", set_font())
    if what in ("all", "samples"):
        make_samples()
    if what in ("all", "results"):
        rows = read_numbers()
        make_bars(rows)
        make_table(rows)
