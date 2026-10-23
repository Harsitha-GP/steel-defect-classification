"""Descriptive check: are structures in each class mostly vertical, horizontal or mixed?
Uses TRAIN split only (design decisions must not peek at val/test).
Run:  python -m src.orientation_check

Score per image = (sum gx^2 - sum gy^2) / (sum gx^2 + sum gy^2) from image gradients
   +1 -> dominant intensity change along x  => structures (lines/streaks) mostly VERTICAL
   -1 -> dominant intensity change along y  => structures mostly HORIZONTAL
    0 -> no preferred direction
Caveat: broad illumination bands also contribute, so this is descriptive, not a defect detector.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFilter

from . import config as C

THRESH = 0.2


def orientation_score(path):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(1)), dtype=np.float64) / 255.0
    gy, gx = np.gradient(g)
    jxx, jyy = (gx ** 2).sum(), (gy ** 2).sum()
    return float((jxx - jyy) / (jxx + jyy + 1e-12))


def main():
    C.ensure_dirs()
    if not C.SPLIT_CSV.exists():
        raise SystemExit("Run Stage 2 first (split.csv missing).")
    df = pd.read_csv(C.SPLIT_CSV)
    tr = df[df["split"] == "train"].copy()
    tr["score"] = tr["path"].map(orientation_score)
    tr["Class"] = tr["label"].map(C.CLASS_DISPLAY)

    rows = []
    for c in C.CLASS_NAMES:
        s = tr.loc[tr["label"] == c, "score"]
        rows.append({"Class": C.CLASS_DISPLAY[c], "n": len(s), "mean_score": s.mean(),
                     "pct_vertical (>+0.2)": 100 * (s > THRESH).mean(),
                     "pct_horizontal (<-0.2)": 100 * (s < -THRESH).mean(),
                     "pct_mixed": 100 * (s.abs() <= THRESH).mean()})
    out = pd.DataFrame(rows).set_index("Class")
    print("Orientation of structures per class (TRAIN split)\n")
    print(out.round(2).to_string())
    out.round(3).to_csv(C.REPORT_DIR / "orientation_by_class.csv")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    order = [C.CLASS_DISPLAY[c] for c in C.CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.stripplot(data=tr, x="Class", y="score", order=order, size=2.5, alpha=0.5, color="gray", ax=ax)
    sns.boxplot(data=tr, x="Class", y="score", order=order, showfliers=False,
                boxprops=dict(alpha=0.4), ax=ax)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(THRESH, ls="--", lw=0.6, color="k"); ax.axhline(-THRESH, ls="--", lw=0.6, color="k")
    ax.set_ylabel("Orientation score\n(-1 horizontal ... +1 vertical)"); ax.set_xlabel("")
    ax.set_title("Dominant structure orientation per image (training split)")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(C.FIGURE_DIR / "09_orientation_by_class.png", dpi=200); plt.close()
    print("\nSaved: outputs\\figures\\09_orientation_by_class.png, outputs\\reports\\orientation_by_class.csv")


if __name__ == "__main__":
    main()
