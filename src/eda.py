"""Stage 1: environment verification + dataset inspection + EDA.
Run from the project root:   python -m src.eda --data_root /path/to/NEU-CLS
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from . import config as C
from .dataset import discover_dataset, md5_of_file, print_structure
from .utils import get_environment_info, save_json, set_seed

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
DPI = 200


def disp(c):
    return C.CLASS_DISPLAY[c]


def analyse_images(df):
    """Read every image once; collect metadata + pixel histograms."""
    meta, hist_all = [], np.zeros(256, dtype=np.int64)
    hist_cls = {c: np.zeros(256, dtype=np.int64) for c in C.CLASS_NAMES}
    for r in df.itertuples():
        with Image.open(r.path) as im:
            mode, (w, h) = im.mode, im.size
            raw = np.array(im)
            gray = np.array(im.convert("L"))
        replicated = None
        if raw.ndim == 3 and raw.shape[2] >= 3:
            replicated = bool(np.all(raw[..., 0] == raw[..., 1]) and np.all(raw[..., 1] == raw[..., 2]))
        hb = np.bincount(gray.ravel(), minlength=256)
        hist_all += hb
        hist_cls[r.label] += hb
        g = gray.astype(np.float64) / 255.0
        meta.append({"path": r.path, "label": r.label, "width": w, "height": h,
                     "mode": mode, "channels_identical": replicated,
                     "mean": g.mean(), "std": g.std(), "min": int(gray.min()), "max": int(gray.max())})
    return pd.DataFrame(meta), hist_all, hist_cls


def plot_class_distribution(df, out):
    counts = df["label"].value_counts().reindex(C.CLASS_NAMES).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([disp(c) for c in counts.index], counts.values,
                  color=sns.color_palette("deep", len(counts)))
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", va="bottom")
    ax.set_ylabel("Number of images"); ax.set_title("Class distribution (NEU-CLS)")
    ax.set_ylim(0, max(counts.max() * 1.15, 1))
    plt.xticks(rotation=20); plt.tight_layout()
    fig.savefig(out, dpi=DPI); plt.close(fig)


def plot_dimensions(meta, out):
    combo = (meta["width"].astype(str) + "x" + meta["height"].astype(str)).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(combo.index, combo.values, color=sns.color_palette("deep")[0])
    for i, v in enumerate(combo.values):
        ax.text(i, v + 1, str(v), ha="center", va="bottom")
    ax.set_xlabel("Width x Height (pixels)"); ax.set_ylabel("Number of images")
    ax.set_title("Image dimensions"); plt.tight_layout()
    fig.savefig(out, dpi=DPI); plt.close(fig)


def plot_pixel_distribution(meta, hist_all, hist_cls, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].bar(np.arange(256), hist_all / hist_all.sum(), width=1.0, color="gray")
    axes[0].set_title("Overall pixel-intensity distribution")
    axes[0].set_xlabel("Gray level (0-255)"); axes[0].set_ylabel("Fraction of pixels")
    pal = sns.color_palette("deep", len(C.CLASS_NAMES))
    for c, col in zip(C.CLASS_NAMES, pal):
        h = hist_cls[c]
        if h.sum() > 0:
            axes[1].plot(np.arange(256), h / h.sum(), label=disp(c), color=col, lw=1.4)
    axes[1].set_title("Per-class intensity distribution")
    axes[1].set_xlabel("Gray level (0-255)"); axes[1].legend(fontsize=8)
    m = meta.copy(); m["Class"] = m["label"].map(disp)
    sns.boxplot(data=m, x="Class", y="mean", hue="Class", order=[disp(c) for c in C.CLASS_NAMES],
                hue_order=[disp(c) for c in C.CLASS_NAMES], ax=axes[2], palette=pal, legend=False)
    axes[2].set_title("Per-image mean intensity by class")
    axes[2].set_ylabel("Mean intensity (0-1)")
    axes[2].tick_params(axis="x", rotation=25)
    plt.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def plot_intensity_contrast(meta, out):
    m = meta.copy(); m["Class"] = m["label"].map(disp)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=m, x="mean", y="std", hue="Class", hue_order=[disp(c) for c in C.CLASS_NAMES],
                    s=18, alpha=0.7, ax=ax, palette="deep")
    ax.set_xlabel("Per-image mean intensity"); ax.set_ylabel("Per-image std (contrast)")
    ax.set_title("Global intensity vs contrast per image")
    plt.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def plot_samples(df, n, seed, out):
    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(len(C.CLASS_NAMES), n, figsize=(2.1 * n, 2.2 * len(C.CLASS_NAMES)))
    for i, c in enumerate(C.CLASS_NAMES):
        sub = df[df["label"] == c]
        picks = sub.iloc[rng.choice(len(sub), size=min(n, len(sub)), replace=False)] if len(sub) else sub
        for j in range(n):
            ax = axes[i, j]; ax.axis("off")
            if j < len(picks):
                ax.imshow(Image.open(picks.iloc[j]["path"]).convert("L"), cmap="gray", vmin=0, vmax=255)
            if j == 0:
                ax.set_title(disp(c), loc="left", fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def grayscale_note(meta) -> str:
    modes = meta["mode"].value_counts().to_dict()
    all_gray = list(modes.keys()) == ["L"] or (
        meta["channels_identical"].dropna().all() and meta["channels_identical"].notna().any()
        and set(modes) <= {"RGB", "RGBA"})
    txt = [f"# Grayscale input decision\n",
           f"Observed image modes: {modes}\n"]
    txt.append("The images carry a single intensity channel (stored as 'L', or as RGB with identical channels)."
               if all_gray else
               "WARNING: some images have colour information beyond a replicated gray channel - inspect before proceeding.")
    txt.append("""
## Recommendation
* **Custom CNN (trained from scratch): use a single-channel (1x H x W) input.** All colour
  channels would be identical copies, so 3 channels add no information, triple the first-layer
  weights and the memory for the input batch. Normalise with the dataset's own mean/std (computed
  on the TRAINING split only).
* **Pretrained ImageNet models (ResNet18, EfficientNet-B0): feed 3 channels** by replicating the gray
  channel (or, equivalently, sum conv1 weights over the RGB axis to make a 1-channel stem). Replication is
  simplest and keeps the pretrained filters exactly as trained; normalise with ImageNet mean/std
  because that is the input distribution those weights expect.
* Both choices are implemented later in preprocessing.py so the comparison stays fair.
""")
    return "\n".join(txt)


def main():
    ap = argparse.ArgumentParser(description="Stage 1: environment + dataset EDA")
    ap.add_argument("--data_root", type=str, default=str(C.DATASET_ROOT))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--samples_per_class", type=int, default=6)
    ap.add_argument("--skip_duplicates", action="store_true", help="skip md5 duplicate check")
    args = ap.parse_args()

    C.ensure_dirs(); set_seed(args.seed)

    # 1. environment ----------------------------------------------------------
    print("=" * 70, "\n1. ENVIRONMENT\n", "=" * 70, sep="")
    env = get_environment_info(args.seed)
    for k, v in env.items():
        print(f"  {k:22s}: {v}")
    save_json(env, C.REPORT_DIR / "environment.json")

    # 2. structure ------------------------------------------------------------
    root = Path(args.data_root)
    print("\n" + "=" * 70, "\n2. DATASET STRUCTURE\n", "=" * 70, sep="")
    print(f"Dataset root: {root.resolve()}")
    if not root.exists():
        raise SystemExit(f"ERROR: '{root}' does not exist. Use --data_root.")
    print_structure(root)

    df, rep = discover_dataset(root)
    print(f"\nImage files found : {rep['n_image_files_found']}")
    print(f"Labelled          : {rep['n_labelled']}")
    print(f"Unmatched files   : {rep['n_unmatched']}  {rep['unmatched_examples']}")
    if df.empty:
        raise SystemExit("No labelled images. Send me the structure printout above.")
    print("Label source      :", df["label_source"].value_counts().to_dict())
    print("Folders containing images:", df["parent_folder"].value_counts().to_dict())

    counts = df["label"].value_counts().reindex(C.CLASS_NAMES).fillna(0).astype(int)
    print("\nImages per class:")
    for c, n in counts.items():
        print(f"  {C.CLASS_DISPLAY[c]:16s}: {n}")
    warns = []
    if len(df) != C.EXPECTED_TOTAL: warns.append(f"total {len(df)} != expected {C.EXPECTED_TOTAL}")
    if df["label"].nunique() != C.EXPECTED_NUM_CLASSES: warns.append(f"{df['label'].nunique()} classes found")
    if (counts != C.EXPECTED_IMAGES_PER_CLASS).any(): warns.append("per-class counts differ from 300")
    print("\nSanity checks:", "ALL MATCH the expected NEU-CLS description." if not warns else "WARNINGS -> " + "; ".join(warns))

    # 3. duplicates -----------------------------------------------------------
    n_dup = None
    if not args.skip_duplicates:
        df["md5"] = df["path"].map(md5_of_file)
        dup = df[df.duplicated("md5", keep=False)]
        n_dup = int(len(dup))
        print(f"\nExact duplicate files (md5): {n_dup}")
        if n_dup:
            conflicting = dup.groupby("md5")["label"].nunique()
            print(f"  duplicate groups: {dup['md5'].nunique()}, groups with conflicting labels: {int((conflicting > 1).sum())}")
            print("  NOTE: duplicates across splits would cause leakage; we'll handle this in Stage 2.")

    # 4. image analysis -------------------------------------------------------
    print("\n" + "=" * 70, "\n3. IMAGE ANALYSIS (reading all images)\n", "=" * 70, sep="")
    meta, hist_all, hist_cls = analyse_images(df)
    meta_out = meta.copy()
    meta_out.to_csv(C.REPORT_DIR / "image_metadata.csv", index=False)
    print("Image modes      :", meta["mode"].value_counts().to_dict())
    print("Image dimensions :", (meta["width"].astype(str) + "x" + meta["height"].astype(str)).value_counts().to_dict())

    levels = np.arange(256)
    mean_all = float((hist_all * levels).sum() / hist_all.sum() / 255.0)
    std_all = float(np.sqrt(((levels / 255.0 - mean_all) ** 2 * hist_all).sum() / hist_all.sum()))
    print(f"\nDataset-wide pixel stats (scale 0-1): mean={mean_all:.4f}  std={std_all:.4f}")
    print("  (for reference only - final normalisation stats will use the TRAIN split)")

    stats = meta.groupby("label").agg(
        n=("mean", "size"), mean_intensity=("mean", "mean"), mean_contrast=("std", "mean"),
        min_pixel=("min", "min"), max_pixel=("max", "max")).reindex(C.CLASS_NAMES)
    stats.index = [disp(c) for c in stats.index]
    print("\nPer-class statistics:\n", stats.round(4).to_string())
    stats.to_csv(C.REPORT_DIR / "class_statistics.csv")

    save_json({"n_images": int(len(df)), "n_classes": int(df["label"].nunique()),
               "images_per_class": {k: int(v) for k, v in counts.items()},
               "image_modes": meta["mode"].value_counts().to_dict(),
               "dimensions": (meta["width"].astype(str) + "x" + meta["height"].astype(str)).value_counts().to_dict(),
               "global_mean_0_1": mean_all, "global_std_0_1": std_all,
               "pixel_min": int(meta["min"].min()), "pixel_max": int(meta["max"].max()),
               "n_duplicate_files": n_dup, "warnings": warns},
              C.REPORT_DIR / "dataset_stats.json")

    note = grayscale_note(meta)
    (C.REPORT_DIR / "grayscale_decision.md").write_text(note)

    # 5. figures --------------------------------------------------------------
    print("\n" + "=" * 70, "\n4. FIGURES\n", "=" * 70, sep="")
    plot_class_distribution(df, C.FIGURE_DIR / "01_class_distribution.png")
    plot_samples(df, args.samples_per_class, args.seed, C.FIGURE_DIR / "02_sample_images.png")
    plot_dimensions(meta, C.FIGURE_DIR / "03_image_dimensions.png")
    plot_pixel_distribution(meta, hist_all, hist_cls, C.FIGURE_DIR / "04_pixel_distribution.png")
    plot_intensity_contrast(meta, C.FIGURE_DIR / "05_intensity_vs_contrast.png")
    for f in sorted(C.FIGURE_DIR.glob("0[1-5]_*.png")):
        print("  saved", f.relative_to(C.PROJECT_ROOT))

    print("\n" + note)
    print("Stage 1 complete. Reports in", C.REPORT_DIR.relative_to(C.PROJECT_ROOT))


if __name__ == "__main__":
    main()
