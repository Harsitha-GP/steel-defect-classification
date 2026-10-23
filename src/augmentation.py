"""Stage 3: data augmentation (TRAINING SET ONLY).

Run from project root:   python -m src.augmentation

Presets (compared later in the augmentation experiment)
  none  : resize + normalise only (baseline)
  basic : geometric only  - flips + small affine (rotation / translation / scale)
  full  : basic + mild brightness/contrast jitter + mild Gaussian noise

Why these, and why not others (justification for the report)
  Horizontal / vertical flip : a defect keeps its class under mirroring (a scratch is still a scratch).
  Small rotation (+/-10 deg) : strips are not always perfectly aligned with the camera; small angles
      add variety while keeping the overall layout.
  Small translation/scale    : defects appear at arbitrary positions and sizes. Kept small so thin
      defects (e.g. a narrow scratch near the edge) are rarely pushed out of the frame.
  Brightness/contrast (+/-20%): illumination varies strongly within classes (Stage 1). Kept modest because
      brightness also differs between classes (e.g. Pitted surface is brighter on average). Observed side
      effect: brightness increases can saturate bright images and wash out texture (Stage 3 figures);
      the basic-vs-full comparison will show whether this jitter helps.
  Gaussian noise (sigma<=0.03): simulates sensor noise. Kept small so fine textures (crazing cracks) survive.
  NOT used (yet): 90-degree rotations. Stage 3 figures show scratch-like defects in BOTH vertical and
      horizontal orientation, so orientation is not a fixed property of the data and 90-degree rotation
      cannot be ruled out on physical grounds. src/orientation_check.py measures orientation per class on
      TRAIN data; if orientation is mixed within classes, rot90 is a candidate to test on VALIDATION data.
  Also not used: large crops (may remove the defect), hue/saturation (grayscale), elastic/cutout/mixup
      (extra complexity without a clear benefit for this objective).
  Known artifact: rotation/translation leave gray borders that never occur in val/test images.
  Whether each component actually helps is decided by the experiments, not assumed here.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image

from . import config as C
from .dataset import SteelDefectDataset
from .preprocessing import get_eval_transform, normalization_steps
from .utils import set_seed

sns.set_theme(style="white", context="paper", font_scale=1.0)
PRESETS = ("none", "basic", "full")


class AddGaussianNoise:
    """Adds N(0, sigma^2) noise to a [0,1] tensor, sigma ~ U(0, max_std), with probability p."""
    def __init__(self, max_std=0.03, p=0.5):
        self.max_std, self.p = max_std, p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            sigma = torch.rand(1).item() * self.max_std
            x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
        return x

    def __repr__(self):
        return f"AddGaussianNoise(max_std={self.max_std}, p={self.p})"


def get_train_transform(preset, mode, stats, size=C.IMAGE_SIZE, for_display=False):
    """Training transform. If for_display=True, stops before normalisation (tensor in [0,1])."""
    from torchvision import transforms as T
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    steps = [T.Resize((size, size))]
    if preset != "none":
        fill = int(round(stats["mean"] * 255))          # neutral gray for rotated/shifted borders
        steps += [T.RandomHorizontalFlip(0.5),
                  T.RandomVerticalFlip(0.5),
                  T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=fill)]
    if preset == "full":
        steps.append(T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8))
    steps.append(T.ToTensor())
    if preset == "full":
        steps.append(AddGaussianNoise(max_std=0.03, p=0.5))
    if not for_display:
        steps += normalization_steps(mode, stats)
    return T.Compose(steps)


def build_transforms(preset, mode, stats, size=C.IMAGE_SIZE):
    """(train_transform, eval_transform). Val/test ALWAYS use the eval transform."""
    return get_train_transform(preset, mode, stats, size), get_eval_transform(mode, stats, size)


# -------------------------------------------------------------------- figures
def _pick(train_df, label, rng):
    sub = train_df[train_df["label"] == label]
    return sub.iloc[int(rng.randint(len(sub)))]["path"]


def fig_examples(train_df, stats, size, seed, out, n_aug=5):
    rng = np.random.RandomState(seed)
    tf = get_train_transform("full", "custom", stats, size, for_display=True)
    fig, axes = plt.subplots(len(C.CLASS_NAMES), n_aug + 1, figsize=(2.1 * (n_aug + 1), 2.2 * len(C.CLASS_NAMES)))
    for i, c in enumerate(C.CLASS_NAMES):
        img = Image.open(_pick(train_df, c, rng)).convert("L")
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"{C.CLASS_DISPLAY[c]}\n(original)", fontsize=9, loc="left")
        for j in range(1, n_aug + 1):
            axes[i, j].imshow(tf(img).squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(f"augmented #{j}", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    plt.suptitle("Random augmentation examples ('full' preset, training images only)", y=1.0)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()


def fig_individual(train_df, stats, size, seed, out, classes=("scratches", "pitted_surface")):
    import torchvision.transforms.functional as TF
    rng = np.random.RandomState(seed)
    fill = int(round(stats["mean"] * 255))
    names = ["Original", "H-flip", "V-flip", "Rotate +10\u00b0", "Shift + scale",
             "Brightness x1.2", "Contrast x1.2", "Noise \u03c3=0.03", "Rotate 90\u00b0\n(not used yet)"]
    fig, axes = plt.subplots(len(classes), len(names), figsize=(2.0 * len(names), 2.4 * len(classes)))
    for r, c in enumerate(classes):
        base = Image.open(_pick(train_df, c, rng)).convert("L").resize((size, size), Image.BILINEAR)
        arr = np.asarray(base, dtype=np.float32) / 255.0
        noisy = Image.fromarray((np.clip(arr + rng.randn(*arr.shape) * 0.03, 0, 1) * 255).astype(np.uint8))
        variants = [base, TF.hflip(base), TF.vflip(base),
                    TF.affine(base, angle=10, translate=(0, 0), scale=1.0, shear=0.0, fill=fill),
                    TF.affine(base, angle=0, translate=(int(0.05 * size), int(0.05 * size)), scale=1.1, shear=0.0, fill=fill),
                    TF.adjust_brightness(base, 1.2), TF.adjust_contrast(base, 1.2), noisy, TF.rotate(base, 90)]
        for k, (v, nm) in enumerate(zip(variants, names)):
            ax = axes[r, k]; ax.imshow(v, cmap="gray", vmin=0, vmax=255); ax.axis("off")
            if r == 0:
                ax.set_title(nm, fontsize=9)
        axes[r, 0].text(-0.05, 0.5, C.CLASS_DISPLAY[c], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
    plt.suptitle("Individual augmentation operations", y=1.0)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Stage 3: augmentation demo + checks")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    C.ensure_dirs(); set_seed(args.seed)

    if not C.SPLIT_CSV.exists() or not C.TRAIN_STATS_JSON.exists():
        raise SystemExit("Run Stage 2 first (python -m src.preprocessing ...) - split.csv / train_stats.json missing.")
    df = pd.read_csv(C.SPLIT_CSV)
    stats = json.load(open(C.TRAIN_STATS_JSON))
    size = int(stats["image_size"])
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    assert (train_df["split"] == "train").all()
    print("=" * 70, "\n1. AUGMENTATION POLICY\n", "=" * 70, sep="")
    print(f"Augmentation source: TRAINING split only ({len(train_df)} images). "
          f"Val ({(df.split=='val').sum()}) and test ({(df.split=='test').sum()}) use the deterministic eval transform.")
    for p in PRESETS:
        print(f"\n[{p}] train transform:")
        print(get_train_transform(p, "custom", stats, size))
    print("\nEval transform (val/test, and the 'none' baseline):")
    print(get_eval_transform("custom", stats, size))

    print("\n" + "=" * 70, "\n2. FIGURES\n", "=" * 70, sep="")
    fig_examples(train_df, stats, size, args.seed, C.FIGURE_DIR / "07_augmentation_examples.png")
    fig_individual(train_df, stats, size, args.seed, C.FIGURE_DIR / "08_augmentation_individual.png")
    print("Saved: outputs\\figures\\07_augmentation_examples.png")
    print("Saved: outputs\\figures\\08_augmentation_individual.png")

    print("\n" + "=" * 70, "\n3. BATCH CHECK (shuffle=True, so batches are class-mixed)\n", "=" * 70, sep="")
    from torch.utils.data import DataLoader
    for preset in ("none", "full"):
        for mode in ("custom", "pretrained"):
            tf, _ = build_transforms(preset, mode, stats, size)
            dl = DataLoader(SteelDefectDataset(train_df, tf), batch_size=C.BATCH_SIZE,
                            shuffle=True, num_workers=C.NUM_WORKERS)
            x, y = next(iter(dl))
            print(f"[{preset:5s}|{mode:10s}] shape={tuple(x.shape)} min={x.min():.2f} max={x.max():.2f} "
                  f"mean={x.mean():.3f} std={x.std():.3f} classes_in_batch={len(set(y.tolist()))}")
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
