"""Stage 2: preprocessing + leakage-safe stratified train/val/test split.

Run from project root:   python -m src.preprocessing --data_root "D:\\Downloads\\NEU-CLS"

Design decisions
* Split FIRST, on the original images. No augmentation is involved here (Stage 3 augments
  the training set only, on the fly).
* Stratified: every class is divided in the same 70/15/15 proportion.
* Exact duplicate files (same md5) are kept TOGETHER in one split, so a copy of a training
  image can never appear in the test set.
* Normalisation statistics (mean/std) are computed from the TRAINING split only.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config as C
from .dataset import SteelDefectDataset, discover_dataset, md5_of_file
from .utils import save_json, set_seed

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# --------------------------------------------------------------------------- split
def make_split(df, seed=C.SEED, ratios=C.SPLIT_RATIOS):
    """Return df with a 'split' column in {train,val,test}. df must contain 'md5','label'."""
    tr, va, te = ratios
    assert abs(tr + va + te - 1.0) < 1e-9, "ratios must sum to 1"
    conflict = df.groupby("md5")["label"].nunique()
    if (conflict > 1).any():
        raise ValueError("Identical files with different labels found - resolve manually.")

    # one representative per duplicate group -> split representatives -> propagate to copies
    reps = df.drop_duplicates("md5")[["md5", "label"]].reset_index(drop=True)
    rep_train, rep_tmp = train_test_split(
        reps, test_size=(va + te), stratify=reps["label"], random_state=seed)
    rep_val, rep_test = train_test_split(
        rep_tmp, test_size=te / (va + te), stratify=rep_tmp["label"], random_state=seed)
    mapping = {}
    for name, part in (("train", rep_train), ("val", rep_val), ("test", rep_test)):
        mapping.update({m: name for m in part["md5"]})
    out = df.copy()
    out["split"] = out["md5"].map(mapping)
    return out


def verify_split(df):
    """Hard checks against leakage. Raises AssertionError on failure."""
    assert df["split"].notna().all(), "unassigned images"
    assert not df["path"].duplicated().any(), "same file listed twice"
    sets = {s: set(df.loc[df["split"] == s, "md5"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]), "LEAK: content shared by train and val"
    assert not (sets["train"] & sets["test"]), "LEAK: content shared by train and test"
    assert not (sets["val"] & sets["test"]), "LEAK: content shared by val and test"
    assert df.groupby("split")["label"].nunique().eq(df["label"].nunique()).all(), \
        "a split is missing a class"
    return True


# --------------------------------------------------------------------- statistics
def _load_resized(path, size):
    with Image.open(path) as im:
        return im.convert("L").resize((size, size), Image.BILINEAR)


def compute_train_stats(train_df, size=C.IMAGE_SIZE):
    """Mean/std of pixel values in [0,1] over TRAINING images after resizing."""
    n, s1, s2 = 0, 0.0, 0.0
    for p in train_df["path"]:
        a = np.asarray(_load_resized(p, size), dtype=np.float64) / 255.0
        n += a.size; s1 += a.sum(); s2 += (a ** 2).sum()
    mean = s1 / n
    std = float(np.sqrt(s2 / n - mean ** 2))
    return {"mean": float(mean), "std": std, "n_train_images": int(len(train_df)),
            "image_size": int(size)}


# -------------------------------------------------------------------- transforms
class ReplicateChannels:
    """1xHxW tensor -> 3xHxW (picklable, unlike a lambda; needed on Windows workers)."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)

    def __repr__(self):
        return "ReplicateChannels(3)"


def normalization_steps(mode, stats=None):
    """Tensor-level steps shared by train and eval pipelines (after ToTensor).
    mode='custom'     -> 1 channel, normalised with TRAIN mean/std
    mode='pretrained' -> 3 channels (replicated), normalised with ImageNet mean/std
    """
    from torchvision import transforms as T
    if mode == "custom":
        if stats is None:
            raise ValueError("custom mode needs train stats")
        return [T.Normalize([stats["mean"]], [stats["std"]])]
    if mode == "pretrained":
        return [ReplicateChannels(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)]
    raise ValueError("mode must be 'custom' or 'pretrained'")


def get_eval_transform(mode, stats=None, size=C.IMAGE_SIZE):
    """Deterministic transform (no augmentation) for val, test and the no-aug baseline."""
    from torchvision import transforms as T
    return T.Compose([T.Resize((size, size)), T.ToTensor()] + normalization_steps(mode, stats))


# ------------------------------------------------------------------------- figure
def plot_split(df, out):
    ct = pd.crosstab(df["label"], df["split"]).reindex(C.CLASS_NAMES)[["train", "val", "test"]]
    ct.index = [C.CLASS_DISPLAY[c] for c in ct.index]
    ax = ct.plot(kind="bar", figsize=(9, 4.8), width=0.8,
                 color=sns.color_palette("deep")[:3])
    for cont in ax.containers:
        ax.bar_label(cont, fontsize=9)
    ax.set_ylabel("Number of images"); ax.set_xlabel("")
    ax.set_title("Stratified train / validation / test split")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(out, dpi=200); plt.close()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Stage 2: split + preprocessing")
    ap.add_argument("--data_root", type=str, default=str(C.DATASET_ROOT))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--image_size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--skip_tensor_check", action="store_true",
                    help="skip the PyTorch transform/DataLoader sanity check")
    args = ap.parse_args()

    C.ensure_dirs(); set_seed(args.seed)

    print("=" * 70, "\n1. LOAD + DUPLICATE HASHING\n", "=" * 70, sep="")
    df, rep = discover_dataset(args.data_root)
    print(f"Images labelled: {len(df)}")
    df["md5"] = df["path"].map(md5_of_file)
    print(f"Unique contents: {df['md5'].nunique()}  (duplicates kept together in one split)")

    print("\n" + "=" * 70, "\n2. STRATIFIED SPLIT\n", "=" * 70, sep="")
    print(f"Seed={args.seed}  ratios train/val/test = {C.SPLIT_RATIOS}")
    df = make_split(df, seed=args.seed)
    verify_split(df)
    print("Leakage checks PASSED (no content shared between splits, all classes in every split)\n")
    ct = pd.crosstab(df["label"], df["split"]).reindex(C.CLASS_NAMES)[["train", "val", "test"]]
    ct.index = [C.CLASS_DISPLAY[c] for c in ct.index]
    ct.loc["TOTAL"] = ct.sum()
    print(ct.to_string())
    prop = (ct.iloc[:-1].div(ct.iloc[:-1].sum(axis=1), axis=0) * 100).round(1)
    print("\nPer-class share of each split (%):\n", prop.to_string())

    df.drop(columns=["md5"]).assign(md5=df["md5"]).to_csv(C.SPLIT_CSV, index=False)
    save_json({"seed": args.seed, "ratios": C.SPLIT_RATIOS, "n_train": int((df.split == 'train').sum()),
               "n_val": int((df.split == 'val').sum()), "n_test": int((df.split == 'test').sum()),
               "method": "stratified, duplicate-group-aware, split before augmentation"},
              C.REPORT_DIR / "split_config.json")
    print("\nSaved:", C.SPLIT_CSV.relative_to(C.PROJECT_ROOT))

    print("\n" + "=" * 70, "\n3. NORMALISATION STATS (TRAIN SPLIT ONLY)\n", "=" * 70, sep="")
    stats = compute_train_stats(df[df["split"] == "train"], args.image_size)
    print(f"Train mean = {stats['mean']:.4f}   std = {stats['std']:.4f}   "
          f"(images: {stats['n_train_images']}, size: {stats['image_size']})")
    print(f"ImageNet mean/std for pretrained models: {C.IMAGENET_MEAN} / {C.IMAGENET_STD}")
    save_json(stats, C.TRAIN_STATS_JSON)
    print("Saved:", C.TRAIN_STATS_JSON.relative_to(C.PROJECT_ROOT))

    plot_split(df, C.FIGURE_DIR / "06_split_distribution.png")
    print("Saved: outputs\\figures\\06_split_distribution.png")

    if not args.skip_tensor_check:
        print("\n" + "=" * 70, "\n4. TENSOR / DATALOADER CHECK\n", "=" * 70, sep="")
        import torch
        from torch.utils.data import DataLoader
        train_df = df[df["split"] == "train"]
        for mode in ("custom", "pretrained"):
            tf = get_eval_transform(mode, stats, args.image_size)
            ds = SteelDefectDataset(train_df, tf)
            dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)
            x, y = next(iter(dl))
            print(f"[{mode:10s}] batch shape={tuple(x.shape)}  dtype={x.dtype}  "
                  f"min={x.min():.2f} max={x.max():.2f}  mean={x.mean():.3f} std={x.std():.3f}  "
                  f"labels in batch={sorted(set(y.tolist()))}")
        print("(custom-mode batch mean should be near 0 and std near 1 on training data)")

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
