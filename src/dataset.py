"""Dataset discovery. Does NOT assume a directory layout.

Supported layouts (auto-detected, per image):
  A) class sub-folders:          root/crazing/img1.bmp
  B) flat with class prefix:     root/crazing_1.bmp , root/Cr_1.bmp
  C) any nesting of A/B, e.g.    root/train/crazing/..., root/images/...
Pre-existing train/val folders are MERGED; we do our own stratified split later.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, IMAGE_EXTENSIONS


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


_ALIASES = {
    "rolled_in_scale": "rolled_in_scale", "rolled_in": "rolled_in_scale",
    "rolledin_scale": "rolled_in_scale", "rolledinscale": "rolled_in_scale",
    "rs": "rolled_in_scale",
    "patches": "patches", "patch": "patches", "pa": "patches",
    "crazing": "crazing", "cr": "crazing",
    "pitted_surface": "pitted_surface", "pitted": "pitted_surface",
    "ps": "pitted_surface",
    "inclusion": "inclusion", "in": "inclusion",
    "scratches": "scratches", "scratch": "scratches", "sc": "scratches",
}
_PREFIX_RE = re.compile(r"^([A-Za-z_\-\s]+?)[_\-\s]*\d+")


def _label_from_path(path: Path, root: Path):
    rel = path.relative_to(root)
    for part in reversed(rel.parts[:-1]):          # nearest folder first
        key = _norm(part)
        if key in _ALIASES:
            return _ALIASES[key], "folder"
    m = _PREFIX_RE.match(path.stem)
    if m:
        key = _norm(m.group(1))
        if key in _ALIASES:
            return _ALIASES[key], "filename"
    return None, None


def print_structure(root: Path, max_depth: int = 3) -> None:
    """Print directory tree with image counts per folder."""
    root = Path(root)

    def n_imgs(d):
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

    def walk(d, depth, prefix=""):
        subdirs = sorted(p for p in d.iterdir() if p.is_dir())
        print(f"{prefix}{d.name}/  [{n_imgs(d)} images directly inside]")
        if depth >= max_depth:
            if subdirs:
                print(f"{prefix}  ... ({len(subdirs)} sub-folders not shown)")
            return
        for s in subdirs:
            walk(s, depth + 1, prefix + "  ")

    walk(root, 0)


def discover_dataset(root):
    """Return (DataFrame, report_dict). Raises FileNotFoundError if root is missing."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. Set NEU_DATASET_ROOT, "
            f"edit src/config.py, or pass --data_root.")
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    rows, unmatched = [], []
    for p in files:
        label, src = _label_from_path(p, root)
        if label is None:
            unmatched.append(str(p.relative_to(root)))
            continue
        rows.append({"path": str(p), "filename": p.name, "label": label,
                     "label_idx": CLASS_NAMES.index(label), "label_source": src,
                     "extension": p.suffix.lower(),
                     "parent_folder": p.parent.name})
    df = pd.DataFrame(rows)
    report = {"root": str(root), "n_image_files_found": len(files),
              "n_labelled": len(df), "n_unmatched": len(unmatched),
              "unmatched_examples": unmatched[:10]}
    return df, report


def md5_of_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SteelDefectDataset:
    """PyTorch-style Dataset over a DataFrame with columns 'path' and 'label_idx'.
    Always loads the image as single-channel grayscale ('L').
    The transform decides 1-channel (custom CNN) vs 3-channel (pretrained) output.
    """

    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            img = im.convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]
