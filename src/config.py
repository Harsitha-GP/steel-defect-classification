"""Central configuration. Change the dataset location in ONE place:
   - edit DATASET_ROOT below, or
   - set the environment variable NEU_DATASET_ROOT, or
   - pass --data_root on the command line (overrides both).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- single dataset root path ------------------------------------------------
DATASET_ROOT = Path(os.environ.get("NEU_DATASET_ROOT", PROJECT_ROOT / "data" / "NEU-CLS"))

# ---- output locations --------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"

# ---- reproducibility ---------------------------------------------------------
SEED = 42

# ---- dataset expectations (used only to WARN, never to assume) ---------------
EXPECTED_NUM_CLASSES = 6
EXPECTED_IMAGES_PER_CLASS = 300
EXPECTED_TOTAL = 1800
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Canonical class order and display names
CLASS_NAMES = ["rolled_in_scale", "patches", "crazing",
               "pitted_surface", "inclusion", "scratches"]
CLASS_DISPLAY = {
    "rolled_in_scale": "Rolled-in scale",
    "patches": "Patches",
    "crazing": "Crazing",
    "pitted_surface": "Pitted surface",
    "inclusion": "Inclusion",
    "scratches": "Scratches",
}

def ensure_dirs():
    for d in (FIGURE_DIR, CHECKPOINT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

# ---- Stage 2: preprocessing / split -------------------------------------------
IMAGE_SIZE = 224                      # input resolution (native images are 200x200)
SPLIT_RATIOS = (0.70, 0.15, 0.15)     # train / val / test
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
NUM_WORKERS = 0                       # 0 is safest on Windows
SPLIT_CSV = REPORT_DIR / "split.csv"
TRAIN_STATS_JSON = REPORT_DIR / "train_stats.json"
