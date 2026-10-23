"""Reproducibility + environment helpers."""
import importlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_environment_info(seed=None) -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed": seed,
    }
    for pkg in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn",
                "PIL", "cv2", "torch", "torchvision"]:
        try:
            info[f"{pkg}_version"] = importlib.import_module(pkg).__version__
        except Exception:
            info[f"{pkg}_version"] = "NOT INSTALLED"
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_GB"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        else:
            info["gpu_name"] = None
    except Exception:
        info["cuda_available"] = False
        info["gpu_name"] = None
    return info


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
