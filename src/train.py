"""Stage 5: reusable training pipeline (train + validation ONLY - the test set is never loaded here).

Examples (run from the project root):
  python -m src.train --model custom_cnn --aug full --smoke
  python -m src.train --model custom_cnn --aug full --epochs 40 --exp_name custom_full
  python -m src.train --model resnet18 --strategy frozen   --aug full --exp_name resnet18_frozen_full
  python -m src.train --model efficientnet_b0 --strategy finetune --aug full

Model selection: the checkpoint with the best VALIDATION metric (default: lowest val loss).
"""
import argparse
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import json

from . import config as C
from .augmentation import build_transforms
from .dataset import SteelDefectDataset
from .models import MODEL_INPUT_MODE, TransferNet, build_model, count_parameters
from .utils import get_environment_info, save_json, set_seed

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
EXP_DIR = C.PROJECT_ROOT / "experiments"
RESULTS_VAL_CSV = EXP_DIR / "results_val.csv"


@dataclass
class TrainConfig:
    exp_name: str = ""
    model: str = "custom_cnn"
    strategy: Optional[str] = None          # 'frozen' | 'finetune' for transfer models
    aug: str = "full"                       # 'none' | 'basic' | 'full'
    epochs: int = 40
    batch_size: int = C.BATCH_SIZE
    optimizer: str = "adamw"
    lr: float = 1e-3                        # new head / whole custom CNN
    backbone_lr: float = 1e-4               # unfrozen pretrained layers
    weight_decay: float = 1e-2
    dropout: Optional[float] = None         # None -> model default (custom 0.4, transfer 0.2)
    scheduler: str = "plateau"              # 'plateau' | 'cosine'
    sched_patience: int = 3
    sched_factor: float = 0.5
    early_stop_patience: int = 10
    monitor: str = "val_loss"               # 'val_loss' | 'val_acc'
    seed: int = C.SEED
    image_size: int = C.IMAGE_SIZE
    num_workers: int = C.NUM_WORKERS
    pretrained: bool = True
    smoke: bool = False


def get_device():
    if torch.cuda.is_available():
        print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Device: CPU (no CUDA GPU detected)")
    return torch.device("cpu")


def _seed_worker(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s); random.seed(s)


def make_optimizer(model, cfg):
    if isinstance(model, TransferNet):
        head, backbone = model.parameter_groups()
        groups = [{"params": head, "lr": cfg.lr}]
        if backbone:
            groups.append({"params": backbone, "lr": cfg.backbone_lr})
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr}]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, weight_decay=cfg.weight_decay)
    raise ValueError("optimizer must be 'adamw' or 'adam'")


def make_scheduler(optimizer, cfg):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.sched_factor, patience=cfg.sched_patience)
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    raise ValueError("scheduler must be 'plateau' or 'cosine'")


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    t0, n, loss_sum, correct, nb = time.time(), 0, 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        n += bs; nb += 1
        loss_sum += loss.item() * bs
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / n, correct / n, time.time() - t0, nb


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    t0, n, loss_sum, nb = time.time(), 0, 0.0, 0
    ys, ps = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        n += y.size(0); nb += 1
        ys.append(y.cpu().numpy()); ps.append(out.argmax(1).cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return {"loss": loss_sum / n, "acc": float((ys == ps).mean()),
            "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
            "seconds": time.time() - t0, "nb": nb}


def plot_curves(hist, best_epoch, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (a, b, yl) in zip(axes, [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy")]):
        ax.plot(hist["epoch"], hist[a], label="train", lw=1.8)
        ax.plot(hist["epoch"], hist[b], label="validation", lw=1.8)
        ax.axvline(best_epoch, ls="--", color="gray", lw=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.set_title(f"{yl} - {title}"); ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def append_result(row):
    EXP_DIR.mkdir(exist_ok=True)
    new = pd.DataFrame([row])
    if RESULTS_VAL_CSV.exists():
        old = pd.read_csv(RESULTS_VAL_CSV)
        old = old[old["exp_name"] != row["exp_name"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULTS_VAL_CSV, index=False)


def run_experiment(cfg: TrainConfig):
    if cfg.model != "custom_cnn" and cfg.strategy is None:
        raise SystemExit("Transfer models need --strategy frozen|finetune")
    if not cfg.exp_name:
        cfg.exp_name = f"{cfg.model}{'_' + cfg.strategy if cfg.strategy else ''}_{cfg.aug}"
    if cfg.smoke:
        cfg.epochs, cfg.exp_name = 2, cfg.exp_name + "_smoke"
    if not C.SPLIT_CSV.exists() or not C.TRAIN_STATS_JSON.exists():
        raise SystemExit("Run Stage 2 first (split.csv / train_stats.json missing).")

    C.ensure_dirs(); set_seed(cfg.seed)
    device = get_device()
    exp_dir = EXP_DIR / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.SPLIT_CSV)
    stats = json.load(open(C.TRAIN_STATS_JSON))
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)      # test rows are deliberately NOT used
    assert not (set(train_df["md5"]) & set(val_df["md5"])), "train/val content overlap"
    print(f"Experiment '{cfg.exp_name}': train={len(train_df)}  val={len(val_df)}  (test set not loaded)")

    mode = MODEL_INPUT_MODE[cfg.model]
    train_tf, eval_tf = build_transforms(cfg.aug, mode, stats, cfg.image_size)
    g = torch.Generator(); g.manual_seed(cfg.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(SteelDefectDataset(train_df, train_tf), batch_size=cfg.batch_size, shuffle=True,
                              generator=g, num_workers=cfg.num_workers, worker_init_fn=_seed_worker, pin_memory=pin)
    val_loader = DataLoader(SteelDefectDataset(val_df, eval_tf), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)

    model = build_model(cfg.model, cfg.strategy, cfg.pretrained, cfg.dropout).to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {cfg.model} {cfg.strategy or ''} | params total={total:,} trainable={trainable:,} | "
          f"input={mode} | aug={cfg.aug}")
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    save_json({"config": asdict(cfg), "environment": get_environment_info(cfg.seed), "device": str(device),
               "n_train": len(train_df), "n_val": len(val_df), "total_params": total,
               "trainable_params": trainable, "train_mean_std": [stats["mean"], stats["std"]]},
              exp_dir / "config.json")

    max_tr, max_va = (3, 2) if cfg.smoke else (None, None)
    ckpt_path = C.CHECKPOINT_DIR / f"{cfg.exp_name}_best.pt"
    maximize = cfg.monitor == "val_acc"
    best = -np.inf if maximize else np.inf
    best_epoch, bad, hist, best_row = 0, 0, [], None
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_sec, tr_nb = train_one_epoch(model, train_loader, criterion, optimizer, device, max_tr)
        va = evaluate(model, val_loader, criterion, device, max_va)
        metric = va["acc"] if maximize else va["loss"]
        improved = metric > best + 1e-4 if maximize else metric < best - 1e-4
        if improved:
            best, best_epoch, bad = metric, epoch, 0
            best_row = {"train_acc": tr_acc, "train_loss": tr_loss, "val_loss": va["loss"],
                        "val_acc": va["acc"], "val_macro_f1": va["macro_f1"]}
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
                        "train_stats": stats}, ckpt_path)
        else:
            bad += 1
        lr_now = optimizer.param_groups[0]["lr"]
        if cfg.scheduler == "plateau":
            scheduler.step(va["loss"])
        else:
            scheduler.step()
        hist.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va["loss"],
                     "val_acc": va["acc"], "val_macro_f1": va["macro_f1"], "lr": lr_now,
                     "seconds": tr_sec + va["seconds"]})
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f} | "
              f"lr {lr_now:.1e} | {tr_sec + va['seconds']:.0f}s {'*' if improved else ''}")
        if cfg.smoke and epoch == 1:
            est = tr_sec / tr_nb * len(train_loader) + va["seconds"] / va["nb"] * len(val_loader)
            print(f"  -> estimated time per FULL epoch on this machine: ~{est:.0f}s ({est / 60:.1f} min)")
        if bad >= cfg.early_stop_patience:
            print(f"Early stopping: no improvement in {cfg.monitor} for {bad} epochs.")
            break

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(exp_dir / "history.csv", index=False)
    minutes = (time.time() - t_start) / 60
    if cfg.smoke:
        print("\nSmoke test passed (pipeline runs end to end). Nothing was added to results_val.csv.")
        return None

    plot_curves(hist_df, best_epoch, cfg.exp_name, C.FIGURE_DIR / f"curves_{cfg.exp_name}.png")
    print("\n" + "=" * 70)
    print(f"BEST EPOCH {best_epoch} (selected on VALIDATION {cfg.monitor}) - {minutes:.1f} min total")
    print(f"  validation: loss {best_row['val_loss']:.4f} | acc {best_row['val_acc']:.4f} | "
          f"macro-F1 {best_row['val_macro_f1']:.4f}")
    print(f"  train (augmented, train mode): acc {best_row['train_acc']:.4f}")
    print(f"Checkpoint : {ckpt_path.relative_to(C.PROJECT_ROOT)}")
    print(f"Curves     : outputs\\figures\\curves_{cfg.exp_name}.png")
    print("NOTE: these are VALIDATION numbers. The test set has not been touched.")
    append_result({"exp_name": cfg.exp_name, "model": cfg.model, "strategy": cfg.strategy or "scratch",
                   "aug": cfg.aug, "dropout": cfg.dropout, "weight_decay": cfg.weight_decay,
                   "lr": cfg.lr, "backbone_lr": cfg.backbone_lr if cfg.strategy == "finetune" else None,
                   "best_epoch": best_epoch, "epochs_run": len(hist), "val_loss": round(best_row["val_loss"], 4),
                   "val_acc": round(best_row["val_acc"], 4), "val_macro_f1": round(best_row["val_macro_f1"], 4),
                   "train_acc_aug": round(best_row["train_acc"], 4), "trainable_params": trainable,
                   "minutes": round(minutes, 1), "seed": cfg.seed})
    return best_row


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 5: train a model (train/val only)")
    ap.add_argument("--model", choices=list(MODEL_INPUT_MODE), default="custom_cnn")
    ap.add_argument("--strategy", choices=["frozen", "finetune"], default=None)
    ap.add_argument("--aug", choices=["none", "basic", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--monitor", choices=["val_loss", "val_acc"], default="val_loss")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--image_size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--num_workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--exp_name", type=str, default="")
    ap.add_argument("--smoke", action="store_true", help="2 tiny epochs to verify the pipeline + estimate speed")
    a = ap.parse_args()
    return TrainConfig(exp_name=a.exp_name, model=a.model, strategy=a.strategy, aug=a.aug, epochs=a.epochs,
                       batch_size=a.batch_size, optimizer=a.optimizer, lr=a.lr, backbone_lr=a.backbone_lr,
                       weight_decay=a.weight_decay, dropout=a.dropout, scheduler=a.scheduler,
                       early_stop_patience=a.early_stop_patience, monitor=a.monitor, seed=a.seed,
                       image_size=a.image_size, num_workers=a.num_workers,
                       pretrained=not a.no_pretrained, smoke=a.smoke)


if __name__ == "__main__":
    run_experiment(parse_args())
