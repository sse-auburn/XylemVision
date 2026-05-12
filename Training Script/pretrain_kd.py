"""
Knowledge-Distillation Pretraining: DINOv2 -> YOLO11-seg
=========================================================
Uses lightly-train to pretrain YOLO11x-seg backbone on unlabeled root images
with DINOv2 as the frozen teacher (distillationv2 method: global KL-div +
local token-to-token KL-div + FIFO memory queue, label-free).


Augmentations used by lightly-train (identical to the paper):
  • RandomResizedCrop ≥14% area
  • Random horizontal / vertical flip
  • ColorJitter (brightness, contrast, saturation, hue)
  • Random Grayscale
  • GaussianBlur
  • Normalize (ImageNet mean/std)
  • MixUp inside training step
  → No separate dataset augmentation needed; handled on-the-fly.

Epoch recommendation
--------------------
  • Paper used 20 000 epochs with ~1 500 images, batch 64 (~460k gradient steps).
  • This script defaults to 20 000 epochs.
  • With ~1 000 unlabeled images and batch 64 that is ≈300k steps (≈65% of paper).
  • For an exact step-count match use --epochs 30000.
  • Quick single-GPU test: --epochs 500 (finishes in minutes).
  • Full 2-GPU run: --epochs 20000 (recommended; matches the paper setting).

Usage
-----
  # Quick test (single GPU)
  python pretrain_kd.py --epochs 500 --devices 1

  # Full single-GPU training
  python pretrain_kd.py --epochs 20000 --devices 1

  # Full 2-GPU training (Windows, DDP)
  python pretrain_kd.py --epochs 20000 --devices 2

  # Override data paths
  python pretrain_kd.py --unlabeled-dirs "/path/to/unlabeled/data" "/path/to/extra"

Output
------
  out/kd_pretrain/
    exported_models/exported_last.pt   ← pass this to finetune_seg.py
    checkpoints/                       ← PyTorch Lightning checkpoints
    logs/                              ← training logs
"""

import argparse
import logging
import os
from pathlib import Path

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

# ─────────────────────────── Paths ────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()

# Default unlabeled image directories.
# Labeled images (used WITHOUT labels) can be included to maximize pretraining data.
DEFAULT_DATA_DIRS = [
    str(REPO_ROOT / "Unlabeled_data"),
]

DEFAULT_MODEL    = str(REPO_ROOT / "yolo11x-seg.pt")
DEFAULT_OUT      = str(REPO_ROOT / "out" / "kd_pretrain")
DEFAULT_EPOCHS   = 20000
DEFAULT_BATCH    = 64
DEFAULT_TEACHER  = "dinov2/vitl16"   # DINOv2 ViT-B/16, LVD-1689M pretrained
                                     # Alternatives (stronger but slower):
                                     #   "dinov2/vitl16"  — ViT-L/16
                                     #   "dinov2/vith16plus" — ViT-H+/16 (largest)
                                     #   "dinov2/vitb14"  — DINOv2 ViT-B/14 (paper default)


def parse_args():
    p = argparse.ArgumentParser(
        description="KD pretraining: DINOv2 teacher → YOLO11-seg student",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to YOLO11-seg .pt file (student backbone)"
    )
    p.add_argument(
        "--unlabeled-dirs", nargs="+", default=DEFAULT_DATA_DIRS,
        help="One or more directories containing unlabeled images for pretraining"
    )
    p.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Output directory for checkpoints and exported model"
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="Pretraining epochs. Paper used 20000. Quick test: 500."
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help="Global batch size (paper used 64)"
    )
    p.add_argument(
        "--teacher", default=DEFAULT_TEACHER,
        help="Teacher model string (dinov2/vitb16, dinov2/vitl16, dinov2/vitb14, etc.)"
    )
    p.add_argument(
        "--devices", type=int, default=2,
        help="Number of GPUs (1 for single-GPU, 2 for two-GPU distributed)"
    )
    p.add_argument(
        "--num-nodes", type=int, default=1,
        help="Number of machines (1 for single-node)"
    )
    p.add_argument(
        "--queue-size", type=int, default=None,
        help="FIFO memory queue size. 'None' = auto-scaled by dataset size "
             "(≤512 for ~1000 images). Paper-equivalent: 512."
    )
    p.add_argument(
        "--temperature", type=float, default=0.07,
        help="KL divergence temperature (paper default 0.07)"
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Resume from a prior lightly-train checkpoint"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted training run from the same --out directory"
    )
    p.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    p.add_argument(
        "--wandb-project", default="xylem_kd_pretrain",
        help="W&B project name"
    )
    return p.parse_args()


def validate_data_dirs(dirs):
    """Warn about missing directories, return only existing ones."""
    valid = []
    for d in dirs:
        p = Path(d)
        if p.exists() and p.is_dir():
            imgs = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.tif"))
            print(f"  [OK]  {d}  ({len(imgs)} images)")
            valid.append(d)
        else:
            print(f"  [SKIP] {d}  (not found)")
    return valid


def main():
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    args = parse_args()

    # ── Validate inputs ─────────────────────────────────────────────────────
    from ultralytics import YOLO
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"YOLO model not found: {model_path}")
        print("\nDownloading YOLO11-seg model …")
        model = YOLO("yolo11x-seg.pt")
    else:
        print("\nLoading YOLO11-seg model …")
        model = YOLO(str(model_path))


    print("\n=== KD Pretraining: DINOv2 -> YOLO11-seg ===")
    print(f"Teacher  : {args.teacher}")
    print(f"Student  : {args.model}")
    print(f"Epochs   : {args.epochs}")
    print(f"Batch    : {args.batch_size}")
    print(f"Devices  : {args.devices}")
    print(f"Output   : {args.out}\n")

    print("Checking data directories:")
    data_dirs = validate_data_dirs(args.unlabeled_dirs)
    if not data_dirs:
        raise RuntimeError("No valid data directories found.")
    total_imgs = sum(
        len(list(Path(d).glob("*.png")) + list(Path(d).glob("*.jpg")) + list(Path(d).glob("*.tif")))
        for d in data_dirs
    )
    print(f"\nTotal images for pretraining: {total_imgs}")

    # ── Build method_args ────────────────────────────────────────────────────
    method_args = {
        "teacher": args.teacher,
        "temperature_global": args.temperature,
        "temperature_local": args.temperature,
    }
    if args.queue_size is not None:
        method_args["queue_size"] = args.queue_size
    # queue_size=None → lightly-train auto-scales based on dataset size
    # For ~1000 images → auto sets queue_size=256

    # ── Build loggers ────────────────────────────────────────────────────────
    loggers = {"tensorboard": {}}
    if args.wandb:
        loggers["wandb"] = {"project": args.wandb_project}

    # ── Import and run lightly-train ─────────────────────────────────────────
    import lightly_train

    print("Starting KD pretraining …\n")
    lightly_train.pretrain(
        out=args.out,
        data=data_dirs,                  # list of image directories
        model=model,                     # YOLO11-seg student
        method="distillationv2",         # global KL-div + local patch KL-div + queue
        method_args=method_args,
        epochs=args.epochs,
        batch_size=args.batch_size,
        devices=1,                       # single GPU only — gloo DDP crashes on Windows + torch 2.11
        num_nodes=1,
        # LARS optimizer (matches paper: lr=0.3, wd=1e-6, momentum=0.9)
        optim="lars",
        optim_args={
            "lr": 0.3,
            "weight_decay": 1e-6,
            "momentum": 0.9,
        },
        loggers=loggers,
        checkpoint=args.checkpoint,
        resume_interrupted=args.resume,
        overwrite=True,
    )

    exported = Path(args.out) / "exported_models" / "exported_last.pt"
    print(f"\n[DONE] Pretraining complete.")
    print(f"   Exported model -> {exported}")
    print(f"\n   Next step -- fine-tune with:")
    print(f'   python finetune_seg.py --pretrained "{exported}"')


if __name__ == "__main__":
    main()