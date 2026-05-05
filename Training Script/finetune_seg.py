#!/usr/bin/env python3
"""
Fine-tune YOLO11-seg with KD-Pretrained Backbone
=================================================
Loads the backbone weights exported by pretrain_kd.py and fine-tunes
YOLO11x-seg on the labeled root cross-section dataset for:
  • Xylem 
  • Vascular Bundle
  • Total Root 

Supports data-fraction experiments (10%, 25%, 50%, 75%, 90%, 100%) to
reproduce the data-efficiency analysis from the paper.

Dataset expected format
-----------------------
YOLO polygon segmentation format (.txt labels with class + polygon vertices).
Default dataset: Data/root_dataset  (train/valid splits).

Usage
-----
  # Fine-tune with 100% of labeled data (default)
  python finetune_seg.py --pretrained out/kd_pretrain/exported_models/exported_last.pt

  # Fine-tune with 50% of labeled data
  python finetune_seg.py --pretrained out/kd_pretrain/exported_models/exported_last.pt --fraction 0.5

  # Run all fractions sequentially (reproduces paper Table)
  python finetune_seg.py --pretrained out/kd_pretrain/exported_models/exported_last.pt --all-fractions

  # Fine-tune from scratch (baseline, no KD pretraining)
  python finetune_seg.py --pretrained yolo11x-seg.pt --run-name scratch_baseline

Output
------
  runs/finetune/<run_name>/weights/best.pt   ← best checkpoint
  runs/finetune/<run_name>/weights/last.pt
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path

# ─────────────────────────── Paths ────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()

DEFAULT_PRETRAINED   = str(REPO_ROOT / "out" / "kd_pretrain" / "exported_models" / "exported_last.pt")
DEFAULT_DATA_DIR     = str(REPO_ROOT / "Data" / "root_dataset")
DEFAULT_OUT          = str(REPO_ROOT / "runs" / "finetune")
DEFAULT_EPOCHS       = 300

# Data fractions for ablation study (matches paper)
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]

CLASS_NAMES = ["Total root", "Vascular bundle", "Xylem"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO11-seg from KD-pretrained backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pretrained", default=DEFAULT_PRETRAINED,
        help="Path to KD-pretrained .pt (from pretrain_kd.py) or any YOLO .pt checkpoint"
    )
    p.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Root of labeled dataset (must contain train/ and valid/ subdirectories)"
    )
    p.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Output parent directory for training runs"
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="Fine-tuning epochs"
    )
    p.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of training data to use (0.1 to 1.0)"
    )
    p.add_argument(
        "--all-fractions", action="store_true",
        help="Run fine-tuning for all fractions: 10%%, 25%%, 50%%, 75%%, 90%%, 100%%"
    )
    p.add_argument(
        "--run-name", default=None,
        help="Custom run name; defaults to 'kd_frac<fraction>' or 'scratch_baseline'"
    )
    p.add_argument(
        "--device", default="0",
        help="Device(s) to train on: '0' (single GPU), '0,1' (2 GPUs)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible data splits"
    )
    return p.parse_args()


def make_data_yaml(train_imgs_dir: Path, val_imgs_dir: Path, out_yaml: Path) -> Path:
    """Write a YOLO data.yaml with absolute paths."""
    data = {
        "train": str(train_imgs_dir),
        "val":   str(val_imgs_dir),
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out_yaml.write_text(yaml.dump(data, default_flow_style=False))
    return out_yaml


def subsample_dataset(
    src_imgs: Path, src_labels: Path,
    dst_imgs: Path, dst_labels: Path,
    fraction: float, seed: int
):
    """
    Copy a random fraction of images+labels from src to dst directories.
    Used to reproduce data-efficiency experiments.
    """
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(list(src_imgs.glob("*.png")) + list(src_imgs.glob("*.jpg")))
    random.seed(seed)
    n = max(1, round(len(all_imgs) * fraction))
    selected = random.sample(all_imgs, n)

    for img_path in selected:
        shutil.copy2(img_path, dst_imgs / img_path.name)
        lbl_path = src_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_labels / lbl_path.name)

    print(f"  Subsampled {n}/{len(all_imgs)} images ({fraction*100:.0f}%)")
    return n


def run_finetune(
    pretrained_path: Path,
    data_yaml: Path,
    out_dir: str,
    run_name: str,
    epochs: int,
    device: str,
):
    from ultralytics import YOLO

    print(f"\n  Loading model: {pretrained_path}")
    model = YOLO(str(pretrained_path))

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        project=out_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        device=device,
    )

    best = Path(out_dir) / run_name / "weights" / "best.pt"
    print(f"  Done -> {best}")
    return best


def main():
    args = parse_args()

    pretrained = Path(args.pretrained)
    data_dir   = Path(args.data_dir)
    out_dir    = args.out

    # ── Validate ────────────────────────────────────────────────────────────
    if not pretrained.exists():
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained}\n"
            "Run pretrain_kd.py first, or pass --pretrained yolo11x-seg.pt for baseline."
        )
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_imgs   = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    val_imgs     = data_dir / "valid" / "images"

    if not train_imgs.exists():
        raise FileNotFoundError(f"Expected train images at: {train_imgs}")
    if not val_imgs.exists():
        raise FileNotFoundError(f"Expected val images at: {val_imgs}")

    fractions = FRACTIONS if args.all_fractions else [args.fraction]

    print("\n=== YOLO11-seg Fine-tuning ===")
    print(f"Pretrained : {pretrained}")
    print(f"Dataset    : {data_dir}")
    print(f"Epochs     : {args.epochs}")
    print(f"Device     : {args.device}")
    print(f"Fractions  : {[f'{f*100:.0f}%' for f in fractions]}")

    results_summary = {}

    for frac in fractions:
        frac_tag = f"{int(frac*100):03d}"
        run_name = args.run_name if (args.run_name and len(fractions) == 1) \
                   else f"kd_frac{frac_tag}"

        print(f"\n--- Fraction {frac*100:.0f}% | Run: {run_name} ---")

        if frac < 1.0:
            # Create a temporary subsampled dataset
            tmp_dir    = REPO_ROOT / "out" / "tmp_splits" / frac_tag
            sub_imgs   = tmp_dir / "images"
            sub_labels = tmp_dir / "labels"

            n = subsample_dataset(
                train_imgs, train_labels,
                sub_imgs, sub_labels,
                fraction=frac, seed=args.seed
            )

            data_yaml = tmp_dir / "data.yaml"
            make_data_yaml(sub_imgs, val_imgs, data_yaml)
        else:
            # Use full training set
            data_yaml = REPO_ROOT / "out" / "data_full.yaml"
            data_yaml.parent.mkdir(parents=True, exist_ok=True)
            make_data_yaml(train_imgs, val_imgs, data_yaml)
            print(f"  Using full training set ({len(list(train_imgs.glob('*.png')) + list(train_imgs.glob('*.jpg')))} images)")

        best_pt = run_finetune(
            pretrained_path=pretrained,
            data_yaml=data_yaml,
            out_dir=out_dir,
            run_name=run_name,
            epochs=args.epochs,
            device=args.device,
        )
        results_summary[f"{frac*100:.0f}%"] = str(best_pt)

        # Clean up temp split
        if frac < 1.0:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n=== Fine-tuning Summary ===")
    for frac_label, path in results_summary.items():
        print(f"  {frac_label:>5s}  ->  {path}")


if __name__ == "__main__":
    main()