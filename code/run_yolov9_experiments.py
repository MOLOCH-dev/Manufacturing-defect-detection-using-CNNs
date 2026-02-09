#!/usr/bin/env python3
"""
Run all required YOLOv9 experiments using the OFFICIAL YOLOv9 repo (subprocess calls).

Experiments (epochs fixed):
  1) LR sweep at batch=16: lr0 = base_lr0 * [0.2, 1.0, 5.0]
  2) Batch sweep at lr=1x: batch = [8, 16, 32] with lr0 = base_lr0

It trains each unique (batch, lr0) combo once, then runs detect.py on TEST images and
saves prediction .txt files (with confidences) for FP/FN evaluation.

REQUIREMENTS:
- Git installed (if repo needs cloning)
- YOLOv9 repo compatible with train.py/detect.py + utils/general.py checks
- Your dataset data.yaml has a 'test:' entry that points to a test images folder.

USAGE EXAMPLE:
python run_yolov9_experiments.py \
  --weights /path/to/yolov9-m.pt \
  --cfg /path/to/yolov9/models/yolov9-m.yaml \
  --data /path/to/data.yaml \
  --project /path/to/yolo_v9_runs \
  --base_hyp /path/to/yolov9/data/hyps/hyp.scratch-high.yaml \
  --imgsz 768 --epochs 15 --device 0 --workers 8
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


# -----------------------------
# Helpers
# -----------------------------
def run(cmd, cwd=None):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def ensure_repo(repo_dir: Path, repo_url: str):
    """Clone YOLOv9 repo if not present."""
    if repo_dir.exists() and (repo_dir / "train.py").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(repo_dir)])

def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def resolve_path_from_data_yaml(data_yaml: Path, key: str) -> Path:
    """
    Resolve e.g. data['test'] to an absolute path.
    Accepts string path. If relative, resolve relative to data_yaml directory.
    """
    d = load_yaml(data_yaml)
    if key not in d:
        raise ValueError(f"data.yaml missing '{key}:' entry")
    v = d[key]
    if not isinstance(v, str):
        raise ValueError(f"data.yaml '{key}:' must be a string path, got {type(v)}")
    p = Path(v)
    if not p.is_absolute():
        p = (data_yaml.parent / p).resolve()
    return p

def make_hyp_with_lr(base_hyp: Path, out_hyp: Path, lr0: float):
    """Copy hyp yaml and override lr0."""
    hyp = load_yaml(base_hyp)
    hyp["lr0"] = float(lr0)
    out_hyp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_hyp, "w") as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

def pick_default_hyp(repo_dir: Path) -> Path:
    """
    Try common hyp locations if user didn't provide base_hyp.
    (Your repo uses data/hyps/*.yaml, but some forks use data/hyp*.yaml)
    """
    candidates = [
        repo_dir / "data" / "hyps" / "hyp.scratch-high.yaml",
        repo_dir / "data" / "hyps" / "hyp.scratch-med.yaml",
        repo_dir / "data" / "hyps" / "hyp.scratch-low.yaml",
        repo_dir / "data" / "hyp.scratch-high.yaml",
        repo_dir / "data" / "hyp.scratch-med.yaml",
        repo_dir / "data" / "hyp.scratch-low.yaml",
        repo_dir / "data" / "hyp.scratch.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find a default hyp yaml in the repo. "
        "Pass --base_hyp /path/to/hyp.yaml"
    )

def ensure_exists(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--project", type=str, required=True, help="Output directory for runs (outside repo OK)")
    ap.add_argument("--weights", type=str, required=True, help="Path to pretrained YOLOv9 .pt weights")
    ap.add_argument("--cfg", type=str, required=True, help="Path to YOLOv9 model YAML (e.g., models/yolov9-m.yaml)")

    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--repo_dir", type=str, default="./_ext/yolov9", help="Where YOLOv9 repo is / will be cloned")
    ap.add_argument("--repo_url", type=str, default="https://github.com/WongKinYiu/yolov9.git")

    ap.add_argument("--base_lr0", type=float, default=0.01, help="Baseline lr0 used for multipliers")
    ap.add_argument("--base_hyp", type=str, default="", help="Path to base hyp yaml. If empty, uses repo default.")
    ap.add_argument("--conf_pred", type=float, default=0.001, help="Low conf for saving preds; threshold later")
    ap.add_argument("--iou_pred", type=float, default=0.6)
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    project = Path(args.project).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    weights = Path(args.weights).resolve()
    cfg_yaml = Path(args.cfg).resolve()

    ensure_exists(data_yaml, "data.yaml")
    ensure_exists(weights, "weights (.pt)")
    ensure_exists(cfg_yaml, "cfg yaml")

    # Ensure repo is present
    ensure_repo(repo_dir, args.repo_url)
    ensure_exists(repo_dir / "train.py", "YOLOv9 train.py")
    ensure_exists(repo_dir / "detect.py", "YOLOv9 detect.py")

    # Base hyp
    if args.base_hyp:
        base_hyp = Path(args.base_hyp).resolve()
        ensure_exists(base_hyp, "base hyp yaml")
    else:
        base_hyp = pick_default_hyp(repo_dir)

    # Need test images directory for detect.py
    test_images_dir = resolve_path_from_data_yaml(data_yaml, "test")
    ensure_exists(test_images_dir, "test images dir")

    # Define experiments
    lr_mults = [0.2, 1.0, 5.0]
    batch_sizes = [8, 16, 32]

    run_specs = []
    # LR sweep with batch=16
    for m in lr_mults:
        run_specs.append(("lr_sweep", f"lr{m:g}_b16", 16, args.base_lr0 * m))
    # Batch sweep with lr=1x
    for b in batch_sizes:
        run_specs.append(("batch_sweep", f"lr1_b{b}", b, args.base_lr0 * 1.0))

    # Deduplicate baseline (batch=16, lr=base_lr0)
    seen = set()
    uniq_specs = []
    for group, name, batch, lr0 in run_specs:
        key = (int(batch), float(lr0))
        if key in seen:
            continue
        seen.add(key)
        uniq_specs.append((group, name, int(batch), float(lr0)))

    # Output dirs
    project.mkdir(parents=True, exist_ok=True)
    hyp_out_dir = project / "_yolov9_hyp"
    hyp_out_dir.mkdir(parents=True, exist_ok=True)

    # Run all experiments
    for group, name, batch, lr0 in uniq_specs:
        run_name = f"yolov9_{group}_{name}"
        print(f"\n===== YOLOv9 RUN: {run_name} | batch={batch} lr0={lr0} =====\n")

        # Create hyp for this run
        hyp_file = hyp_out_dir / f"{run_name}_hyp.yaml"
        make_hyp_with_lr(base_hyp, hyp_file, lr0)

        # Train command (IMPORTANT: pass --cfg)
        train_cmd = [
            sys.executable, "train.py",
            "--img", str(args.imgsz),
            "--batch", str(batch),
            "--epochs", str(args.epochs),
            "--data", str(data_yaml),
            "--weights", str(weights),
            "--cfg", str(cfg_yaml),
            "--hyp", str(hyp_file),
            "--device", str(args.device),
            "--workers", str(args.workers),
            "--project", str(project),
            "--name", run_name,
            "--exist-ok",
        ]
        run(train_cmd, cwd=repo_dir)

        # Locate best weights
        best_pt = project / run_name / "weights" / "best.pt"
        ensure_exists(best_pt, f"best.pt for {run_name}")

        # Predict on test set, save txt + conf
        pred_name = run_name + "_testpred"
        detect_cmd = [
            sys.executable, "detect.py",
            "--weights", str(best_pt),
            "--source", str(test_images_dir),
            "--img", str(args.imgsz),
            "--conf", str(args.conf_pred),
            "--iou", str(args.iou_pred),
            "--device", str(args.device),
            "--save-txt",
            "--save-conf",
            "--project", str(project),
            "--name", pred_name,
            "--exist-ok",
        ]
        run(detect_cmd, cwd=repo_dir)

        print(f"[OK] Completed: {run_name}")
        print(f"[OK] Test preds: {project / pred_name}")

    print("\nAll YOLOv9 experiments complete.\n")


if __name__ == "__main__":
    main()
