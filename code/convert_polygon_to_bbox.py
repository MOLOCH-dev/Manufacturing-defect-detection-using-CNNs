import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ========================
# CONFIG (EDIT THESE)
# ========================
DATASET_ROOT = Path("/home/mmpug/Desktop/AIforManuf/project1/data/defect_detection_3d_printing_yolo")
SPLITS = ["train", "valid", "test"]

# Writes cleaned labels here first, then swaps into labels/ (in-place)
CLEAN_LABELS_SUBDIR = "labels_bbox_clean"

# Visualization output
VIZ_OUTDIR = DATASET_ROOT / "_viz_poly_to_bbox"
N_VIZ_SAMPLES = 12
random.seed(42)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ========================
# Helpers
# ========================
def find_image_for_stem(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def parse_line_to_bbox_or_polygon(parts: List[str]) -> Tuple[int, str, Optional[Tuple[float, float, float, float]], Optional[np.ndarray]]:
    """
    Returns:
      cls_id, kind in {"bbox","poly","invalid"},
      bbox=(xc,yc,w,h) if available,
      poly Nx2 if available (normalized coords)
    """
    if len(parts) < 2:
        return -1, "invalid", None, None

    try:
        cls = int(parts[0])
    except ValueError:
        return -1, "invalid", None, None

    # Standard YOLO bbox: cls xc yc w h
    if len(parts) == 5:
        try:
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            return cls, "invalid", None, None
        return cls, "bbox", (xc, yc, bw, bh), None

    # Polygon-like: cls x1 y1 x2 y2 ... (even # coords)
    if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
        try:
            coords = list(map(float, parts[1:]))
        except ValueError:
            return cls, "invalid", None, None
        xs = coords[0::2]
        ys = coords[1::2]
        poly = np.stack([xs, ys], axis=1)  # Nx2 normalized

        x_min, x_max = float(np.min(poly[:, 0])), float(np.max(poly[:, 0]))
        y_min, y_max = float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))

        x_min, x_max = clamp01(x_min), clamp01(x_max)
        y_min, y_max = clamp01(y_min), clamp01(y_max)

        bw = max(0.0, x_max - x_min)
        bh = max(0.0, y_max - y_min)
        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0

        return cls, "poly", (xc, yc, bw, bh), poly

    return cls, "invalid", None, None

def bbox_norm_to_xyxy_px(xc, yc, bw, bh, W, H):
    x1 = (xc - bw / 2) * W
    y1 = (yc - bh / 2) * H
    x2 = (xc + bw / 2) * W
    y2 = (yc + bh / 2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def poly_norm_to_pts_px(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    pts = np.zeros_like(poly, dtype=np.int32)
    pts[:, 0] = np.clip(np.round(poly[:, 0] * W), 0, W - 1).astype(np.int32)
    pts[:, 1] = np.clip(np.round(poly[:, 1] * H), 0, H - 1).astype(np.int32)
    return pts

def draw_polygon(img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    if len(pts) >= 2:
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=2)
        for p in pts:
            cv2.circle(out, tuple(p), 2, (0, 255, 255), -1)
    return out

def draw_bbox(img_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    out = img_bgr.copy()
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return out


# ========================
# Step 1: Convert labels -> write cleaned copies
# ========================
poly_records = []  # (split, label_path, image_path, poly_pts_norm, bbox_norm)

summary = {
    "bbox_lines": 0,
    "poly_lines": 0,
    "invalid_lines": 0,
    "files_with_polys": 0,
    "files_processed": 0,
}

for split in SPLITS:
    lbl_dir = DATASET_ROOT / split / "labels"
    img_dir = DATASET_ROOT / split / "images"

    if not lbl_dir.exists():
        print(f"[WARN] Missing labels dir: {lbl_dir}")
        continue

    out_lbl_dir = DATASET_ROOT / split / CLEAN_LABELS_SUBDIR
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        summary["files_processed"] += 1
        img_path = find_image_for_stem(img_dir, lbl_path.stem)

        out_lines = []
        file_had_poly = False

        raw_lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]

        for ln in raw_lines:
            parts = ln.split()
            cls, kind, bbox, poly = parse_line_to_bbox_or_polygon(parts)

            if bbox is None:
                summary["invalid_lines"] += 1
                continue

            xc, yc, bw, bh = bbox
            xc, yc, bw, bh = clamp01(xc), clamp01(yc), clamp01(bw), clamp01(bh)

            # Optional extra validity check
            if bw <= 0 or bh <= 0:
                summary["invalid_lines"] += 1
                continue

            out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if kind == "bbox":
                summary["bbox_lines"] += 1
            elif kind == "poly":
                summary["poly_lines"] += 1
                file_had_poly = True
                if img_path is not None and poly is not None:
                    poly_records.append((split, lbl_path, img_path, poly, (xc, yc, bw, bh)))
            else:
                summary["invalid_lines"] += 1

        if file_had_poly:
            summary["files_with_polys"] += 1

        (out_lbl_dir / lbl_path.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

print("\n=== Conversion summary ===")
print(f"files processed:         {summary['files_processed']}")
print(f"bbox lines kept:         {summary['bbox_lines']}")
print(f"polygon lines converted: {summary['poly_lines']}")
print(f"invalid lines skipped:   {summary['invalid_lines']}")
print(f"files containing polys:  {summary['files_with_polys']}")
print(f"Clean labels written to: <split>/{CLEAN_LABELS_SUBDIR}/\n")


# ========================
# Step 1b: MODIFY LABELS IN PLACE (with backup)
# ========================
print("=== In-place overwrite (with backup) ===")
for split in SPLITS:
    orig = DATASET_ROOT / split / "labels"
    clean = DATASET_ROOT / split / CLEAN_LABELS_SUBDIR
    backup = DATASET_ROOT / split / "labels_backup_before_bbox_clean"

    if not orig.exists() or not clean.exists():
        print(f"[WARN] Skipping split {split}: missing dirs")
        continue

    backup.mkdir(parents=True, exist_ok=True)

    # Backup originals once (only if backup folder is empty)
    if not any(backup.glob("*.txt")):
        for p in orig.glob("*.txt"):
            shutil.copy2(p, backup / p.name)
        print(f"[{split}] Backed up originals to: {backup}")
    else:
        print(f"[{split}] Backup already exists (not overwriting): {backup}")

    # Overwrite orig labels with clean labels
    for p in clean.glob("*.txt"):
        shutil.copy2(p, orig / p.name)

    print(f"[{split}] Overwrote labels/ with cleaned bbox labels.")

print("\nDone: labels are now bbox-only in-place.\n")


# ========================
# Step 2: Visualization (before polygon vs after bbox)
# ========================
VIZ_OUTDIR.mkdir(parents=True, exist_ok=True)

if len(poly_records) == 0:
    print("No polygon records found to visualize. Labels may already have been bbox-only.")
else:
    samples = random.sample(poly_records, min(N_VIZ_SAMPLES, len(poly_records)))
    print(f"Visualizing {len(samples)} polygon->bbox samples into: {VIZ_OUTDIR}")

    for k, (split, lbl_path, img_path, poly_norm, bbox_norm) in enumerate(samples, start=1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        # Left: polygon overlay
        pts_px = poly_norm_to_pts_px(poly_norm, W, H)
        left = draw_polygon(img_bgr, pts_px)

        # Right: bbox overlay (converted)
        xc, yc, bw, bh = bbox_norm
        xyxy = bbox_norm_to_xyxy_px(xc, yc, bw, bh, W, H)
        right = draw_bbox(img_bgr, xyxy)

        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(left_rgb)
        plt.title(f"Polygon (orig) | {split}/{lbl_path.name}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(right_rgb)
        plt.title("BBox (converted)")
        plt.axis("off")

        plt.tight_layout()
        out_png = VIZ_OUTDIR / f"poly_to_bbox_{k:03d}_{split}_{lbl_path.stem}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

    print("Done. Open the PNGs in:", VIZ_OUTDIR)
