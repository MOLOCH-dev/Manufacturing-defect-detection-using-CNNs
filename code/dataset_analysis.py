import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# ------------------------
# CONFIG
# ------------------------
DATASET_ROOT = Path("/home/mmpug/Desktop/AIforManuf/project1/data/defect_detection_3d_printing_yolo")
SPLITS = ["train", "valid", "test"]

# ------------------------
# LOAD YAML
# ------------------------
with open(DATASET_ROOT / "data.yaml", "r") as f:
    cfg = yaml.safe_load(f)

class_names = cfg["names"]
nc = cfg["nc"]

print(f"\nDataset root: {DATASET_ROOT}")
print(f"Classes ({nc}): {class_names}\n")

# ------------------------
# STORAGE
# ------------------------
class_counts = {s: Counter() for s in SPLITS}
bbox_stats = {s: defaultdict(list) for s in SPLITS}
image_sizes = defaultdict(list)
empty_labels = defaultdict(int)
invalid_boxes = defaultdict(int)

# ------------------------
# MAIN LOOP
# ------------------------
for split in SPLITS:
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    for lbl_path in lbl_dir.glob("*.txt"):
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            p = img_dir / f"{lbl_path.stem}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            print(f"[WARN] Image missing for label: {lbl_path.name}")
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        image_sizes[split].append((w, h))

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            empty_labels[split] += 1
            continue

        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            x, y, bw, bh = map(float, parts[1:])

            # integrity checks
            if not (0 <= cls < nc):
                invalid_boxes[split] += 1
                continue
            if bw <= 0 or bh <= 0:
                invalid_boxes[split] += 1
                continue

            class_counts[split][cls] += 1
            bbox_stats[split]["area"].append(bw * bh)
            bbox_stats[split]["width"].append(bw)
            bbox_stats[split]["height"].append(bh)

# ------------------------
# PRINT SUMMARY
# ------------------------
print("📊 CLASS DISTRIBUTION (number of boxes)\n")
for split in SPLITS:
    print(f"{split.upper()}:")
    for cid, name in enumerate(class_names):
        print(f"  {name:10s}: {class_counts[split][cid]}")
    print(f"  empty label files: {empty_labels[split]}")
    print(f"  invalid boxes:     {invalid_boxes[split]}\n")

# ------------------------
# PLOTS
# ------------------------
def plot_class_distribution():
    for split in SPLITS:
        counts = [class_counts[split][i] for i in range(nc)]
        plt.figure()
        plt.bar(class_names, counts)
        plt.title(f"{split} – class distribution")
        plt.ylabel("Number of boxes")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()

def plot_bbox_stats():
    for split in SPLITS:
        if not bbox_stats[split]["area"]:
            continue

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(bbox_stats[split]["area"], bins=50)
        plt.title(f"{split} bbox area")

        plt.subplot(1, 3, 2)
        plt.hist(bbox_stats[split]["width"], bins=50)
        plt.title(f"{split} bbox width")

        plt.subplot(1, 3, 3)
        plt.hist(bbox_stats[split]["height"], bins=50)
        plt.title(f"{split} bbox height")

        plt.tight_layout()
        plt.show()

def plot_image_sizes():
    for split, sizes in image_sizes.items():
        if not sizes:
            continue
        w, h = zip(*sizes)
        plt.figure()
        plt.scatter(w, h, alpha=0.3)
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.title(f"{split} image resolutions")
        plt.tight_layout()
        plt.show()

# ------------------------
# RUN VISUALS
# ------------------------
plot_class_distribution()
plot_bbox_stats()
plot_image_sizes()
