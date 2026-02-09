import random
import shutil
from pathlib import Path

# ------------------------
# CONFIG
# ------------------------
SRC_ROOT = Path("/home/mmpug/Desktop/AIforManuf/project1/data/3d print failure detection.v4i.yolov11")
SRC_SPLITS = ["train", "valid", "test"]

# Source class ids in YOUR Roboflow dataset
SRC_STRINGING_ID = 1
SRC_WARPING_ID = 2

# Output dataset root (YOLO format)
OUT_ROOT = Path("/home/mmpug/Desktop/AIforManuf/project1/data/defect_detection_3d_printing_yolo")
OUT_SPLITS = ["train", "valid", "test"]

# How many exclusive images per class to use total (before split)
MAX_PER_CLASS = 100

# Train/valid/test split fractions
SPLIT_FRACS = {"train": 0.8, "valid": 0.1, "test": 0.1}

# Output class mapping (new dataset)
# 0=stringing, 1=warping
OUT_CLASS_MAP = {SRC_STRINGING_ID: 0, SRC_WARPING_ID: 1}

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
random.seed(42)

# ------------------------
# HELPERS
# ------------------------
def find_img(image_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def parse_classes(label_path: Path) -> set[int]:
    classes = set()
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            classes.add(int(line.split()[0]))
    return classes

def remap_label_file(src_label: Path, dst_label: Path, class_map: dict[int, int]) -> None:
    out_lines = []
    with open(src_label, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls in class_map:
                parts[0] = str(class_map[cls])
                out_lines.append(" ".join(parts))
            # if cls not in class_map: drop it (keeps dataset clean)
    dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

def ensure_out_dirs(root: Path):
    for split in OUT_SPLITS:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

def unique_stem(dst_labels_dir: Path, base_stem: str) -> str:
    # Prevent collisions across splits
    cand = base_stem
    k = 1
    while (dst_labels_dir / f"{cand}.txt").exists():
        cand = f"{base_stem}_{k}"
        k += 1
    return cand

def split_list(items, fracs):
    items = items[:]
    random.shuffle(items)
    n = len(items)
    n_train = int(round(n * fracs["train"]))
    n_valid = int(round(n * fracs["valid"]))
    # ensure total == n
    n_test = n - n_train - n_valid
    return {
        "train": items[:n_train],
        "valid": items[n_train:n_train + n_valid],
        "test": items[n_train + n_valid:],
    }

# ------------------------
# 1) Collect EXCLUSIVE candidates for each class
# ------------------------
stringing_only = []
warping_only = []

for split in SRC_SPLITS:
    lbl_dir = SRC_ROOT / split / "labels"
    img_dir = SRC_ROOT / split / "images"

    for lbl in lbl_dir.glob("*.txt"):
        classes = parse_classes(lbl)

        # Exclusive logic: contains one defect but not the other
        if SRC_STRINGING_ID in classes and SRC_WARPING_ID not in classes:
            img = find_img(img_dir, lbl.stem)
            if img:
                stringing_only.append((lbl, img))

        elif SRC_WARPING_ID in classes and SRC_STRINGING_ID not in classes:
            img = find_img(img_dir, lbl.stem)
            if img:
                warping_only.append((lbl, img))

# sample up to MAX_PER_CLASS from each
stringing_only = random.sample(stringing_only, min(MAX_PER_CLASS, len(stringing_only)))
warping_only  = random.sample(warping_only,  min(MAX_PER_CLASS, len(warping_only)))

print(f"Exclusive candidates chosen: stringing={len(stringing_only)}, warping={len(warping_only)}")

# ------------------------
# 2) Split into train/valid/test (within each class) and merge
# ------------------------
str_splits = split_list(stringing_only, SPLIT_FRACS)
war_splits = split_list(warping_only, SPLIT_FRACS)

merged = {s: [] for s in OUT_SPLITS}
for s in OUT_SPLITS:
    merged[s].extend(str_splits[s])
    merged[s].extend(war_splits[s])
    random.shuffle(merged[s])

# ------------------------
# 3) Write YOLO-format output
# ------------------------
ensure_out_dirs(OUT_ROOT)

for split in OUT_SPLITS:
    out_img_dir = OUT_ROOT / split / "images"
    out_lbl_dir = OUT_ROOT / split / "labels"

    for src_lbl, src_img in merged[split]:
        # make names unique (across this split)
        base = f"{src_img.stem}"
        stem = unique_stem(out_lbl_dir, base)

        # copy image
        shutil.copy2(src_img, out_img_dir / f"{stem}{src_img.suffix.lower()}")

        # remap + write label
        remap_label_file(src_lbl, out_lbl_dir / f"{stem}.txt", OUT_CLASS_MAP)

print("Done writing YOLO-format dataset.")

# ------------------------
# 4) Write data.yaml for new dataset
# ------------------------
yaml_text = f"""train: {OUT_ROOT / "train" / "images"}
val: {OUT_ROOT / "valid" / "images"}
test: {OUT_ROOT / "test" / "images"}

nc: 2
names: ['stringing', 'warping']
"""
(OUT_ROOT / "data.yaml").write_text(yaml_text)
print(f"Wrote: {OUT_ROOT / 'data.yaml'}")
