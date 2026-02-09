import random
import shutil
from pathlib import Path

# ------------------------
# CONFIG (EDIT THESE)
# ------------------------
# Existing YOLO dataset you already created (stringing+warping)
DEST_ROOT = Path("/home/mmpug/Desktop/AIforManuf/project1/data/defect_detection_3d_printing_yolo")

# Cracking dataset root (has ONLY train/images and train/labels)
CRACK_ROOT = Path("/home/mmpug/Desktop/AIforManuf/project1/data/crack_aug.v1i.yolov11")  # <-- CHANGE THIS
CRACK_SPLIT = "train"

# How many cracking images (total) to import
N_SAMPLES = 100

# Cracking dataset class id for cracking
SRC_CRACK_ID = 0

# Destination class id for cracking in your combined dataset:
# existing: 0=stringing, 1=warping  -> add cracking as 2
DEST_CRACK_ID = 2

# How to split cracking into dest train/valid/test
SPLIT_FRACS = {"train": 0.8, "valid": 0.1, "test": 0.1}

# Reproducibility
random.seed(42)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# ------------------------
# HELPERS
# ------------------------
def find_img(image_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def label_contains_class(label_path: Path, class_id: int) -> bool:
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if int(line.split()[0]) == class_id:
                return True
    return False

def remap_label_text(src_label: Path, src_id: int, dst_id: int) -> str:
    """Keep only src_id boxes; remap src_id -> dst_id; drop other classes."""
    out = []
    with open(src_label, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls == src_id:
                parts[0] = str(dst_id)
                out.append(" ".join(parts))
    return "\n".join(out) + ("\n" if out else "")

def unique_stem(dst_labels_dir: Path, base_stem: str) -> str:
    """Avoid filename collisions inside each split."""
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
    n_test = n - n_train - n_valid
    return {
        "train": items[:n_train],
        "valid": items[n_train:n_train + n_valid],
        "test":  items[n_train + n_valid:],
    }

# ------------------------
# PATHS
# ------------------------
src_img_dir = CRACK_ROOT / CRACK_SPLIT / "images"
src_lbl_dir = CRACK_ROOT / CRACK_SPLIT / "labels"

if not src_img_dir.exists() or not src_lbl_dir.exists():
    raise FileNotFoundError(
        f"Cracking dataset folders not found:\n"
        f"  images: {src_img_dir}\n"
        f"  labels: {src_lbl_dir}\n"
        f"Edit CRACK_ROOT in the script."
    )

# Ensure dest dirs exist
for split in ["train", "valid", "test"]:
    (DEST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

# ------------------------
# 1) Collect cracking candidates (must contain class 0)
# ------------------------
candidates = []
for lbl in src_lbl_dir.glob("*.txt"):
    if label_contains_class(lbl, SRC_CRACK_ID):
        img = find_img(src_img_dir, lbl.stem)
        if img is not None:
            candidates.append((lbl, img))

if not candidates:
    raise RuntimeError("No cracking candidates found. Check label format and CRACK_ROOT path.")

# Sample N
sampled = random.sample(candidates, min(N_SAMPLES, len(candidates)))
print(f"Found {len(candidates)} cracking candidates; importing {len(sampled)} total.")

# ------------------------
# 2) Split cracking into train/valid/test
# ------------------------
crack_splits = split_list(sampled, SPLIT_FRACS)
print({k: len(v) for k, v in crack_splits.items()})

# ------------------------
# 3) Copy into DEST_ROOT/<split>/ and remap 0 -> 2
# ------------------------
for split, items in crack_splits.items():
    dst_img_dir = DEST_ROOT / split / "images"
    dst_lbl_dir = DEST_ROOT / split / "labels"

    for src_lbl, src_img in items:
        base = f"crack_{src_img.stem}"
        stem = unique_stem(dst_lbl_dir, base)

        # image
        shutil.copy2(src_img, dst_img_dir / f"{stem}{src_img.suffix.lower()}")

        # label (only crack boxes; remapped)
        (dst_lbl_dir / f"{stem}.txt").write_text(
            remap_label_text(src_lbl, SRC_CRACK_ID, DEST_CRACK_ID)
        )

print("Done adding cracking samples across train/valid/test.")

# ------------------------
# 4) Update data.yaml to nc=3
# ------------------------
yaml_path = DEST_ROOT / "data.yaml"
yaml_text = f"""train: {DEST_ROOT / "train" / "images"}
val: {DEST_ROOT / "valid" / "images"}
test: {DEST_ROOT / "test" / "images"}

nc: 3
names: ['stringing', 'warping', 'cracking']
"""
yaml_path.write_text(yaml_text)
print(f"Updated: {yaml_path}")
