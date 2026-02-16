#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import os
import cv2
from ultralytics import YOLO

# ==========================
# USER SETTINGS
# ==========================
# WEIGHTS = Path("/home/mmpug/Desktop/AIforManuf/project1/code/best.pt")
WEIGHTS = Path("/home/mmpug/Desktop/AIforManuf/project1/code/runs/detect/train/weights/best.pt")

OUT_DIR = Path("./out_webcam_3")
OUT_TXT = OUT_DIR / "detections.txt"
OUT_VIDEO = OUT_DIR / "annotated.mp4"   # use .avi if mp4 codec fails

# Webcam
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

# Video writer
VIDEO_FPS = 30
VIDEO_CODEC = "mp4v"
RESIZE_FOR_VIDEO: Optional[Tuple[int, int]] = None

# Model
IMG_SIZE = 768
CONF_THRES = 0.3
IOU_THRES = 0.5
DEVICE = 0  # or "cpu"
# CLASSES = ["stringing", "warping", "Cracksc"]
CLASSES = ["Blobs", "Cracks", "Spaghetti", "Stringging", "Under Extrusion"]

# Optional saving of frames
SAVE_FRAMES = False
SAVE_EVERY_N = 10
OUT_FRAMES = OUT_DIR / "annotated_frames"

# rename label for display only
DISPLAY_NAME_MAP = {"Spaghetti": "Stringging"}


def _make_writer(out_path: Path, fps: float, frame_size_wh: Tuple[int, int], codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w, h = frame_size_wh
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter at {out_path} with codec='{codec}'. "
            f"Try OUT_VIDEO ending with .avi and VIDEO_CODEC='MJPG'."
        )
    return writer


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_FRAMES:
        OUT_FRAMES.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))

    id_to_name: Dict[int, str] = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    name_to_id: Dict[str, int] = {v: k for k, v in id_to_name.items()}

    missing = [c for c in CLASSES if c not in name_to_id]
    if missing:
        raise ValueError(
            "These class names were not found in model.names:\n"
            f"  missing={missing}\n"
            f"  model.names={id_to_name}\n"
            "Fix CLASSES to match your data.yaml exactly."
        )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAM_INDEX}")

    if CAM_WIDTH is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    if CAM_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if CAM_FPS is not None:
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    writer: Optional[cv2.VideoWriter] = None
    video_frame_wh: Optional[Tuple[int, int]] = None

    print("[INFO] Press 'q' to quit.")
    print(f"[INFO] Appending detections to: {OUT_TXT}")
    print(f"[INFO] Writing annotated video to: {OUT_VIDEO}")

    frame_idx = 0

    # IMPORTANT:
    # - use append mode so the serial script can follow the file
    # - line-buffering (buffering=1) flushes at each newline for text IO
    with open(OUT_TXT, "a", buffering=1) as fdet:
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] Failed to read frame from webcam.")
                    break

                frame_idx += 1

                results = model.predict(
                    source=frame,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRES,
                    iou=IOU_THRES,
                    device=DEVICE,
                    verbose=False,
                )
                r = results[0]

                detected = False
                if r.boxes is not None and len(r.boxes) > 0:
                    cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                    for cid in cls_ids:
                        if id_to_name.get(cid, "") in CLASSES:
                            detected = True
                            break

                # ---- CONTINUOUS WRITE: flush + fsync so other process sees it immediately ----
                fdet.write("TRUE\n" if detected else "FALSE\n")
                fdet.flush()
                os.fsync(fdet.fileno())

                # display "Springing" instead of "Spaghetti" (plot text only)
                annotated = r.plot(labels=[DISPLAY_NAME_MAP.get(id_to_name[int(c)], id_to_name[int(c)]) for c in (r.boxes.cls.detach().cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else [])])

                if RESIZE_FOR_VIDEO is not None:
                    annotated = cv2.resize(annotated, RESIZE_FOR_VIDEO, interpolation=cv2.INTER_AREA)

                if writer is None:
                    h, w = annotated.shape[:2]
                    video_frame_wh = (w, h)
                    writer = _make_writer(OUT_VIDEO, VIDEO_FPS, video_frame_wh, VIDEO_CODEC)

                if video_frame_wh is not None:
                    tw, th = video_frame_wh
                    h, w = annotated.shape[:2]
                    if (w, h) != (tw, th):
                        annotated = cv2.resize(annotated, (tw, th), interpolation=cv2.INTER_AREA)

                writer.write(annotated)

                if SAVE_FRAMES and (frame_idx % SAVE_EVERY_N == 0):
                    cv2.imwrite(str(OUT_FRAMES / f"{frame_idx:06d}.jpg"), annotated)

                cv2.imshow("YOLOv11 - annotated", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if frame_idx % 100 == 0:
                    print(f"[INFO] Processed {frame_idx} frames")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    print("[DONE]")


if __name__ == "__main__":
    main()
