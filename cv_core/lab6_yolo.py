"""Lab 6: Object detection with YOLO11 (Ultralytics). Helper functions, no UI code."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = ROOT / "models" / "yolo11n.pt"
CUSTOM_DIR = ROOT / "models" / "lab6"
CUSTOM = CUSTOM_DIR / "best.pt"
WILDLIFE_SAMPLES = ROOT / "assets" / "samples" / "wildlife"

os.environ.setdefault("YOLO_OFFLINE", "1")   # never phone home from the web app


def load_model(path: Path | str):
    from ultralytics import YOLO

    return YOLO(str(path))


def detect(model, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640):
    """Run one image through YOLO.

    Returns (annotated BGR image, list of detection dicts, inference time in ms).
    """
    t0 = time.perf_counter()
    r = model.predict(img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        dets.append({"class": model.names[int(c)], "confidence": round(float(s), 3),
                     "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                     "width": int(x2 - x1), "height": int(y2 - y1)})
    return r.plot(line_width=max(2, round(max(img_bgr.shape[:2]) / 400))), dets, ms


def process_video(model, src_path: str, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
                  max_seconds: float = 10.0, stride: int = 1, max_width: int = 960, progress=None):
    """Detect objects in every `stride`-th frame of a video (up to max_seconds).

    Writes a browser-playable H.264 MP4 and returns (output path, per-frame counts, stats).
    """
    import imageio.v2 as imageio

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(min(total if total > 0 else 1e9, max_seconds * fps))
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(out_path, fps=max(1.0, fps / stride), codec="libx264",
                                quality=7, macro_block_size=2, ffmpeg_log_level="error")
    counts, times, idx = [], [], 0
    try:
        while idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                if frame.shape[1] > max_width:
                    s = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                vis, dets, ms = detect(model, frame, conf, iou, imgsz)
                times.append(ms)
                c = {}
                for d in dets:
                    c[d["class"]] = c.get(d["class"], 0) + 1
                counts.append(c)
                cv2.putText(vis, f"frame {idx}  {1000 / ms:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)
                writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                if progress:
                    progress(min(1.0, (idx + 1) / max(1, limit)))
            idx += 1
    finally:
        writer.close()
        cap.release()
    stats = {"frames processed": len(times), "source fps": round(fps, 1),
             "mean inference (ms)": round(float(np.mean(times)), 1) if times else 0,
             "throughput (FPS)": round(1000 / float(np.mean(times)), 1) if times else 0}
    return out_path, counts, stats


# --------------------------------------------------------------------------- #
# Dataset (YOLO format) helpers
# --------------------------------------------------------------------------- #
def read_yolo_labels(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if txt_path.exists():
        for line in txt_path.read_text().split("\n"):
            parts = line.split()
            if len(parts) == 5:
                rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def draw_yolo_labels(img_bgr: np.ndarray, labels, names) -> np.ndarray:
    """Draw ground-truth boxes from normalised (class, xc, yc, w, h) rows."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255)]
    t = max(2, round(max(h, w) / 300))
    for c, xc, yc, bw, bh in labels:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        col = colors[c % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        cv2.putText(out, names[c], (x1 + 3, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out


def wildlife_samples() -> list[Path]:
    return sorted(WILDLIFE_SAMPLES.glob("*.jpg"))


def load_summary() -> dict | None:
    f = CUSTOM_DIR / "summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
