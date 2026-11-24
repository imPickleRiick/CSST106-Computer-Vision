"""Lab 6, Part 1: real-time object detection from your webcam with a pre-trained YOLO11n.

Runs locally (a web server cannot open your webcam as a live stream).
    python scripts/yolo_webcam.py                      # COCO-pretrained yolo11n
    python scripts/yolo_webcam.py --model models/lab6/best.pt   # custom wildlife model
Press Q to quit.

Needs a GUI build of OpenCV for cv2.imshow:  pip install opencv-python
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo11n.pt"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    model = YOLO(args.model)                 # load the pre-trained YOLO11n model
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()        # draws boxes, labels and confidences

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO11 real-time detection (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
