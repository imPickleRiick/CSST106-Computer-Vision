import cv2

from cv_core import lab6_yolo as Y
from utils.ui import load_sample


def test_pretrained_detects_bus_and_people():
    model = Y.load_model(Y.PRETRAINED)
    vis, dets, ms = Y.detect(model, load_sample("bus.jpg"), conf=0.4)
    classes = {d["class"] for d in dets}
    assert {"bus", "person"} <= classes
    assert vis.shape[2] == 3 and ms > 0


def test_yolo_label_parsing_and_drawing():
    files = Y.wildlife_samples()
    assert len(files) == 8
    labels = Y.read_yolo_labels(files[0].parent / "labels" / (files[0].stem + ".txt"))
    assert labels and all(0 <= v <= 1 for row in labels for v in row[1:])
    img = cv2.imread(str(files[0]))
    out = Y.draw_yolo_labels(img, labels, ["buffalo", "elephant", "rhino", "zebra"])
    assert out.shape == img.shape


def test_video_processing_writes_mp4():
    model = Y.load_model(Y.PRETRAINED)
    out, counts, stats = Y.process_video(model, str(Y.ROOT / "assets/samples/wildlife_clip.mp4"),
                                         max_seconds=1.0, stride=5, imgsz=320)
    cap = cv2.VideoCapture(out)
    assert cap.isOpened() and stats["frames processed"] >= 2
