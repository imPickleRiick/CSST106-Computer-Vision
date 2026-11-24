"""Lab 6: Object Detection with YOLO11."""
import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from cv_core import lab6_yolo as Y
from utils.ui import code_block, decode_image, download_button, image_input, lab_header, show, show_row

lab_header(
    6,
    "Object Detection with YOLO11",
    "Run real-time object detection with a pre-trained YOLO11 model, prepare an annotated dataset in YOLO "
    "format, train (fine-tune) YOLO11 on a custom dataset, and evaluate the trained model with precision, "
    "recall and mAP.",
)


@st.cache_resource(show_spinner="Loading YOLO model…")
def get_model(path: str):
    return Y.load_model(path)


custom_ready = Y.CUSTOM.exists()
summary = Y.load_summary()

with st.sidebar:
    st.markdown("### Detection settings")
    model_choice = st.radio(
        "Model",
        ["YOLO11n pre-trained (COCO, 80 classes)"] + (["Custom YOLO11n (African wildlife, 4 classes)"]
                                                       if custom_ready else []),
        key="l6_model")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="l6_conf")
    iou = st.slider("NMS IoU threshold", 0.1, 0.95, 0.7, 0.05, key="l6_iou",
                    help="Overlapping boxes with IoU above this are merged by non-maximum suppression.")
    imgsz = st.select_slider("Inference size (imgsz)", [320, 416, 480, 640, 960], 640, key="l6_sz")

model_path = str(Y.PRETRAINED if model_choice.startswith("YOLO11n pre") else Y.CUSTOM)
model = get_model(model_path)

tabs = st.tabs([
    "Part 1 · Detect (image / webcam)",
    "Part 1 · Detect (video)",
    "Part 2 · Custom dataset",
    "Part 3 · Training",
    "Part 4 · Evaluation",
    "Question & conclusion",
])

# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Object detection on an image or webcam snapshot")
    c1, c2 = st.columns([1, 2])
    with c1:
        default = "bus.jpg" if "pre-trained" in model_choice else None
        if default:
            img, name = image_input("lab6", default_sample=default, max_side=1280)
        else:
            src = st.radio("Source", ["Wildlife test image", "Upload image", "Webcam snapshot"], key="l6_wsrc")
            img, name = None, ""
            if src == "Wildlife test image":
                files = Y.wildlife_samples()
                pick = st.selectbox("Image", files, format_func=lambda p: p.stem, key="l6_wpick")
                img, name = cv2.imread(str(pick)), pick.name
            elif src == "Upload image":
                up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="l6_wup")
                if up:
                    img, name = decode_image(up.getvalue()), up.name
            else:
                shot = st.camera_input("Take a photo", key="l6_wcam")
                if shot:
                    img, name = decode_image(shot.getvalue()), "webcam.jpg"
    if img is None:
        st.info("Choose an input image.")
    else:
        with st.spinner("Detecting…"):
            vis, dets, ms = Y.detect(model, img, conf, iou, imgsz)
        with c2:
            show(vis, f"{len(dets)} objects detected in {ms:.0f} ms ({1000 / ms:.1f} FPS on this server's CPU)")
        download_button(vis, "lab6_detections.png", "Download result")
        if dets:
            df = pd.DataFrame(dets)
            cc1, cc2 = st.columns([2, 1])
            cc1.dataframe(df, hide_index=True, width="stretch")
            cc2.dataframe(df["class"].value_counts().rename("count"), width="stretch")
    st.info("**Real-time webcam detection.** A web server cannot read your webcam as a live video stream, "
            "so the web version uses webcam *snapshots*. For true real-time detection (Part 1 of the handout), "
            "run the included script on your own computer: `python scripts/yolo_webcam.py` "
            "(add `--model models/lab6/best.pt` for the custom model). It draws boxes, labels, confidences "
            "and FPS live.", icon="🎥")
    code_block("Part 1 inference", f"""
from ultralytics import YOLO
model = YOLO("yolo11n.pt")                 # COCO pre-trained YOLO11 nano
results = model("image.jpg", conf={conf}, iou={iou}, imgsz={imgsz})
annotated = results[0].plot()              # boxes + labels + confidences
for box in results[0].boxes:
    print(model.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
""")

# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Object detection on a video")
    vsrc = st.radio("Video", ["Sample wildlife clip (13 s)", "Upload a video"], horizontal=True, key="l6_vsrc")
    path = None
    if vsrc.startswith("Sample"):
        path = str(Y.ROOT / "assets" / "samples" / "wildlife_clip.mp4")
        st.caption("Tip: choose the **Custom YOLO11n (African wildlife)** model in the sidebar for this clip. "
                   "The COCO model only knows *elephant* and *zebra*, not buffalo or rhino.")
    else:
        up = st.file_uploader("MP4 / MOV / AVI", type=["mp4", "mov", "avi", "mkv", "webm"], key="l6_vup")
        if up:
            f = tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False)
            f.write(up.getvalue())
            f.close()
            path = f.name
    c1, c2 = st.columns(2)
    max_s = c1.slider("Process at most (seconds)", 2, 30, 13, key="l6_maxs",
                      help="Video detection runs on a CPU server, so long videos are cut.")
    stride = c2.select_slider("Process every n-th frame", [1, 2, 3, 5], 2, key="l6_stride")
    if path and st.button("Run detection on video", type="primary", key="l6_run"):
        bar = st.progress(0.0, "Processing frames…")
        out, counts, stats = Y.process_video(model, path, conf, iou, imgsz, max_s, stride,
                                             progress=lambda p: bar.progress(p, "Processing frames…"))
        bar.empty()
        st.video(out)
        st.dataframe(pd.DataFrame([stats]), hide_index=True, width="stretch")
        if counts:
            st.markdown("**Objects per processed frame**")
            st.line_chart(pd.DataFrame(counts).fillna(0))
        with open(out, "rb") as fh:
            st.download_button("Download annotated video", fh.read(), "lab6_detections.mp4", "video/mp4")
        os.unlink(out)

# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Part 2: Preparing a custom dataset")
    names = summary["classes"] if summary else ["buffalo", "elephant", "rhino", "zebra"]
    st.markdown(
        f"""
**Dataset used: Ultralytics *African Wildlife*.** It has 1,504 photos with **4 classes**: {", ".join(names)}.
It is already annotated with bounding boxes in YOLO format, split into **1,052 train / 225 val / 227 test** images.

**How the annotation is done (LabelImg / Roboflow):**
1. Collect images of the objects (different backgrounds, angles, sizes and lighting).
2. Open them in an annotation tool such as **LabelImg** (`pip install labelImg`), **Roboflow** or **CVAT**,
   and set the output format to **YOLO**.
3. Draw a tight box around every object and choose its class.
4. The tool saves one `.txt` file per image with one line per object:
   `class_id x_center y_center width height`. All four coordinates are **normalized to 0–1** by the
   image width and height.
"""
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Folder structure**")
        st.code("""datasets/african-wildlife/
│── images/
│     ├── train/   (1052 .jpg)
│     ├── val/     (225 .jpg)
│     └── test/    (227 .jpg)
│── labels/
      ├── train/   (one .txt per image)
      ├── val/
      └── test/""", language="text")
    with c2:
        st.markdown("**data.yaml**")
        st.code(f"""train: ./datasets/african-wildlife/images/train
val: ./datasets/african-wildlife/images/val
test: ./datasets/african-wildlife/images/test
nc: {len(names)}
names: {names}""", language="yaml")

    st.markdown("#### Browse annotated examples (ground-truth boxes from the label files)")
    files = Y.wildlife_samples()
    pick = st.select_slider("Example", files, format_func=lambda p: p.stem, key="l6_ds")
    gt_img = cv2.imread(str(pick))
    labels = Y.read_yolo_labels(pick.parent / "labels" / (pick.stem + ".txt"))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(Y.draw_yolo_labels(gt_img, labels, names), f"{pick.name}: {len(labels)} annotated object(s)")
    with c2:
        st.markdown(f"`labels/{pick.stem}.txt`")
        st.code("\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in labels), language="text")
        h_, w_ = gt_img.shape[:2]
        st.markdown("Converted to pixels:")
        st.dataframe(pd.DataFrame([{
            "class": names[c], "x_center": round(xc * w_), "y_center": round(yc * h_),
            "width": round(w * w_), "height": round(h * h_)} for c, xc, yc, w, h in labels]),
            hide_index=True, width="stretch")
    if summary and (Y.CUSTOM_DIR / "labels.jpg").exists():
        with st.expander("Label statistics for the training set (generated by Ultralytics)"):
            st.image(str(Y.CUSTOM_DIR / "labels.jpg"), width="stretch")

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Part 3: Training YOLO11 on the custom dataset")
    if not summary:
        st.warning("The custom model has not been trained yet. Run `python training/train_lab6.py`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Base model", summary["base_model"])
        c2.metric("Epochs", summary["epochs"])
        c3.metric("Image size", summary["imgsz"])
        on_gpu = str(summary["device"]) != "cpu"
        c4.metric("Training time", f"{summary['train_minutes']:.1f} min")
        c4.caption("on a Google Colab T4 GPU" if on_gpu else "on CPU")
        st.markdown(
            "Training uses **transfer learning**: it starts from YOLO11n weights pre-trained on COCO (which "
            "already learned edges, textures and animal shapes) and fine-tunes all layers on the 1,052 wildlife "
            f"training images with the handout's settings (`epochs={summary['epochs']}, imgsz={summary['imgsz']}`). "
            + ("Training ran on a free **Google Colab T4 GPU** with `training/train_lab6_colab.ipynb`, because "
               "CPU training took about 4 minutes per epoch." if on_gpu else "")
        )
        res_csv = Y.CUSTOM_DIR / "results.csv"
        if res_csv.exists():
            df = pd.read_csv(res_csv)
            df.columns = [c.strip() for c in df.columns]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Training losses per epoch** (lower is better)")
                st.line_chart(df.set_index("epoch")[["train/box_loss", "train/cls_loss", "train/dfl_loss"]])
            with c2:
                st.markdown("**Validation metrics per epoch** (higher is better)")
                st.line_chart(df.set_index("epoch")[["metrics/precision(B)", "metrics/recall(B)",
                                                     "metrics/mAP50(B)", "metrics/mAP50-95(B)"]])
        if (Y.CUSTOM_DIR / "results.png").exists():
            with st.expander("Full Ultralytics training curves (results.png)"):
                st.image(str(Y.CUSTOM_DIR / "results.png"), width="stretch")
        if (Y.CUSTOM_DIR / "train_batch0.jpg").exists():
            with st.expander("A training batch with augmentation (mosaic, flips, HSV)"):
                st.image(str(Y.CUSTOM_DIR / "train_batch0.jpg"), width="stretch")
    code_block("Part 3 training", """
# CLI
yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640

# Python (what training/train_lab6.py does; run on Colab with --device 0)
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=32, device=0)
# -> weights and metrics in runs/detect/<name>/ (best.pt, last.pt, results.csv, ...)
""")

# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Part 4: Evaluating and using the trained model")
    if not summary:
        st.warning("The custom model has not been trained yet.")
    else:
        split = st.radio("Split", ["test", "val"], horizontal=True, key="l6_split",
                         format_func=lambda s: {"test": "Test set (227 unseen images)", "val": "Validation set (225)"}[s])
        m = summary["metrics"][split]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m['precision']:.3f}")
        c2.metric("Recall", f"{m['recall']:.3f}")
        c3.metric("mAP@0.5", f"{m['mAP50']:.3f}")
        c4.metric("mAP@0.5:0.95", f"{m['mAP50-95']:.3f}")
        st.dataframe(pd.DataFrame(m["per_class"]).T.round(3), width="stretch")
        st.caption(
            "**Precision** is the share of predicted boxes that are correct. **Recall** is the share of real "
            "objects that were found. **mAP@0.5** is the mean average precision when a box counts as correct "
            "at IoU ≥ 0.5. **mAP@0.5:0.95** averages over stricter IoU thresholds, from 0.5 to 0.95.")
        c1, c2 = st.columns(2)
        for col, f, cap in ((c1, f"{split}_confusion_matrix_normalized.png", "Normalized confusion matrix"),
                            (c2, f"{split}_BoxPR_curve.png", "Precision–recall curve")):
            if (Y.CUSTOM_DIR / f).exists():
                col.image(str(Y.CUSTOM_DIR / f), caption=cap, width="stretch")
        c1, c2 = st.columns(2)
        for col, f, cap in ((c1, f"{split}_val_batch0_labels.jpg", "Ground truth"),
                            (c2, f"{split}_val_batch0_pred.jpg", "Predictions")):
            if (Y.CUSTOM_DIR / f).exists():
                col.image(str(Y.CUSTOM_DIR / f), caption=cap, width="stretch")

        st.markdown("#### Pre-trained COCO model vs custom model on the same image")
        pick = st.selectbox("Wildlife test image", Y.wildlife_samples(), format_func=lambda p: p.stem, key="l6_cmp")
        im = cv2.imread(str(pick))
        v1, d1, _ = Y.detect(get_model(str(Y.PRETRAINED)), im, conf, iou, imgsz)
        v2, d2, _ = Y.detect(get_model(str(Y.CUSTOM)), im, conf, iou, imgsz)
        show_row([(v1, "Pre-trained YOLO11n (COCO): " + (", ".join(sorted({d['class'] for d in d1})) or "nothing")),
                  (v2, "Custom YOLO11n (wildlife): " + (", ".join(sorted({d['class'] for d in d2})) or "nothing"))])
        st.caption("COCO has no *buffalo* or *rhino* class, so the pre-trained model mislabels them (on these "
                   "samples, buffalo as *elephant* or even *person*, rhinos as *cow* or *elephant*). The fine-tuned "
                   "model detects all four species. Its limitation is the opposite: it **only** knows these four "
                   "classes, so other animals are forced into one of them (in *zebra_105* the deer in the "
                   "background are detected as *elephant*). This is the closed-set problem; the fix is adding "
                   "more classes or negative examples to the training data.")
    code_block("Part 4 evaluation", """
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="data.yaml", split="test")
print(metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map)   # P, R, mAP50, mAP50-95
results = model("new_image.jpg")
results[0].show()
""")

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Question")
    st.markdown(
        """
**What is object detection in computer vision, and how does it differ from image classification? Explain your
answer using a real-world example.**

**Image classification** answers *"what is in this image?"*. It assigns **one label (or a set of labels) to the
whole image**, for example *"elephant"*, with a confidence score. It does not say **where** the object is or
**how many** there are.

**Object detection** answers *"what objects are in this image, and where are they?"*. For **every** object it
outputs a **class label**, a **confidence score** and a **bounding box** (x, y, width, height). One image can
contain many objects of different classes. A detector like YOLO solves classification and localization
together in a single forward pass, then uses non-maximum suppression to remove duplicate boxes.

**Real-world example: a wildlife camera trap in a national park.**
- A *classifier* looking at a camera-trap photo can only say *"this photo contains a zebra"*. If a zebra and a
  rhino are both present, it gives one dominant label and the rhino may go unnoticed.
- The *detector* trained in this lab outputs *"zebra 0.91 at (120, 80, 210, 160)"*, *"zebra 0.88 at
  (330, 95, 190, 150)"* and *"rhino 0.84 at (40, 200, 260, 180)"*. Rangers can **count** animals per species,
  **track** them across frames of a video, and get alerts when a protected species such as the rhino
  appears, which classification alone cannot do.

The same difference applies to self-driving cars: knowing that a scene contains "a pedestrian" is not
enough. The car must know **where each pedestrian, car and traffic light is** to plan a safe path.

### Conclusion
The pre-trained YOLO11n model detected the 80 COCO classes in images, webcam snapshots and videos in real
time, even on a CPU. Because COCO has no buffalo or rhino classes (it called them *elephant*, *cow* or *person*),
we fine-tuned YOLO11n on the annotated African Wildlife dataset in YOLO format for 100 epochs at 640 px on a
Google Colab T4 GPU (28.5 minutes). Transfer learning from COCO weights let the model learn the four classes
quickly. On the 227 held-out test images it reached **precision 0.97, recall 0.95, mAP@0.5 0.98 and
mAP@0.5:0.95 0.83**, so it generalizes well to unseen photos. Its main limitation is that it only knows these
four classes, so other animals get forced into one of them. This lab covered the full detection workflow:
inference, dataset preparation and annotation, training, and evaluation.
"""
    )
