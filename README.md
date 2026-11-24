# CSST106 Perception and Computer Vision: Laboratory Activities

A single Streamlit web application that compiles all Computer Vision laboratory
activities (Lab 1 to Lab 7) for CSST106, Laguna State Polytechnic University,
College of Computer Studies, A.Y. 2026–2027.

Each lab has its own page in the sidebar. On every page you can pick a sample
image, upload your own or take a webcam snapshot, adjust parameters, and see the
outputs together with explanations and the OpenCV code behind them.

| Lab | Topic | Status |
|---|---|---|
| 1 | Image Manipulation | ✅ |
| 2 | Edge Detection (Sobel, Canny) | ✅ |
| 3 | Histogram Equalization and Contrast Adjustment | ✅ |
| 4 | SURF Feature Detection (from-scratch SURF, SIFT, ORB) | ✅ |
| 5 | PCA / Eigenfaces Face Recognition | ✅ |
| 6 | YOLO11 Object Detection (pre-trained + custom wildlife model) | ✅ |
| 7 | Image Segmentation (thresholding, region, edge, DeepLabV3) | ✅ |

## Live web app

Deployed on Streamlit Community Cloud: *(link added after deployment)*

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Lab 6 custom model

The custom YOLO11n model (`models/lab6/best.pt`) was fine-tuned on the Ultralytics
*African Wildlife* dataset (buffalo, elephant, rhino, zebra) for 100 epochs at 640 px
on a Google Colab T4 GPU using [`training/train_lab6_colab.ipynb`](training/train_lab6_colab.ipynb)
(same code as [`training/train_lab6.py`](training/train_lab6.py)).
Test set: precision 0.971, recall 0.949, mAP@0.5 0.976, mAP@0.5:0.95 0.833.

Real-time webcam detection (Part 1) runs locally: `python scripts/yolo_webcam.py`.


## Project structure

```
app.py            Main entry point and sidebar navigation
labs/             One Streamlit page per laboratory activity
cv_core/          The computer vision algorithms for each lab (no UI code)
utils/ui.py       Shared UI helpers: image input, display, downloads
assets/samples/   Sample images (scikit-image test images)
models/           Pre-trained and trained model weights (YOLO)
training/         Lab 6 training script and Colab notebook
scripts/          Stand-alone scripts (real-time webcam detection)
tests/            Automated tests (pytest)
```

## Tests

```bash
pytest -q
```
