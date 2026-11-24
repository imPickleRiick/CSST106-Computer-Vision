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
| 5 | PCA / Eigenfaces Face Recognition | ⏳ |
| 6 | YOLOv11 Object Detection | ⏳ |
| 7 | Image Segmentation | ⏳ |

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Project structure

```
app.py            Main entry point and sidebar navigation
labs/             One Streamlit page per laboratory activity
cv_core/          The computer vision algorithms for each lab (no UI code)
utils/ui.py       Shared UI helpers: image input, display, downloads
assets/samples/   Sample images (scikit-image test images)
tests/            Automated tests (pytest)
```

## Tests

```bash
pytest -q
```
