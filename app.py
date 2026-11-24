"""CSST106 Perception and Computer Vision: compiled laboratory activities.

Run locally with:  streamlit run app.py
Each laboratory activity is its own page under labs/.
"""
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="CSST106 Computer Vision Labs",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent

# (file, sidebar title, icon). Pages that do not exist yet are skipped, so the
# app always runs while labs are being added one by one.
LABS = [
    ("labs/lab1_image_manipulation.py", "Lab 1 · Image Manipulation", "🖼️"),
    ("labs/lab2_edge_detection.py", "Lab 2 · Edge Detection", "📐"),
    ("labs/lab3_histogram.py", "Lab 3 · Histogram Equalization", "📊"),
    ("labs/lab4_surf_features.py", "Lab 4 · SURF Feature Detection", "🎯"),
    ("labs/lab5_pca_eigenfaces.py", "Lab 5 · PCA / Eigenfaces", "🧑"),
    ("labs/lab6_yolo_detection.py", "Lab 6 · YOLOv11 Object Detection", "📦"),
    ("labs/lab7_segmentation.py", "Lab 7 · Image Segmentation", "🧩"),
]

lab_pages = [st.Page(p, title=t, icon=i) for p, t, i in LABS if (ROOT / p).exists()]

nav = st.navigation(
    {
        "Overview": [st.Page("labs/home.py", title="Home", icon="🏠", default=True)],
        "Laboratory Activities": lab_pages,
    }
)

with st.sidebar:
    st.caption(
        "CSST106 · Perception and Computer Vision  \n"
        "Laguna State Polytechnic University · College of Computer Studies  \n"
        "A.Y. 2026–2027"
    )

nav.run()
