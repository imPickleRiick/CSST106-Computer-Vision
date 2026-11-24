"""Shared Streamlit helpers used by every lab page.

Images are kept in OpenCV's native BGR order inside the app and converted to
RGB only when they are displayed, exactly like a normal OpenCV program.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "assets" / "samples"

SAMPLE_DESCRIPTIONS = {
    "astronaut.png": "Astronaut portrait (color)",
    "coffee.png": "Coffee cup (color)",
    "chelsea_cat.png": "Cat (color)",
    "rocket.png": "Rocket launch (color)",
    "cameraman.png": "Cameraman (grayscale)",
    "coins.png": "Coins (grayscale, touching objects)",
    "moon.png": "Moon surface (grayscale, low contrast)",
    "page_text.png": "Book page (uneven lighting)",
}


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #
def lab_header(number: int, title: str, objective: str) -> None:
    """Standard heading shown at the top of every lab page."""
    st.caption(f"CSST106 · Perception and Computer Vision · Laboratory Activity {number}")
    st.title(f"Lab {number}: {title}")
    st.info(f"**Objective.** {objective}")


def code_block(title: str, code: str) -> None:
    """Collapsible block with the core OpenCV code behind a section."""
    with st.expander(f"Show code: {title}"):
        st.code(code.strip(), language="python")


# --------------------------------------------------------------------------- #
# Image input
# --------------------------------------------------------------------------- #
def list_samples() -> list[str]:
    return [name for name in SAMPLE_DESCRIPTIONS if (SAMPLES_DIR / name).exists()]


@st.cache_data(show_spinner=False)
def load_sample(name: str) -> np.ndarray:
    return cv2.imread(str(SAMPLES_DIR / name), cv2.IMREAD_COLOR)


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode uploaded bytes into a 3-channel BGR image (like cv2.imread)."""
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def image_input(
    key: str,
    label: str = "Input image",
    default_sample: str = "astronaut.png",
    allow_camera: bool = True,
    max_side: int | None = 1280,
) -> tuple[np.ndarray | None, str]:
    """Let the user pick a sample image, upload a file, or take a webcam photo.

    Returns (BGR image or None, short name of the source).
    """
    sources = ["Sample image", "Upload image"] + (["Webcam snapshot"] if allow_camera else [])
    st.markdown(f"**{label}**")
    source = st.radio("Source", sources, horizontal=True, key=f"{key}_src",
                      label_visibility="collapsed")

    img, name = None, ""
    if source == "Sample image":
        samples = list_samples()
        idx = samples.index(default_sample) if default_sample in samples else 0
        choice = st.selectbox("Sample", samples, index=idx, key=f"{key}_sample",
                              format_func=lambda n: SAMPLE_DESCRIPTIONS.get(n, n))
        img, name = load_sample(choice), choice
    elif source == "Upload image":
        up = st.file_uploader("Upload a JPG / PNG / BMP image",
                              type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                              key=f"{key}_upload")
        if up is not None:
            img, name = decode_image(up.getvalue()), up.name
            if img is None:
                st.error("Could not read that file as an image.")
    else:
        shot = st.camera_input("Take a photo", key=f"{key}_cam")
        if shot is not None:
            img, name = decode_image(shot.getvalue()), "webcam.jpg"

    if img is not None and max_side and max(img.shape[:2]) > max_side:
        scale = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        st.caption(f"Large image downscaled to {img.shape[1]}×{img.shape[0]} for speed.")
    return img, name


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert an OpenCV image (gray, BGR or BGRA, any dtype) to displayable RGB uint8."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show(img: np.ndarray, caption: str = "", raw_rgb: bool = False) -> None:
    """Display an OpenCV image. raw_rgb=True shows the array as-is (no BGR→RGB swap)."""
    st.image(img if raw_rgb else to_rgb(img), caption=caption, width="stretch")


def show_row(items: list[tuple[np.ndarray, str]]) -> None:
    """Show several (image, caption) pairs side by side."""
    cols = st.columns(len(items))
    for col, (img, cap) in zip(cols, items):
        with col:
            show(img, cap)


def png_bytes(img: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", img if img.dtype == np.uint8 else
                           cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    return enc.tobytes() if ok else b""


def download_button(img: np.ndarray, filename: str, label: str = "Download result") -> None:
    st.download_button(label, png_bytes(img), file_name=filename, mime="image/png",
                        key=f"dl_{filename}")


def fig_to_streamlit(fig) -> None:
    """Render a Matplotlib figure and free its memory."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    st.image(buf.getvalue(), width="stretch")
