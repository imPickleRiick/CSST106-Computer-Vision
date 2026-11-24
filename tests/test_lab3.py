import cv2
import numpy as np

from cv_core import lab3_histogram as H
from utils.ui import load_sample


def test_manual_matches_opencv():
    gray = cv2.cvtColor(load_sample("moon.png"), cv2.COLOR_BGR2GRAY)
    manual, _ = H.equalize_manual(gray)
    assert np.abs(manual.astype(int) - cv2.equalizeHist(gray).astype(int)).max() <= 1


def test_equalization_spreads_range():
    low = H.reduce_contrast(np.tile(np.arange(256, dtype=np.uint8), (20, 1)), 100, 140)
    assert low.min() >= 100 and low.max() <= 140
    eq = H.equalize_gray(low)
    assert eq.min() == 0 and eq.max() == 255
    st = H.contrast_stretch(low)
    assert st.min() == 0 and st.max() == 255


def test_color_and_clahe_shapes():
    img = load_sample("coffee.png")
    for s in ("YUV", "HSV", "LAB"):
        assert H.equalize_color(img, s).shape == img.shape
    g = H.to_gray(img)
    assert H.clahe(g, 2.0, 8).shape == g.shape
    m = H.metrics(g)
    assert 0 < m["entropy (bits)"] <= 8


def test_constant_image():
    out, _ = H.equalize_manual(np.full((5, 5), 7, np.uint8))
    assert out.shape == (5, 5)
