"""Lab 3: Histogram Equalization and Contrast Adjustment. Pure OpenCV/NumPy functions."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def reduce_contrast(img: np.ndarray, low: int, high: int) -> np.ndarray:
    """Squeeze intensities into [low, high] to simulate a poor-contrast image."""
    f = img.astype(np.float32) / 255.0
    return (low + f * (high - low)).round().astype(np.uint8)


def histogram(gray: np.ndarray) -> np.ndarray:
    """256-bin histogram with cv2.calcHist."""
    return cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()


def cdf(hist: np.ndarray) -> np.ndarray:
    c = hist.cumsum()
    return c / c[-1]


# ----------------------------- Part 1 ------------------------------------- #
def equalize_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def equalize_manual(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Histogram equalization from scratch.

    s_k = round( (CDF(k) - CDF_min) / (N - CDF_min) * 255 )
    Returns (equalized image, look-up table).
    """
    hist = np.bincount(gray.ravel(), minlength=256)
    c = hist.cumsum()
    cdf_min = c[np.nonzero(c)[0][0]]
    n = gray.size
    if n == cdf_min:                       # constant image: nothing to stretch
        return gray.copy(), np.arange(256, dtype=np.uint8)
    lut = np.round((c - cdf_min) / (n - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    return lut[gray], lut


def equalize_color(img_bgr: np.ndarray, space: str = "YUV") -> np.ndarray:
    """Equalize only the luminance/value channel so colors are not distorted."""
    if space == "YUV":
        yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    if space == "HSV":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if space == "LAB":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.equalizeHist(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    raise ValueError(space)


def equalize_rgb_naive(img_bgr: np.ndarray) -> np.ndarray:
    """Wrong way, shown for comparison: equalizing B, G, R independently shifts the colors."""
    return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img_bgr)])


# ----------------------------- Part 2 ------------------------------------- #
def contrast_stretch(img: np.ndarray, low_pct: float = 0.0, high_pct: float = 100.0) -> np.ndarray:
    """Linear contrast stretching: map [p_low, p_high] → [0, 255] (min-max when 0/100)."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        return img.copy()
    out = (img.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """g(x) = alpha·f(x) + beta, saturated to [0, 255]."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(gray)


def clahe_color(img_bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe(lab[..., 0], clip_limit, tile)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------- Part 4 ------------------------------------- #
def entropy(gray: np.ndarray) -> float:
    p = histogram(gray) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def noise_level(gray: np.ndarray) -> float:
    """Rough noise estimate: mean absolute difference from a 3×3 median-filtered version."""
    return float(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)).mean())


def metrics(gray: np.ndarray) -> dict:
    return {
        "mean": round(float(gray.mean()), 1),
        "std (RMS contrast)": round(float(gray.std()), 1),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "levels used": int((histogram(gray) > 0).sum()),
        "entropy (bits)": round(entropy(gray), 2),
        "noise estimate": round(noise_level(gray), 2),
    }
