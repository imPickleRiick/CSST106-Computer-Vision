"""Lab 2: Edge Detection (Sobel and Canny). Pure OpenCV/NumPy functions."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess(img: np.ndarray, blur_ksize: int = 3, sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale + Gaussian blur, as in the handout. Returns (gray, blurred)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if blur_ksize <= 1:
        return gray, gray.copy()
    return gray, cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), sigma)


def add_noise(gray: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise (to test each detector's noise sensitivity)."""
    if std <= 0:
        return gray
    rng = np.random.default_rng(seed)
    noisy = gray.astype(np.float32) + rng.normal(0, std, gray.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def sobel(gray: np.ndarray, ksize: int = 3) -> dict[str, np.ndarray]:
    """Sobel gradients. Returns float64 Gx, Gy, magnitude and direction (degrees)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 360) % 360
    return {"gx": gx, "gy": gy, "magnitude": mag, "direction": ang}


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Scale an absolute gradient image to 0–255 for display (like cv2.convertScaleAbs + normalise)."""
    a = np.abs(x)
    m = a.max()
    return np.zeros_like(a, np.uint8) if m == 0 else (a / m * 255).astype(np.uint8)


def threshold_magnitude(mag: np.ndarray, percent: float) -> np.ndarray:
    """Binary edge map: keep pixels whose magnitude ≥ percent% of the max."""
    return ((mag >= (percent / 100.0) * mag.max()) * 255).astype(np.uint8)


def direction_color(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Colour-code gradient direction (hue) weighted by magnitude (value). Returns BGR."""
    hsv = np.zeros((*mag.shape, 3), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)        # OpenCV hue range 0–179
    hsv[..., 1] = 255
    hsv[..., 2] = to_uint8(mag)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def canny(gray: np.ndarray, low: int = 100, high: int = 200, aperture: int = 3,
          l2: bool = False) -> np.ndarray:
    return cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=l2)


def auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """Median-based thresholds: [ (1-σ)·median , (1+σ)·median ]."""
    v = float(np.median(gray))
    return int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))


def other_operators(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Extra first/second-order operators for comparison."""
    sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float64)
    px = cv2.filter2D(gray.astype(np.float64), -1, kx)
    py = cv2.filter2D(gray.astype(np.float64), -1, kx.T)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return {
        "Prewitt magnitude": to_uint8(cv2.magnitude(px, py)),
        "Scharr magnitude": to_uint8(cv2.magnitude(sx, sy)),
        "Laplacian |∇²I|": to_uint8(lap),
    }


def edge_stats(edge_map: np.ndarray) -> dict:
    """Simple numbers for comparing binary edge maps."""
    on = edge_map > 0
    n_edge = int(on.sum())
    n_comp, _ = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
    return {
        "edge pixels": n_edge,
        "edge density (%)": round(100.0 * n_edge / on.size, 2),
        "connected edge segments": n_comp - 1,
    }


SOBEL_X_KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y_KERNEL = SOBEL_X_KERNEL.T
