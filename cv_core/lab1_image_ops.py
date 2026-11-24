"""Lab 1: Image Manipulation. Pure OpenCV/NumPy functions (no UI code)."""
from __future__ import annotations

import cv2
import numpy as np


def describe(img: np.ndarray) -> dict:
    """Basic properties of an image array, as printed by img.shape / img.dtype."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    return {
        "shape": tuple(img.shape),
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(img.dtype),
        "total_pixels": h * w,
        "size_bytes": img.nbytes,
        "min": int(img.min()),
        "max": int(img.max()),
        "mean": float(img.mean()),
    }


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """cv2.cvtColor BGR→GRAY: Y = 0.299 R + 0.587 G + 0.114 B."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def split_channels(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Return each channel shown in its own color (others zeroed)."""
    b, g, r = cv2.split(img_bgr)
    z = np.zeros_like(b)
    return {
        "Blue (index 0)": cv2.merge([b, z, z]),
        "Green (index 1)": cv2.merge([z, g, z]),
        "Red (index 2)": cv2.merge([z, z, r]),
    }


def get_pixel(img: np.ndarray, x: int, y: int):
    """Pixel value at column x, row y. Note NumPy indexing is img[row, col] = img[y, x]."""
    return img[y, x].tolist() if img.ndim == 3 else int(img[y, x])


def set_pixel(img_bgr: np.ndarray, x: int, y: int, bgr: tuple[int, int, int], size: int = 1) -> np.ndarray:
    """Return a copy with the pixel (or a size×size block centred on it) set to a new BGR color."""
    out = img_bgr.copy()
    r = size // 2
    y0, y1 = max(0, y - r), min(out.shape[0], y - r + size)
    x0, x1 = max(0, x - r), min(out.shape[1], x - r + size)
    out[y0:y1, x0:x1] = bgr
    return out


def zoom_patch(img: np.ndarray, x: int, y: int, radius: int = 6, scale: int = 24) -> np.ndarray:
    """Enlarge the neighbourhood around (x, y) with nearest-neighbour so single pixels are visible."""
    h, w = img.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = img[y0:y1, x0:x1]
    big = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    # outline the selected pixel
    cx, cy = (x - x0) * scale, (y - y0) * scale
    cv2.rectangle(big, (cx, cy), (cx + scale - 1, cy + scale - 1), (0, 255, 255), 2)
    return big


def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Region of interest via NumPy slicing: img[y:y+h, x:x+w]."""
    return img[y:y + h, x:x + w].copy()


def draw_roi(img_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    out = img_bgr.copy()
    t = max(2, round(max(out.shape[:2]) / 250))
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), t)
    return out


INTERPOLATIONS = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}


def resize(img: np.ndarray, width: int, height: int, interpolation: str = "INTER_LINEAR") -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=INTERPOLATIONS[interpolation])


ROTATIONS = {
    "90° clockwise": cv2.ROTATE_90_CLOCKWISE,
    "90° counter-clockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180°": cv2.ROTATE_180,
}


def rotate90(img: np.ndarray, how: str = "90° clockwise") -> np.ndarray:
    return cv2.rotate(img, ROTATIONS[how])


def rotate_any(img: np.ndarray, angle: float, keep_whole: bool = True) -> np.ndarray:
    """Rotate by an arbitrary angle about the centre with an affine warp.

    keep_whole=True enlarges the canvas so no corner gets clipped.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    if keep_whole:
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - center[0]
        m[1, 2] += nh / 2 - center[1]
        return cv2.warpAffine(img, m, (nw, nh))
    return cv2.warpAffine(img, m, (w, h))


def flip(img: np.ndarray, mode: int) -> np.ndarray:
    """mode 1 = horizontal, 0 = vertical, -1 = both."""
    return cv2.flip(img, mode)
