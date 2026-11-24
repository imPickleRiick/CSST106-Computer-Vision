"""Lab 7: Image segmentation techniques. Pure OpenCV / NumPy / PyTorch functions."""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TORCH_HOME = ROOT / "models" / "torch"


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def overlay(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    """Tint the pixels where mask > 0."""
    base = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    out = base.copy()
    m = mask > 0
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def label_colors(labels: np.ndarray, seed: int = 3) -> np.ndarray:
    """Random color per label (0 = black background, -1 = white boundary)."""
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 2
    palette = rng.integers(40, 255, (n, 3), dtype=np.uint8)
    out = palette[np.clip(labels, 0, None)]
    out[labels == 0] = 0
    out[labels == -1] = 255
    return out


# --------------------------------------------------------------------------- #
# 1. Thresholding
# --------------------------------------------------------------------------- #
def global_threshold(gray, t: int, invert: bool = False):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(gray, t, 255, mode)[1]


def otsu(gray, invert: bool = False, blur: int = 5):
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    mode = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    t, m = cv2.threshold(g, 0, 255, mode)
    return float(t), m


def adaptive(gray, method: str = "gaussian", block: int = 31, C: int = 5, invert: bool = False):
    am = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, am, mode, block, C)


# --------------------------------------------------------------------------- #
# 2. Region-based
# --------------------------------------------------------------------------- #
def region_grow(gray: np.ndarray, seeds: list[tuple[int, int]], tol: float = 12.0,
                connectivity: int = 8) -> np.ndarray:
    """Seeded region growing.

    Each region starts at a seed (x, y) and absorbs neighbouring pixels whose intensity
    is within `tol` of the region's running mean. Returns int labels (0 = unassigned).
    """
    h, w = gray.shape
    g = gray.astype(np.float32)
    labels = np.zeros((h, w), np.int32)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for rid, (sx, sy) in enumerate(seeds, start=1):
        if not (0 <= sx < w and 0 <= sy < h) or labels[sy, sx]:
            continue
        total, count = float(g[sy, sx]), 1
        labels[sy, sx] = rid
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            mean = total / count
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0 and abs(g[ny, nx] - mean) <= tol:
                    labels[ny, nx] = rid
                    total += g[ny, nx]
                    count += 1
                    q.append((ny, nx))
    return labels


def watershed_steps(img_bgr: np.ndarray, marker_mode: str = "peaks", min_distance: int = 15,
                    fg_ratio: float = 0.5, invert: bool = False, open_iter: int = 2) -> dict[str, np.ndarray]:
    """Marker-controlled watershed for separating touching objects.

    1. Otsu binarisation, then opening (remove specks), closing and hole filling (solid objects).
    2. Sure background = dilated foreground. Distance transform of the foreground.
    3. Markers (sure foreground), one per object:
       - "peaks": local maxima of the distance transform at least `min_distance` apart
         (works when objects have different sizes);
       - "threshold": dist > fg_ratio · max(dist), as in the OpenCV tutorial
         (small objects vanish when sizes differ).
    4. Unknown = sure_bg − sure_fg; cv2.watershed floods from the markers and marks
       borders with −1.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    gray = to_gray(img_bgr)
    _, th = otsu(gray, invert=invert)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    opening = ndi.binary_fill_holes(opening > 0).astype(np.uint8) * 255
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if marker_mode == "peaks":
        comp = cv2.connectedComponents(opening)[1]
        peaks = peak_local_max(dist, min_distance=min_distance, threshold_abs=3, labels=comp)
        sure_fg = np.zeros_like(opening)
        for y, x in peaks:   # a small disc per peak so each object gets exactly one seed
            cv2.circle(sure_fg, (int(x), int(y)), max(2, int(dist[y, x] * 0.3)), 255, -1)
    else:
        _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                  # background = 1, objects = 2..n
    markers[unknown == 255] = 0            # region to be decided by flooding
    color = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(color.copy(), markers.copy())
    result = color.copy()
    result[ws == -1] = (0, 0, 255)
    objects = ws.copy()
    objects[objects == 1] = 0              # background
    return {"binary": th, "opening": opening, "sure_bg": sure_bg, "dist": dist, "sure_fg": sure_fg,
            "unknown": unknown, "markers": markers, "labels": objects, "result": result,
            "n_objects": n - 1}


def watershed_oversegmented(gray: np.ndarray) -> np.ndarray:
    """Watershed seeded at every local minimum of the gradient, to show over-segmentation."""
    from scipy import ndimage
    from skimage.filters import sobel
    from skimage.morphology import local_minima
    from skimage.segmentation import watershed

    grad = sobel(gray.astype(float) / 255.0)
    markers, _ = ndimage.label(local_minima(grad))
    return watershed(grad, markers)


# --------------------------------------------------------------------------- #
# 3. Edge-based
# --------------------------------------------------------------------------- #
def edge_segmentation(gray: np.ndarray, low: int = 50, high: int = 150, close_k: int = 5,
                      min_area: int = 100, blur: int = 5) -> dict[str, np.ndarray]:
    """Canny edges, then morphological closing to seal gaps, then filled closed contours."""
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    edges = cv2.Canny(g, low, high)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return {"edges": edges, "closed": closed, "mask": mask, "contours": contours}


def draw_contours(img_bgr, contours, color=(0, 255, 0), thickness=2):
    out = (img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)).copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


# --------------------------------------------------------------------------- #
# 4. Semantic segmentation (DeepLabV3, PASCAL VOC classes)
# --------------------------------------------------------------------------- #
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa",
               "train", "tv/monitor"]


def voc_palette() -> np.ndarray:
    """Standard PASCAL VOC color map (RGB)."""
    pal = np.zeros((256, 3), np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


def load_deeplab():
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    import torch
    from torchvision.models.segmentation import (DeepLabV3_MobileNet_V3_Large_Weights,
                                                 deeplabv3_mobilenet_v3_large)

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
    return model, weights.transforms()


def semantic_segment(model_and_tf, img_bgr: np.ndarray, max_side: int = 520):
    """Per-pixel VOC class labels. Returns (label map at input size, class probabilities, ms)."""
    import torch

    model, tf = model_and_tf
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    x = torch.from_numpy(small).permute(2, 0, 1)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(tf(x).unsqueeze(0))["out"][0]
        prob = out.softmax(0)
    ms = (time.perf_counter() - t0) * 1000
    lab = prob.argmax(0).numpy().astype(np.uint8)
    lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
    return lab, prob.numpy(), ms


def colorize_voc(labels: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(voc_palette()[labels], cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# 5. Quantitative analysis on a synthetic image with known ground truth
# --------------------------------------------------------------------------- #
def synthetic_scene(noise: float = 12.0, gradient: float = 80.0, seed: int = 0, size: int = 384):
    """Bright objects (some touching) on a background with an illumination gradient.

    Returns (BGR image, ground-truth foreground mask, number of objects).
    """
    rng = np.random.default_rng(seed)
    gt = np.zeros((size, size), np.uint8)
    circles = [(90, 100, 45), (165, 110, 42), (280, 90, 50), (100, 270, 55), (210, 260, 40),
               (300, 290, 48)]
    for x, y, r in circles:
        cv2.circle(gt, (x, y), r, 255, -1)
    cv2.rectangle(gt, (250, 170), (360, 205), 255, -1)
    yy, xx = np.mgrid[0:size, 0:size]
    background = 60 + gradient * xx / size
    img = np.where(gt > 0, background + 90, background).astype(np.float32)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), gt, len(circles) + 1


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    s = p.sum() + g.sum()
    return float(2 * np.logical_and(p, g).sum() / s) if s else 1.0


def compare_methods(img_bgr: np.ndarray, gt: np.ndarray) -> list[dict]:
    """Segment the synthetic scene with every classical method and score it against ground truth."""
    gray = to_gray(img_bgr)
    rows = []

    def run(name, fn):
        t0 = time.perf_counter()
        mask, n_obj = fn()
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"method": name, "IoU": round(iou(mask, gt), 3), "Dice": round(dice(mask, gt), 3),
                     "objects found": n_obj, "time (ms)": round(ms, 1), "mask": mask})

    def count(mask):
        return cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1

    def m_global():
        m = global_threshold(cv2.GaussianBlur(gray, (5, 5), 0), 127)
        return m, count(m)

    def m_otsu():
        m = otsu(gray)[1]
        return m, count(m)

    def m_adapt():
        m = adaptive(cv2.GaussianBlur(gray, (5, 5), 0), "gaussian", 151, -10)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return m, count(m)

    def m_region():
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        seeds = [(x, y) for y in range(10, gray.shape[0], 40) for x in range(10, gray.shape[1], 40)]
        lab = region_grow(blur, seeds, tol=18)
        # region containing the brightest mean = objects: keep regions brighter than the image mean
        m = np.zeros_like(gray)
        for r in np.unique(lab[lab > 0]):
            reg = lab == r
            if blur[reg].mean() > blur.mean() + 10:
                m[reg] = 255
        return m, count(m)

    def m_ws():
        st = watershed_steps(img_bgr, "peaks", 20)
        return (st["labels"] > 1).astype(np.uint8) * 255, st["n_objects"]

    def m_edge():
        e = edge_segmentation(gray, 30, 90, 5, 200, 7)
        return e["mask"], len(e["contours"])

    run("Global threshold (T=127)", m_global)
    run("Otsu threshold", m_otsu)
    run("Adaptive threshold", m_adapt)
    run("Region growing", m_region)
    run("Watershed", m_ws)
    run("Edge-based (Canny + contours)", m_edge)
    return rows
