"""SURF (Speeded-Up Robust Features) implemented from scratch with NumPy.

Why: SURF is patented, so the pip builds of OpenCV (opencv-contrib-python)
ship without it (cv2.xfeatures2d.SURF_create raises "This algorithm is patented
and is excluded in this configuration"). This module follows the original paper
(Bay, Tuytelaars & Van Gool, "SURF: Speeded Up Robust Features", ECCV 2006 / CVIU
2008) and OpenCV's reference implementation:

1. Integral image so any box sum costs 4 look-ups, whatever the box size.
2. Fast-Hessian detector: approximate the Gaussian second derivatives Dxx, Dyy,
   Dxy with box filters of size 9, 15, 21, ... and compute
   det(H) = Dxx·Dyy − (0.9·Dxy)². The image is never resized: larger scales
   use larger filters.
3. 3×3×3 non-maximum suppression in (x, y, scale) plus quadratic sub-pixel
   interpolation.
4. Orientation: Haar-wavelet responses (size 4s) in a circle of radius 6s,
   Gaussian weighted (σ = 2.5s), and a sliding π/3 window picks the dominant
   direction. U-SURF (upright) skips this step.
5. Descriptor: a 20s × 20s window aligned to the orientation is split into 4×4
   sub-regions. Each one stores Σdx, Σdy, Σ|dx|, Σ|dy| of Haar responses
   (size 2s, σ = 3.3s weighting), giving 64 values (128 when extended).

The Hessian threshold has the same meaning as OpenCV's hessianThreshold
(responses are computed on 0–255 intensities and box weights are normalised by
box area, exactly like OpenCV).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

# Box-filter layouts for the 9×9 base filter: (x1, y1, x2, y2, weight), taken from OpenCV.
_DXX = [(0, 2, 3, 7, 1), (3, 2, 6, 7, -2), (6, 2, 9, 7, 1)]
_DYY = [(2, 0, 7, 3, 1), (2, 3, 7, 6, -2), (2, 6, 7, 9, 1)]
_DXY = [(1, 1, 4, 4, 1), (5, 1, 8, 4, -1), (1, 5, 4, 8, -1), (5, 5, 8, 8, 1)]

BASE_SIZE, SIZE_INC = 9, 6


def integral_image(gray: np.ndarray) -> np.ndarray:
    """(h+1, w+1) float64 table where II[y, x] = sum of gray[:y, :x]."""
    return cv2.integral(gray.astype(np.float64), sdepth=cv2.CV_64F)


def _scaled_boxes(pattern, size):
    """Scale a 9×9 box pattern to `size` and normalise each weight by its box area."""
    r = size / BASE_SIZE
    out = []
    for x1, y1, x2, y2, w in pattern:
        dx1, dy1, dx2, dy2 = (int(round(r * v)) for v in (x1, y1, x2, y2))
        out.append((dx1, dy1, dx2, dy2, w / float((dx2 - dx1) * (dy2 - dy1))))
    return out


def box_kernel(pattern_name: str, size: int = 9) -> np.ndarray:
    """Dense kernel of a box filter (for visualisation only)."""
    pattern = {"Dxx": _DXX, "Dyy": _DYY, "Dxy": _DXY}[pattern_name]
    k = np.zeros((size, size))
    for x1, y1, x2, y2, w in _scaled_boxes(pattern, size):
        k[y1:y2, x1:x2] += w * (x2 - x1) * (y2 - y1)  # show the raw ±1/±2 weights
    return k


def _grid_response(ii, boxes, top, left, step):
    """Sum of weighted boxes for every window on a regular grid of top-left corners.

    top/left are arithmetic sequences (start, count) sampled every `step` pixels, so each
    box corner is a strided slice of the integral image (no fancy indexing needed).
    """
    (t0, ny), (l0, nx) = top, left
    acc = 0.0
    for x1, y1, x2, y2, w in boxes:
        r1 = slice(t0 + y1, t0 + y1 + (ny - 1) * step + 1, step)
        r2 = slice(t0 + y2, t0 + y2 + (ny - 1) * step + 1, step)
        c1 = slice(l0 + x1, l0 + x1 + (nx - 1) * step + 1, step)
        c2 = slice(l0 + x2, l0 + x2 + (nx - 1) * step + 1, step)
        acc = acc + w * (ii[r2, c2] - ii[r1, c2] - ii[r2, c1] + ii[r1, c1])
    return acc


@dataclass
class Octave:
    sizes: list[int]          # filter size per layer
    step: int                 # sampling step (pixels) in this octave
    cy: np.ndarray            # centre rows of the grid
    cx: np.ndarray            # centre cols of the grid
    det: np.ndarray           # (layers, len(cy), len(cx)) det(H) responses
    trace_sign: np.ndarray    # sign of the Laplacian (Dxx + Dyy), used for fast matching


def build_response_maps(gray: np.ndarray, n_octaves: int = 4, n_octave_layers: int = 3) -> list[Octave]:
    """Fast-Hessian scale space: det(H) for every octave/layer (no image resizing)."""
    h, w = gray.shape
    ii = integral_image(gray)
    octaves = []
    for o in range(n_octaves):
        step = 1 << o
        sizes = [(BASE_SIZE + SIZE_INC * l) << o for l in range(n_octave_layers + 2)]
        max_size = sizes[-1]
        half = max_size // 2
        cy = np.arange(half, h - (max_size - half) + 1, step)
        cx = np.arange(half, w - (max_size - half) + 1, step)
        if len(cy) < 3 or len(cx) < 3:
            break
        det = np.empty((len(sizes), len(cy), len(cx)))
        sgn = np.empty_like(det, dtype=np.int8)
        for li, L in enumerate(sizes):
            top, left = (int(cy[0]) - L // 2, len(cy)), (int(cx[0]) - L // 2, len(cx))
            dxx = _grid_response(ii, _scaled_boxes(_DXX, L), top, left, step)
            dyy = _grid_response(ii, _scaled_boxes(_DYY, L), top, left, step)
            dxy = _grid_response(ii, _scaled_boxes(_DXY, L), top, left, step)
            det[li] = dxx * dyy - 0.81 * dxy * dxy
            sgn[li] = np.where(dxx + dyy >= 0, 1, -1)
        octaves.append(Octave(sizes, step, cy, cx, det, sgn))
    return octaves


def find_keypoints(octaves: list[Octave], hessian_threshold: float = 400.0) -> np.ndarray:
    """3×3×3 non-maximum suppression + sub-pixel/sub-scale interpolation.

    Returns an (N, 5) array: x, y, filter_size, response, laplacian_sign.
    """
    out = []
    for oc in octaves:
        d = oc.det
        mx = maximum_filter(d, size=3, mode="constant", cval=-np.inf)
        cand = (d == mx) & (d > hessian_threshold)
        cand[0], cand[-1] = False, False                      # need a scale above and below
        cand[:, [0, -1], :], cand[:, :, [0, -1]] = False, False
        ls, ys, xs = np.nonzero(cand)
        if len(ls) == 0:
            continue
        # Quadratic interpolation: offset = -H^-1 g, with finite differences in (x, y, scale).
        D = lambda dl, dy, dx: d[ls + dl, ys + dy, xs + dx]  # noqa: E731
        c = D(0, 0, 0)
        g = np.stack([(D(0, 0, 1) - D(0, 0, -1)) / 2,
                      (D(0, 1, 0) - D(0, -1, 0)) / 2,
                      (D(1, 0, 0) - D(-1, 0, 0)) / 2], axis=1)
        dxx = D(0, 0, 1) + D(0, 0, -1) - 2 * c
        dyy = D(0, 1, 0) + D(0, -1, 0) - 2 * c
        dss = D(1, 0, 0) + D(-1, 0, 0) - 2 * c
        dxy = (D(0, 1, 1) - D(0, 1, -1) - D(0, -1, 1) + D(0, -1, -1)) / 4
        dxs = (D(1, 0, 1) - D(1, 0, -1) - D(-1, 0, 1) + D(-1, 0, -1)) / 4
        dys = (D(1, 1, 0) - D(1, -1, 0) - D(-1, 1, 0) + D(-1, -1, 0)) / 4
        H = np.stack([np.stack([dxx, dxy, dxs], 1), np.stack([dxy, dyy, dys], 1),
                      np.stack([dxs, dys, dss], 1)], 1)
        ok = np.abs(np.linalg.det(H)) > 1e-12
        off = np.zeros_like(g)
        off[ok] = -np.linalg.solve(H[ok], g[ok][..., None])[..., 0]
        keep = np.all(np.abs(off) < 1.0, axis=1)
        off[~keep] = 0.0                                       # fall back to the grid position
        x = oc.cx[xs] + off[:, 0] * oc.step
        y = oc.cy[ys] + off[:, 1] * oc.step
        size_inc = SIZE_INC * oc.step
        L = np.asarray(oc.sizes)[ls] + off[:, 2] * size_inc
        out.append(np.stack([x, y, L, c, oc.trace_sign[ls, ys, xs].astype(float)], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 5))


# --------------------------------------------------------------------------- #
# Haar wavelet responses (vectorised over many sample points)
# --------------------------------------------------------------------------- #
def _box_sum(ii, r0, c0, r1, c1):
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    r0, r1 = np.clip(r0, 0, H), np.clip(r1, 0, H)
    c0, c1 = np.clip(c0, 0, W), np.clip(c1, 0, W)
    return ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]


def haar_xy(ii, x, y, size):
    """Haar wavelet responses dx (right − left), dy (bottom − top) of side `size` centred at (x, y)."""
    x = np.rint(x).astype(np.int64)
    y = np.rint(y).astype(np.int64)
    half = np.maximum(1, np.asarray(size, dtype=np.int64) // 2)
    top, bot, left, right = y - half, y + half, x - half, x + half
    dx = _box_sum(ii, top, x, bot, right) - _box_sum(ii, top, left, bot, x)
    dy = _box_sum(ii, y, left, bot, right) - _box_sum(ii, top, left, y, right)
    return dx, dy


# Orientation sample pattern: integer (i, j) with i² + j² < 36, Gaussian σ = 2.5
_OI, _OJ = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7))
_OMASK = (_OI ** 2 + _OJ ** 2) < 36
_OI, _OJ = _OI[_OMASK].astype(float), _OJ[_OMASK].astype(float)
_OW = np.exp(-(_OI ** 2 + _OJ ** 2) / (2 * 2.5 ** 2))
_N_BINS = 42                       # 2π / 42 ≈ 0.15 rad per bin, as in the paper's window step
_WIN_BINS = 7                      # 7 bins ≈ π/3 sliding window


def assign_orientation(ii, kps: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Dominant orientation (radians) for each keypoint using a sliding π/3 window.

    The weighted Haar responses are binned by angle; a circular running sum over
    7 neighbouring bins implements the π/3 window that slides in 0.15 rad steps.
    """
    angles = np.zeros(len(kps))
    for a in range(0, len(kps), chunk):
        k = kps[a:a + chunk]
        n = len(k)
        s = 1.2 * k[:, 2:3] / 9.0                              # scale σ of each keypoint
        px = k[:, 0:1] + _OI[None] * s
        py = k[:, 1:2] + _OJ[None] * s
        hs = 2 * np.maximum(1, np.rint(2 * s))                 # Haar size 4s (even)
        dx, dy = haar_xy(ii, px, py, np.broadcast_to(hs, px.shape))
        dx, dy = dx * _OW, dy * _OW
        ang = np.arctan2(dy, dx) % (2 * np.pi)                 # (K, P)
        b = np.minimum((ang / (2 * np.pi) * _N_BINS).astype(int), _N_BINS - 1)
        idx = (np.arange(n)[:, None] * _N_BINS + b).ravel()
        hx = np.bincount(idx, dx.ravel(), n * _N_BINS).reshape(n, _N_BINS)
        hy = np.bincount(idx, dy.ravel(), n * _N_BINS).reshape(n, _N_BINS)
        # circular sliding-window sums
        wrap = lambda h: np.concatenate([h, h[:, :_WIN_BINS - 1]], axis=1).cumsum(axis=1)  # noqa: E731
        cx_, cy_ = wrap(hx), wrap(hy)
        pad = lambda c: np.concatenate([np.zeros((n, 1)), c], axis=1)  # noqa: E731
        cx_, cy_ = pad(cx_), pad(cy_)
        sx = cx_[:, _WIN_BINS:_WIN_BINS + _N_BINS] - cx_[:, :_N_BINS]
        sy = cy_[:, _WIN_BINS:_WIN_BINS + _N_BINS] - cy_[:, :_N_BINS]
        best = np.argmax(sx ** 2 + sy ** 2, axis=1)
        r = np.arange(n)
        angles[a:a + chunk] = np.arctan2(sy[r, best], sx[r, best])
    return angles


# Descriptor sample pattern: 20×20 samples at (u, v) ∈ {-9.5 … 9.5}, σ = 3.3
_DU, _DV = np.meshgrid(np.arange(20) - 9.5, np.arange(20) - 9.5)
_DU, _DV = _DU.ravel(), _DV.ravel()
_DW = np.exp(-(_DU ** 2 + _DV ** 2) / (2 * 3.3 ** 2))
_DSUB = ((_DV + 10) // 5).astype(int) * 4 + ((_DU + 10) // 5).astype(int)   # sub-region 0..15


def compute_descriptors(ii, kps: np.ndarray, angles: np.ndarray, extended: bool = False,
                        chunk: int = 512) -> np.ndarray:
    """64-D (or 128-D extended) SURF descriptors in the keypoint's oriented frame."""
    dim = 128 if extended else 64
    out = np.zeros((len(kps), dim), np.float32)
    onehot = np.eye(16)[_DSUB]                                 # (400, 16)
    for a in range(0, len(kps), chunk):
        k = kps[a:a + chunk]
        th = angles[a:a + chunk][:, None]
        co, si = np.cos(th), np.sin(th)
        s = 1.2 * k[:, 2:3] / 9.0
        px = k[:, 0:1] + s * (_DU[None] * co - _DV[None] * si)
        py = k[:, 1:2] + s * (_DU[None] * si + _DV[None] * co)
        hs = 2 * np.maximum(1, np.rint(s))                     # Haar size 2s (even)
        dx, dy = haar_xy(ii, px, py, np.broadcast_to(hs, px.shape))
        rx = (dx * co + dy * si) * _DW                         # rotate into keypoint frame
        ry = (-dx * si + dy * co) * _DW
        if extended:
            pos_y = ry >= 0
            pos_x = rx >= 0
            feats = [rx * pos_y, rx * ~pos_y, np.abs(rx) * pos_y, np.abs(rx) * ~pos_y,
                     ry * pos_x, ry * ~pos_x, np.abs(ry) * pos_x, np.abs(ry) * ~pos_x]
        else:
            feats = [rx, ry, np.abs(rx), np.abs(ry)]
        sums = np.stack([f @ onehot for f in feats], axis=2)  # (K, 16, F)
        desc = sums.reshape(len(k), -1)
        n = np.linalg.norm(desc, axis=1, keepdims=True)
        out[a:a + chunk] = (desc / np.maximum(n, 1e-12)).astype(np.float32)
    return out


def to_cv_keypoints(kps: np.ndarray, angles: np.ndarray, octave_ids=None) -> list[cv2.KeyPoint]:
    """Convert to cv2.KeyPoint so cv2.drawKeypoints / drawMatches can be used.

    The keypoint diameter is the box-filter size L (proportional to the scale σ = 1.2·L/9).
    Upright keypoints get angle −1 (no orientation), like OpenCV.
    """
    res = []
    for i, (x, y, L, resp, _sgn) in enumerate(kps):
        ang = float(np.degrees(angles[i]) % 360) if angles is not None else -1
        res.append(cv2.KeyPoint(float(x), float(y), float(L), ang, float(resp),
                                0 if octave_ids is None else int(octave_ids[i])))
    return res


class SURF:
    """Drop-in style replacement for cv2.xfeatures2d.SURF_create(...)."""

    def __init__(self, hessianThreshold: float = 100, nOctaves: int = 4, nOctaveLayers: int = 3,
                 extended: bool = False, upright: bool = False, max_keypoints: int | None = 4000):
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.extended = extended
        self.upright = upright
        self.max_keypoints = max_keypoints
        self.timings: dict[str, float] = {}

    def detect_raw(self, gray: np.ndarray, octaves=None) -> np.ndarray:
        if octaves is None:
            octaves = build_response_maps(gray, self.nOctaves, self.nOctaveLayers)
        kps = find_keypoints(octaves, self.hessianThreshold)
        if self.max_keypoints and len(kps) > self.max_keypoints:
            kps = kps[np.argsort(-kps[:, 3])[: self.max_keypoints]]
        return kps

    def detectAndCompute(self, img: np.ndarray, mask=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        t0 = time.perf_counter()
        kps = self.detect_raw(gray)
        t1 = time.perf_counter()
        ii = integral_image(gray)
        angles = np.zeros(len(kps)) if self.upright else assign_orientation(ii, kps)
        t2 = time.perf_counter()
        desc = compute_descriptors(ii, kps, angles, self.extended)
        t3 = time.perf_counter()
        self.timings = {"detect": t1 - t0, "orientation": t2 - t1, "describe": t3 - t2}
        cv_kps = to_cv_keypoints(kps, None if self.upright else angles)
        self.laplacian_sign = kps[:, 4].astype(np.int8)
        return cv_kps, desc


def opencv_surf_available() -> bool:
    try:
        cv2.xfeatures2d.SURF_create(400)
        return True
    except Exception:
        return False
