"""Lab 4: SURF feature detection, matching, and comparison with SIFT and ORB."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from cv_core.surf import SURF


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def make_detector(name: str, hessian: float = 400, n_features: int = 2000):
    if name == "SURF":
        return SURF(hessianThreshold=hessian)
    if name == "U-SURF":
        return SURF(hessianThreshold=hessian, upright=True)
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=n_features)
    if name == "ORB":
        return cv2.ORB_create(nfeatures=n_features)
    raise ValueError(name)


DESCRIPTOR_INFO = {
    "SURF": ("64 floats", "L2", "Box-filter Hessian (integral image) + Haar wavelets"),
    "U-SURF": ("64 floats", "L2", "Same as SURF, orientation step skipped"),
    "SIFT": ("128 floats", "L2", "Difference of Gaussians + gradient histograms"),
    "ORB": ("32 bytes (256 bits)", "Hamming", "FAST corners + rotated BRIEF binary tests"),
}


def synthetic_view(img: np.ndarray, angle: float = 30, scale: float = 1.0, brightness: int = 0,
                   noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/scale (canvas enlarged), change brightness and add noise.

    Returns (new image, 3×3 ground-truth homography mapping original → new).
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = max(1, int(h * sin + w * cos)), max(1, int(h * cos + w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    if brightness:
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=brightness)
    if noise > 0:
        rng = np.random.default_rng(seed)
        out = np.clip(out.astype(np.float32) + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out, np.vstack([M, [0, 0, 1]])


@dataclass
class MatchResult:
    method: str
    kp1: list
    kp2: list
    good: list
    n_kp1: int
    n_kp2: int
    t_detect: float
    inlier_mask: np.ndarray | None = None
    homography: np.ndarray | None = None
    correct: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum()) if self.inlier_mask is not None else 0


def detect(method: str, gray: np.ndarray, hessian: float = 400, n_features: int = 2000):
    det = make_detector(method, hessian, n_features)
    t0 = time.perf_counter()
    kps, desc = det.detectAndCompute(gray, None)
    return kps, desc, time.perf_counter() - t0


def match_images(method: str, img1: np.ndarray, img2: np.ndarray, ratio: float = 0.75,
                 hessian: float = 400, n_features: int = 2000,
                 true_h: np.ndarray | None = None, tol: float = 3.0) -> MatchResult:
    """Detect, describe, match with Lowe's ratio test, then verify with a RANSAC homography."""
    g1, g2 = to_gray(img1), to_gray(img2)
    k1, d1, t1 = detect(method, g1, hessian, n_features)
    k2, d2, t2 = detect(method, g2, hessian, n_features)
    res = MatchResult(method, k1, k2, [], len(k1), len(k2), t1 + t2)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return res
    norm = cv2.NORM_HAMMING if d1.dtype == np.uint8 else cv2.NORM_L2
    knn = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    res.good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(res.good) >= 4:
        p1 = np.float32([k1[m.queryIdx].pt for m in res.good])
        p2 = np.float32([k2[m.trainIdx].pt for m in res.good])
        Hm, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if mask is not None:
            res.inlier_mask, res.homography = mask.ravel().astype(bool), Hm
        if true_h is not None:
            proj = cv2.perspectiveTransform(p1[None], true_h)[0]
            res.correct = int((np.linalg.norm(proj - p2, axis=1) < tol).sum())
    elif true_h is not None:
        res.correct = 0
    return res


def draw_matches(img1, img2, r: MatchResult, max_draw: int = 80, inliers_only: bool = True) -> np.ndarray:
    good = r.good
    if inliers_only and r.inlier_mask is not None:
        good = [m for m, ok in zip(good, r.inlier_mask) if ok]
    good = sorted(good, key=lambda m: m.distance)[:max_draw]
    return cv2.drawMatches(img1, r.kp1, img2, r.kp2, good, None, matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def draw_keypoints(img, kps, color=(255, 0, 0), rich: bool = True) -> np.ndarray:
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    return cv2.drawKeypoints(img, kps, None, color, flags)


def summary_row(r: MatchResult) -> dict:
    n_good = len(r.good)
    row = {
        "keypoints (img1 / img2)": f"{r.n_kp1} / {r.n_kp2}",
        "detect + describe time (s)": round(r.t_detect, 3),
        "good matches (ratio test)": n_good,
        "RANSAC inliers": r.n_inliers,
        "inlier ratio (%)": round(100 * r.n_inliers / n_good, 1) if n_good else 0.0,
    }
    if r.correct is not None:
        row["correct matches (ground truth)"] = r.correct
        row["precision (%)"] = round(100 * r.correct / n_good, 1) if n_good else 0.0
    return row


def rotation_sweep(img: np.ndarray, methods: list[str], angles, hessian: float = 400) -> dict:
    """Matching precision vs rotation angle against the known ground-truth homography."""
    g = to_gray(img)
    out = {m: [] for m in methods}
    for a in angles:
        g2, Ht = synthetic_view(g, angle=a)
        for m in methods:
            r = match_images(m, g, g2, hessian=hessian, true_h=Ht)
            out[m].append(r.correct or 0)
    return out


def threshold_sweep(gray: np.ndarray, thresholds) -> list[int]:
    """Number of SURF keypoints for each Hessian threshold (response maps computed once)."""
    from cv_core.surf import build_response_maps, find_keypoints

    maps = build_response_maps(gray)
    return [len(find_keypoints(maps, t)) for t in thresholds]
