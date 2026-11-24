"""Lab 5: PCA in Computer Vision (Eigenfaces + SVM face recognition)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DATA_FILE = Path(__file__).resolve().parent.parent / "assets" / "data" / "lfw_people_min70_resize0.4.npz"


@dataclass
class Dataset:
    images: np.ndarray        # (n, h, w) float32 in [0, 1]
    X: np.ndarray             # (n, h*w) flattened
    y: np.ndarray
    target_names: np.ndarray
    h: int
    w: int


def load_lfw() -> Dataset:
    """LFW people with at least 70 images, resize=0.4.

    Bit-identical to sklearn.datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4),
    bundled so the web app does not have to download about 200 MB on every start.
    Falls back to downloading if the bundled file is missing.
    """
    if DATA_FILE.exists():
        d = np.load(DATA_FILE, allow_pickle=False)
        images, y, names = d["images"], d["target"].astype(int), d["target_names"]
    else:
        from sklearn.datasets import fetch_lfw_people

        lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        images, y, names = lfw.images, lfw.target, lfw.target_names
    n, h, w = images.shape
    return Dataset(images, images.reshape(n, -1), y, names, h, w)


@dataclass
class Experiment:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pca: PCA
    X_train_pca: np.ndarray
    X_test_pca: np.ndarray
    clf: SVC
    y_pred: np.ndarray
    t_pca: float
    t_svm: float

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_test, self.y_pred))


def run_experiment(ds: Dataset, n_components: int = 150, whiten: bool = True, C: float = 1.0,
                   gamma: str | float = "scale", test_size: float = 0.2, seed: int = 42) -> Experiment:
    """Tasks 1–4 of the handout: split, PCA (fit on train only), SVM on PCA weights, predict."""
    X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=test_size, random_state=seed)
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=whiten, random_state=seed).fit(X_train)
    t_pca = time.perf_counter() - t0
    Xtr, Xte = pca.transform(X_train), pca.transform(X_test)
    t0 = time.perf_counter()
    clf = SVC(kernel="rbf", class_weight="balanced", C=C, gamma=gamma).fit(Xtr, y_train)
    t_svm = time.perf_counter() - t0
    return Experiment(X_train, X_test, y_train, y_test, pca, Xtr, Xte, clf, clf.predict(Xte), t_pca, t_svm)


def eigenfaces(pca: PCA, h: int, w: int) -> np.ndarray:
    return pca.components_.reshape((pca.n_components_, h, w))


def cumulative_variance(X_train: np.ndarray) -> np.ndarray:
    """Cumulative explained variance for all components (full PCA on the training set)."""
    return PCA(svd_solver="full").fit(X_train).explained_variance_ratio_.cumsum()


def components_for(cum: np.ndarray, fraction: float) -> int:
    return int(np.argmax(cum >= fraction) + 1)


def reconstruct(pca: PCA, x: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct a face from its first k principal components (works for whitened PCA too)."""
    z = pca.transform(x[None])[0][:k]
    scale = np.sqrt(pca.explained_variance_[:k]) if pca.whiten else 1.0
    return pca.mean_ + (z * scale) @ pca.components_[:k]


def report(exp: Experiment, names) -> tuple[dict, np.ndarray]:
    rep = classification_report(exp.y_test, exp.y_pred, target_names=names, output_dict=True, zero_division=0)
    cm = confusion_matrix(exp.y_test, exp.y_pred, labels=range(len(names)))
    return rep, cm


def raw_pixel_baseline(exp: Experiment) -> tuple[float, float]:
    """SVM trained on the 1,850 raw pixels (no PCA), for comparison."""
    t0 = time.perf_counter()
    clf = SVC(kernel="rbf", class_weight="balanced").fit(exp.X_train, exp.y_train)
    t = time.perf_counter() - t0
    return float(accuracy_score(exp.y_test, clf.predict(exp.X_test))), t


# --------------------------------------------------------------------------- #
# Using your own photo
# --------------------------------------------------------------------------- #
_CASCADE = None


def detect_face(img_bgr: np.ndarray):
    """Largest frontal face found by OpenCV's Haar cascade, as (x, y, w, h) or None."""
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def lfw_style_crop(img_bgr: np.ndarray, box, h: int, w: int, target_mean: float, target_std: float) -> np.ndarray:
    """Crop a face box with roughly the LFW framing and resize it to (h, w) floats in [0, 1].

    fetch_lfw_people keeps the central 125×94 region of the 250×250 LFW images (eyebrows to
    chin, cheek to cheek). We approximate that with a slightly tightened Haar box at the same
    aspect ratio, then match the brightness/contrast (mean, std) of the training faces.
    """
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh * 0.55
    ch = bh * 1.05
    cw = ch * w / h
    x0, y0 = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    x1, y1 = int(min(img_bgr.shape[1], cx + cw / 2)), int(min(img_bgr.shape[0], cy + ch / 2))
    gray = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    face = (face - face.mean()) / max(face.std(), 1e-6) * target_std + target_mean
    return np.clip(face, 0, 1).astype(np.float32)


def predict_face(exp: Experiment, face: np.ndarray):
    """Return (predicted label, per-class decision scores, reconstruction error)."""
    x = face.ravel()
    z = exp.pca.transform(x[None])
    label = int(exp.clf.predict(z)[0])
    scores = exp.clf.decision_function(z)[0]
    recon = reconstruct(exp.pca, x, exp.pca.n_components_)
    err = float(np.sqrt(((recon - x) ** 2).mean()))
    return label, scores, err
