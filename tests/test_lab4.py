import cv2
import numpy as np

from cv_core import lab4_features as F
from cv_core import surf as S
from utils.ui import load_sample


def _gray():
    return cv2.cvtColor(load_sample("chelsea_cat.png"), cv2.COLOR_BGR2GRAY)


def test_integral_image_box_sum():
    g = np.arange(30, dtype=np.uint8).reshape(5, 6)
    ii = S.integral_image(g)
    assert ii.shape == (6, 7)
    assert S._box_sum(ii, 1, 2, 4, 5) == g[1:4, 2:5].sum()


def test_box_kernel_patterns_sum_to_zero():
    for name in ("Dxx", "Dyy", "Dxy"):
        assert abs(S.box_kernel(name, 9).sum()) < 1e-9


def test_detect_and_compute_shapes():
    kp, d = S.SURF(400).detectAndCompute(_gray())
    assert len(kp) > 50 and d.shape == (len(kp), 64)
    assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-4)
    _, d128 = S.SURF(400, extended=True).detectAndCompute(_gray())
    assert d128.shape[1] == 128


def test_threshold_reduces_keypoints():
    c = F.threshold_sweep(_gray(), (100, 400, 1600))
    assert c[0] > c[1] > c[2]


def test_surf_is_rotation_invariant_and_usurf_is_not():
    g = _gray()
    g2, H = F.synthetic_view(g, angle=90)
    surf = F.match_images("SURF", g, g2, true_h=H)
    usurf = F.match_images("U-SURF", g, g2, true_h=H)
    assert surf.correct / len(surf.good) > 0.9
    assert usurf.correct < 0.1 * surf.correct
