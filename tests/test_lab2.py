import numpy as np

from cv_core import lab2_edges as E


def _square():
    img = np.zeros((64, 64), np.uint8)
    img[16:48, 16:48] = 255
    return img


def test_sobel_finds_vertical_and_horizontal_edges():
    s = E.sobel(_square())
    # strong x-gradient on the left/right sides, none in the flat centre
    assert abs(s["gx"][32, 16]) > 0 and s["gx"][32, 32] == 0
    assert abs(s["gy"][16, 32]) > 0 and s["gy"][32, 32] == 0
    assert s["magnitude"].max() > 0


def test_canny_is_binary_and_on_boundary():
    e = E.canny(_square(), 100, 200)
    assert set(np.unique(e)) <= {0, 255}
    assert e[32, 32] == 0 and e[:, 14:18].any()


def test_auto_thresholds_and_stats():
    low, high = E.auto_canny_thresholds(np.full((10, 10), 100, np.uint8))
    assert low < 100 < high
    st = E.edge_stats(E.canny(_square()))
    assert st["edge pixels"] > 0 and st["connected edge segments"] >= 1
