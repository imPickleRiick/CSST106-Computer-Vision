import numpy as np

from cv_core import lab1_image_ops as ops


def _img():
    img = np.zeros((40, 60, 3), np.uint8)
    img[..., 2] = 200  # red in BGR
    return img


def test_describe_and_gray():
    img = _img()
    d = ops.describe(img)
    assert (d["height"], d["width"], d["channels"]) == (40, 60, 3)
    g = ops.to_grayscale(img)
    assert g.shape == (40, 60)
    assert abs(int(g[0, 0]) - round(0.299 * 200)) <= 1


def test_pixel_set_get():
    out = ops.set_pixel(_img(), 5, 7, (1, 2, 3))
    assert ops.get_pixel(out, 5, 7) == [1, 2, 3]
    assert ops.get_pixel(out, 6, 7) == [0, 0, 200]


def test_crop_resize_rotate():
    img = _img()
    assert ops.crop(img, 10, 5, 20, 15).shape == (15, 20, 3)
    assert ops.resize(img, 30, 20).shape == (20, 30, 3)
    assert ops.rotate90(img).shape == (60, 40, 3)
    r = ops.rotate_any(img, 90)
    assert abs(r.shape[0] - 60) <= 1 and abs(r.shape[1] - 40) <= 1
