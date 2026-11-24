import numpy as np

from cv_core import lab7_segmentation as G
from utils.ui import load_sample


def test_otsu_and_adaptive_on_synthetic():
    img, gt, _ = G.synthetic_scene(noise=8, gradient=0)
    t, m = G.otsu(G.to_gray(img))
    assert 60 < t < 160
    assert G.iou(m, gt) > 0.9


def test_region_grow_stays_inside_object():
    img = np.full((60, 60), 50, np.uint8)
    img[20:40, 20:40] = 200
    lab = G.region_grow(img, [(30, 30)], tol=10)
    assert (lab > 0).sum() == 400


def test_watershed_counts_coins():
    st = G.watershed_steps(load_sample("coins.png"), "peaks", 18)
    assert 20 <= st["n_objects"] <= 28          # the image contains 24 coins


def test_edge_segmentation_and_metrics():
    img, gt, _ = G.synthetic_scene(noise=5, gradient=40)
    e = G.edge_segmentation(G.to_gray(img), 30, 90, 5, 200, 7)
    assert G.iou(e["mask"], gt) > 0.8
    assert G.dice(gt, gt) == 1.0 and G.iou(gt, gt) == 1.0


def test_semantic_segmentation_finds_person_and_bus():
    model = G.load_deeplab()
    lab, prob, _ = G.semantic_segment(model, load_sample("bus.jpg"))
    found = {G.VOC_CLASSES[i] for i in np.unique(lab)}
    assert {"person", "bus"} <= found
    assert prob.shape[0] == 21
