import numpy as np

from cv_core import lab5_pca as P


def test_dataset_matches_handout():
    ds = P.load_lfw()
    assert ds.images.shape == (1288, 50, 37)
    assert ds.X.shape == (1288, 1850)
    assert len(ds.target_names) == 7


def test_experiment_accuracy_and_variance():
    ds = P.load_lfw()
    e = P.run_experiment(ds, 150, True)
    assert e.X_train_pca.shape[1] == 150
    assert e.accuracy > 0.8
    cum = P.cumulative_variance(e.X_train)
    assert 70 <= P.components_for(cum, 0.90) <= 100


def test_reconstruction_converges():
    ds = P.load_lfw()
    e = P.run_experiment(ds, 150, True)
    x = e.X_test[0]
    err = [np.abs(P.reconstruct(e.pca, x, k) - x).mean() for k in (5, 50, 150)]
    assert err[0] > err[1] > err[2]


def test_own_photo_pipeline():
    import cv2
    from utils.ui import load_sample

    ds = P.load_lfw()
    e = P.run_experiment(ds, 150, True)
    img = load_sample("astronaut.png")
    box = P.detect_face(img)
    assert box is not None
    face = P.lfw_style_crop(img, box, ds.h, ds.w, float(ds.X.mean()), float(ds.X.std()))
    assert face.shape == (50, 37)
    label, scores, err = P.predict_face(e, face)
    assert 0 <= label < 7 and len(scores) == 7
