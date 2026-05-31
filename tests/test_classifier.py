from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from src.classifier import (
    TwoStageClassifier,
    LABEL_CONFIRMED,
    LABEL_CROSSEDOUT,
    LABEL_EMPTY,
    _sift_histogram,
)


def _make_cells(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (64, 64), dtype=np.uint8) for _ in range(n)]


def _make_classifier(stage1_preds, stage2_class_sequence=None) -> TwoStageClassifier:
    """Build a TwoStageClassifier with mocked internals.

    Parameters
    ----------
    stage1_preds : sequence of int
        SVM prediction per cell: 0=empty, 1=filled.
    stage2_class_sequence : sequence of int or None
        CNN predicted class per filled cell (0=confirmed, 1=crossedout).
    """
    kmeans = MagicMock()
    kmeans.cluster_centers_ = np.zeros((200, 128), dtype=np.float32)
    kmeans.predict = MagicMock(return_value=np.zeros(5, dtype=int))

    svm = MagicMock()
    svm.predict = MagicMock(return_value=np.array(stage1_preds, dtype=int))

    bovw_model = {"kmeans": kmeans, "svm": svm}

    cnn_net = MagicMock()
    cnn_net.get_inputs.return_value = [MagicMock(name="input")]
    if stage2_class_sequence is not None:
        outputs = iter(stage2_class_sequence)

        def _run(output_names, feed):
            cls = next(outputs)
            out = np.zeros((1, 2), dtype=np.float32)
            out[0, cls] = 1.0
            return [out]

        cnn_net.run = MagicMock(side_effect=_run)
    else:
        cnn_net.run = MagicMock(return_value=[np.array([[1.0, 0.0]], dtype=np.float32)])

    mock_sift = MagicMock()
    mock_sift.detectAndCompute = MagicMock(return_value=(None, None))
    with patch("cv2.SIFT_create", return_value=mock_sift):
        clf = TwoStageClassifier(bovw_model, cnn_net)
    return clf


def test_predict_returns_one_label_per_cell():
    cells = _make_cells(6)
    clf = _make_classifier([0, 0, 0, 0, 0, 0])
    result = clf.predict(cells)
    assert len(result) == 6


def test_predict_empty_input_returns_empty_list():
    clf = _make_classifier([])
    assert clf.predict([]) == []


def test_all_empty_stage1_returns_empty_labels():
    cells = _make_cells(4)
    clf = _make_classifier([0, 0, 0, 0])
    result = clf.predict(cells)
    assert result == [LABEL_EMPTY] * 4


def test_labels_are_in_valid_range():
    cells = _make_cells(5)
    clf = _make_classifier([1, 0, 1, 0, 1], stage2_class_sequence=[0, 1, 0])
    result = clf.predict(cells)
    assert all(lbl in {LABEL_CONFIRMED, LABEL_CROSSEDOUT, LABEL_EMPTY} for lbl in result)


def test_empty_cells_bypass_stage2():
    cells = _make_cells(3)
    clf = _make_classifier([0, 0, 0])
    clf.predict(cells)
    clf._cnn.run.assert_not_called()


def test_filled_cells_go_through_stage2():
    cells = _make_cells(3)
    clf = _make_classifier([0, 1, 0], stage2_class_sequence=[0])
    result = clf.predict(cells)
    clf._cnn.run.assert_called_once()
    assert result[1] in {LABEL_CONFIRMED, LABEL_CROSSEDOUT}


def test_stage2_class0_maps_to_confirmed():
    cells = _make_cells(1)
    clf = _make_classifier([1], stage2_class_sequence=[0])
    result = clf.predict(cells)
    assert result == [LABEL_CONFIRMED]


def test_stage2_class1_maps_to_crossedout():
    cells = _make_cells(1)
    clf = _make_classifier([1], stage2_class_sequence=[1])
    result = clf.predict(cells)
    assert result == [LABEL_CROSSEDOUT]


def test_mixed_predictions():
    cells = _make_cells(4)
    clf = _make_classifier([0, 1, 0, 1], stage2_class_sequence=[0, 1])
    result = clf.predict(cells)
    assert result[0] == LABEL_EMPTY
    assert result[1] == LABEL_CONFIRMED
    assert result[2] == LABEL_EMPTY
    assert result[3] == LABEL_CROSSEDOUT


def test_sift_histogram_shape():
    sift = MagicMock()
    sift.detectAndCompute = MagicMock(return_value=(None, None))
    kmeans = MagicMock()
    kmeans.predict = MagicMock(return_value=np.array([0, 1, 2]))
    img = np.zeros((64, 64), dtype=np.uint8)
    hist = _sift_histogram(img, sift, kmeans, k=200)
    assert hist.shape == (200,)


def test_sift_histogram_no_descriptors_returns_zeros():
    sift = MagicMock()
    sift.detectAndCompute = MagicMock(return_value=(None, None))
    kmeans = MagicMock()
    img = np.zeros((64, 64), dtype=np.uint8)
    hist = _sift_histogram(img, sift, kmeans, k=50)
    assert np.all(hist == 0)


def test_sift_histogram_is_l1_normalised():
    sift = MagicMock()
    descriptors = np.zeros((10, 128), dtype=np.float32)
    sift.detectAndCompute = MagicMock(return_value=(None, descriptors))
    kmeans = MagicMock()
    kmeans.predict = MagicMock(return_value=np.array([0] * 10))
    img = np.zeros((64, 64), dtype=np.uint8)
    hist = _sift_histogram(img, sift, kmeans, k=200)
    assert abs(hist.sum() - 1.0) < 1e-5


def test_load_classifier_downloads_both_model_files():
    import src.classifier as clf_module

    mock_bovw = {
        "kmeans": MagicMock(cluster_centers_=np.zeros((200, 128))),
        "svm": MagicMock(),
    }
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    with patch.object(clf_module, "hf_hub_download", return_value="/tmp/model") as mock_dl, \
         patch.object(clf_module, "joblib") as mock_joblib, \
         patch.object(clf_module.ort, "InferenceSession", return_value=mock_session), \
         patch("cv2.SIFT_create", return_value=MagicMock()):
        clf_module._HF_AVAILABLE = True
        mock_joblib.load.return_value = mock_bovw
        clf_module.load_classifier("user/repo")

    filenames = {c.kwargs.get("filename", c.args[1] if len(c.args) > 1 else "") for c in mock_dl.call_args_list}
    assert "bovw_svm.pkl" in filenames
    assert "cnn_classifier.onnx" in filenames
    assert "cnn_classifier.onnx.data" in filenames


def test_load_classifier_raises_when_deps_missing():
    import src.classifier as clf_module
    original = clf_module._HF_AVAILABLE
    clf_module._HF_AVAILABLE = False
    try:
        with pytest.raises(ImportError):
            clf_module.load_classifier("user/repo")
    finally:
        clf_module._HF_AVAILABLE = original



def test_load_classifier_local_missing_raises(tmp_path):
    """Empty directory (no model files) must raise FileNotFoundError."""
    from src.classifier import load_classifier_local
    with pytest.raises(FileNotFoundError):
        load_classifier_local(tmp_path)


def test_load_classifier_local_loads_models(tmp_path):
    """Fake model files in a temp dir should produce a TwoStageClassifier."""
    import src.classifier as clf_module
    from src.classifier import load_classifier_local

    bovw_path = tmp_path / "bovw_svm.pkl"
    cnn_path = tmp_path / "cnn_classifier.onnx"
    bovw_path.touch()
    cnn_path.touch()

    mock_bovw = {
        "kmeans": MagicMock(cluster_centers_=np.zeros((200, 128))),
        "svm": MagicMock(),
    }
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    with patch.object(clf_module, "joblib") as mock_jl, \
         patch.object(clf_module.ort, "InferenceSession", return_value=mock_session), \
         patch("cv2.SIFT_create", return_value=MagicMock()):
        mock_jl.load.return_value = mock_bovw
        clf = load_classifier_local(tmp_path)

    assert isinstance(clf, TwoStageClassifier)
