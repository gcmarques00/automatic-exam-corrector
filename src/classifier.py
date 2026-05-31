from __future__ import annotations

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

LABEL_CONFIRMED = 1
LABEL_CROSSEDOUT = 2
LABEL_EMPTY = 3

_BOVW_FILENAME = "bovw_svm.pkl"
_CNN_FILENAME = "cnn_classifier.onnx"


try:
    import joblib
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    joblib = None
    ort = None
    hf_hub_download = None
    _HF_AVAILABLE = False


def load_classifier_local(models_dir: "Path") -> "TwoStageClassifier":
    """Load models from a local directory and return a ready TwoStageClassifier.

    Parameters
    ----------
    models_dir : Path
        Directory containing bovw_svm.pkl and cnn_classifier.onnx.
    """
    from pathlib import Path as _Path
    models_dir = _Path(models_dir)
    bovw_path = models_dir / _BOVW_FILENAME
    cnn_path = models_dir / _CNN_FILENAME
    if not bovw_path.exists():
        raise FileNotFoundError(bovw_path)
    if not cnn_path.exists():
        raise FileNotFoundError(cnn_path)
    bovw_model = joblib.load(bovw_path)
    cnn_session = ort.InferenceSession(str(cnn_path), providers=["CPUExecutionProvider"])
    logger.info("Local models loaded from %s", models_dir)
    return TwoStageClassifier(bovw_model, cnn_session)


def load_classifier(hf_repo: str) -> "TwoStageClassifier":
    """Download models from HuggingFace Hub and return a ready TwoStageClassifier.

    Models are cached locally by huggingface_hub (default: ~/.cache/huggingface/).

    Parameters
    ----------
    hf_repo : str
        HuggingFace repo id, e.g. "username/exam-answer-classifier".
    """
    if not _HF_AVAILABLE:
        raise ImportError(
            "ML classifier requires scikit-learn and huggingface_hub: "
            "pip install scikit-learn huggingface_hub onnxruntime"
        )
    logger.info("Downloading models from %s", hf_repo)
    bovw_path = hf_hub_download(repo_id=hf_repo, filename=_BOVW_FILENAME)
    cnn_path = hf_hub_download(repo_id=hf_repo, filename=_CNN_FILENAME)
    try:
        hf_hub_download(repo_id=hf_repo, filename=_CNN_FILENAME + ".data")
    except Exception:
        pass

    bovw_model = joblib.load(bovw_path)
    cnn_session = ort.InferenceSession(cnn_path, providers=["CPUExecutionProvider"])
    logger.info("Models loaded")
    return TwoStageClassifier(bovw_model, cnn_session)


def _sift_histogram(img: np.ndarray, sift, kmeans, k: int) -> np.ndarray:
    """L1-normalised visual word histogram for a single 64×64 grayscale cell."""
    _, descriptors = sift.detectAndCompute(img, None)
    hist = np.zeros(k, dtype=np.float32)
    if descriptors is not None and len(descriptors) > 0:
        word_ids = kmeans.predict(descriptors)
        for wid in word_ids:
            hist[wid] += 1
        total = hist.sum()
        if total > 0:
            hist /= total
    return hist


class TwoStageClassifier:
    """Two-stage answer cell classifier (Afifi & Hussain IJDAR 2019, Strategy B).

    Parameters
    ----------
    bovw_model : dict with keys "kmeans" (fitted KMeans) and "svm" (fitted SVC)
    cnn_session : onnxruntime.InferenceSession for the CNN model
    """

    def __init__(self, bovw_model: dict, cnn_session) -> None:
        self._kmeans = bovw_model["kmeans"]
        self._svm = bovw_model["svm"]
        self._k = len(self._kmeans.cluster_centers_)
        self._cnn = cnn_session
        self._cnn_input = cnn_session.get_inputs()[0].name
        self._sift = cv2.SIFT_create()

    def predict(self, cells: list[np.ndarray]) -> list[int]:
        """Classify each cell as confirmed (1), crossed-out (2), or empty (3).

        Parameters
        ----------
        cells : list of (64, 64) uint8 grayscale arrays

        Returns
        -------
        list of int, one label per cell
        """
        if not cells:
            return []

        histograms = np.array(
            [_sift_histogram(c, self._sift, self._kmeans, self._k) for c in cells],
            dtype=np.float32,
        )
        stage1 = self._svm.predict(histograms)

        labels: list[int] = [LABEL_EMPTY] * len(cells)
        filled_idx = [i for i, p in enumerate(stage1) if p == 1]

        if not filled_idx:
            return labels

        for i in filled_idx:
            blob = cells[i].astype(np.float32)[np.newaxis, np.newaxis] / 255.0
            out = self._cnn.run(None, {self._cnn_input: blob})[0]
            cls = int(np.argmax(out[0]))
            labels[i] = LABEL_CONFIRMED if cls == 0 else LABEL_CROSSEDOUT

        return labels
