import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def read_image(path: Path) -> np.ndarray:
    """Read an image from disk in BGR format.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot decode the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_image(img: np.ndarray, path: Path) -> None:
    """Write an image to disk, creating parent directories as needed.

    Args:
        img: Image array to save.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def mat_load_record(mat_path: Path, image_name: str) -> dict:
    """Load bounding boxes and answer key for one answer sheet from exams.mat.

    Parameters
    ----------
    mat_path : Path
        Path to data/dataset/ground_truth/exams.mat.
    image_name : str
        Filename of the answer sheet image (e.g. "exam0_1_1.png").

    Returns
    -------
    dict
        rects : list of np.ndarray, one (n_choices, 4) array per question
        key   : dict[int, str] — 0-based question index → answer letter
        n_questions : int

    Raises
    ------
    FileNotFoundError, KeyError
    """
    from scipy.io import loadmat

    import numpy as np

    if not mat_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {mat_path}")

    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    _letters = "ABCDE"

    for rec in data["records"].flat:
        if str(rec.imageName).strip() != image_name:
            continue

        choices_per_q = np.atleast_1d(rec.questionChoices).astype(int)
        answers = np.atleast_1d(rec.questionAnswer).astype(int)
        rects_flat = np.atleast_1d(rec.questionRect).astype(float)

        rects: list[np.ndarray] = []
        idx = 0
        for nc in choices_per_q:
            rects.append(rects_flat[idx : idx + nc * 4].reshape(nc, 4))
            idx += nc * 4

        key = {i: _letters[int(a) - 1] for i, a in enumerate(answers)}
        return {"rects": rects, "key": key, "n_questions": len(choices_per_q)}

    raise KeyError(f"Image not found in ground truth: {image_name}")


def mat_to_answer_key(mat_path: Path, image_name: str) -> dict[int, str]:
    """Extract the answer key for one answer sheet from the exams.mat ground truth.

    Parameters
    ----------
    mat_path : Path
        Path to data/dataset/ground_truth/exams.mat.
    image_name : str
        Filename of the answer sheet image (e.g. "exam0_1_1.png").

    Returns
    -------
    dict[int, str]
        Mapping of 0-based question index to answer letter (A/B/C/D/E).

    Raises
    ------
    FileNotFoundError
        If mat_path does not exist.
    KeyError
        If image_name is not found in the ground truth records.
    """
    from scipy.io import loadmat

    import numpy as np

    if not mat_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {mat_path}")

    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    records = data["records"]
    _letters = "ABCDE"

    for rec in records.flat:
        if str(rec.imageName).strip() == image_name:
            answers = np.atleast_1d(rec.questionAnswer)
            return {i: _letters[int(a) - 1] for i, a in enumerate(answers)}

    raise KeyError(f"Image not found in ground truth: {image_name}")


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
    return logger
