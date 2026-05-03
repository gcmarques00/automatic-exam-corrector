"""Image acquisition from file, directory, or webcam.

Blur detection uses the variance of the Laplacian operator, following:
  Pech-Pacheco et al. (2000). "Diatom autofocusing in brightfield microscopy:
  a comparative study." ICPR, Vol. 3.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils import get_logger, read_image

logger = get_logger(__name__)

_BLUR_THRESHOLD = 100.0
_MIN_DIM = 200


def _is_blurry(img: np.ndarray, threshold: float = _BLUR_THRESHOLD) -> bool:
    """Return True if the image is too blurry to process reliably.

    The variance of the Laplacian is a fast focus measure: a low variance
    indicates that most pixel intensities are similar (no sharp edges), i.e.
    the image is out of focus (Pech-Pacheco et al., 2000).

    Args:
        img: BGR or grayscale image.
        threshold: Variance below this value is considered blurry.

    Returns:
        True if variance of Laplacian < threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.debug("Laplacian variance: %.2f", variance)
    return float(variance) < threshold


def _validate(img: np.ndarray, path: Path) -> None:
    h, w = img.shape[:2]
    if h < _MIN_DIM or w < _MIN_DIM:
        raise ValueError(f"Image too small ({w}x{h}): {path}")
    if _is_blurry(img):
        logger.warning("Image may be blurry: %s", path)


def from_file(path: Path) -> np.ndarray:
    """Load a single exam sheet image from disk.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image is too small or unreadable.
    """
    img = read_image(path)
    _validate(img, path)
    logger.info("Loaded image: %s (%dx%d)", path.name, img.shape[1], img.shape[0])
    return img


def from_directory(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """Load all supported images from a directory.

    Args:
        directory: Path to directory containing exam sheet images.

    Returns:
        List of (path, image) tuples for each valid image found.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    results: list[tuple[Path, np.ndarray]] = []

    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in extensions:
            continue
        try:
            img = from_file(p)
            results.append((p, img))
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Skipping %s: %s", p.name, e)

    logger.info("Loaded %d image(s) from %s", len(results), directory)
    return results


def from_webcam(camera_index: int = 0) -> np.ndarray:
    """Capture a single frame from the webcam.

    Args:
        camera_index: OpenCV camera device index.

    Returns:
        BGR image as a numpy array.

    Raises:
        RuntimeError: If the camera cannot be opened or the frame is empty.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from webcam")

    logger.info("Captured webcam frame (%dx%d)", frame.shape[1], frame.shape[0])
    return frame
