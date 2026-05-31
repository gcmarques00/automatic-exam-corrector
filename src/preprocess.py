from typing import Literal

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Parameters
    ----------
    img : np.ndarray
        BGR image (H × W × 3).

    Returns
    -------
    np.ndarray
        Single-channel grayscale image (H × W).
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    gray: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Smooth the image with a Gaussian kernel to suppress noise.

    Attenuates high-frequency noise while preserving low-frequency structure
    (Gonzalez & Woods, 2018, §3.4).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    kernel_size : tuple of int
        Gaussian kernel dimensions (must be odd).

    Returns
    -------
    np.ndarray
        Blurred grayscale image.
    """
    return cv2.GaussianBlur(gray, kernel_size, 0)


def adaptive_threshold(
    blurred: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Binarize using adaptive (local) Gaussian thresholding.

    Computes a local threshold per pixel neighbourhood, making it robust to
    uneven illumination (Bradley & Roth, 2007).

    Parameters
    ----------
    blurred : np.ndarray
        Gaussian-blurred grayscale image.
    block_size : int
        Size of the local neighbourhood (must be odd, >= 3).
    c : int
        Constant subtracted from the weighted mean.

    Returns
    -------
    np.ndarray
        Binary image (255 = foreground / ink, 0 = background).
    """
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def otsu_threshold(blurred: np.ndarray) -> np.ndarray:
    """Binarize using Otsu's global threshold selection.

    Minimises intra-class variance of pixel intensities, assuming a bimodal
    histogram (Otsu, 1979).

    Parameters
    ----------
    blurred : np.ndarray
        Gaussian-blurred grayscale image.

    Returns
    -------
    np.ndarray
        Binary image (255 = foreground / ink, 0 = background).
    """
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def run(
    img: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    block_size: int = 11,
    c: int = 2,
    method: Literal["adaptive", "otsu"] = "adaptive",
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: grayscale → blur → threshold.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    kernel_size : tuple of int
        Gaussian blur kernel size.
    block_size : int
        Adaptive threshold neighbourhood size.
    c : int
        Adaptive threshold constant.
    method : {"adaptive", "otsu"}
        Thresholding method.

    Returns
    -------
    tuple of np.ndarray
        (grayscale image, binary thresholded image).
    """
    gray = to_grayscale(img)
    blurred = apply_gaussian_blur(gray, kernel_size)
    if method == "otsu":
        binary = otsu_threshold(blurred)
    else:
        binary = adaptive_threshold(blurred, block_size, c)
    logger.debug("Preprocessing done — method=%s", method)
    return gray, binary
