"""Image preprocessing: grayscale conversion, Gaussian blur, thresholding.

Adapted from:
  ua_computerVision / #03 - Low Level Image Processing I (P. Dias, UA)
  ua_computerVision / #04 - Edges_Lines (P. Dias, UA)

Academic references:
  - Gaussian blur: Gonzalez, R.C. & Woods, R.E. (2018). Digital Image
    Processing (4th ed.). Pearson. §3.4.
  - Adaptive thresholding: Bradley, D. & Roth, G. (2007). "Adaptive
    Thresholding Using the Integral Image." Journal of Graphics Tools, 12(2).
  - Otsu binarization: Otsu, N. (1979). "A Threshold Selection Method from
    Gray-Level Histograms." IEEE Trans. Systems, Man, and Cybernetics, 9(1).
"""

from typing import Literal

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Follows the pattern from ua_computerVision #03:
      cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    Args:
        img: BGR image (H x W x 3).

    Returns:
        Single-channel grayscale image (H x W).
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    gray: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Smooth the image with a Gaussian kernel to suppress noise.

    Follows the pattern from ua_computerVision #04 (aula_04_ex_02.py):
      cv2.GaussianBlur(image, kernel_size, 0)

    Gaussian smoothing is the standard pre-step before thresholding and edge
    detection because it attenuates high-frequency noise while preserving
    low-frequency structure (Gonzalez & Woods, 2018, §3.4).

    Args:
        gray: Grayscale image.
        kernel_size: Gaussian kernel dimensions (must be odd).

    Returns:
        Blurred grayscale image.
    """
    return cv2.GaussianBlur(gray, kernel_size, 0)


def adaptive_threshold(
    blurred: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Binarize using adaptive (local) Gaussian thresholding.

    Follows the pattern from ua_computerVision #03 (aula_03_ex_01.py concept):
      cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, block_size, c)

    Adaptive thresholding computes a local threshold for each pixel
    neighbourhood, making it robust to uneven illumination — the dominant
    artefact when photographing paper sheets with a phone camera (Bradley &
    Roth, 2007).

    Args:
        blurred: Gaussian-blurred grayscale image.
        block_size: Size of the local neighbourhood (must be odd, >= 3).
        c: Constant subtracted from the weighted mean.

    Returns:
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

    Otsu's method finds the threshold that minimises intra-class variance of
    pixel intensities, assuming a bimodal histogram (Otsu, 1979).  Provided
    for comparison against adaptive thresholding (proposal.md Phase 3).

    Args:
        blurred: Gaussian-blurred grayscale image.

    Returns:
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

    Args:
        img: Input BGR image.
        kernel_size: Gaussian blur kernel size.
        block_size: Adaptive threshold neighbourhood size.
        c: Adaptive threshold constant.
        method: "adaptive" (default) or "otsu".

    Returns:
        Tuple of (grayscale image, binary thresholded image).
    """
    gray = to_grayscale(img)
    blurred = apply_gaussian_blur(gray, kernel_size)
    if method == "otsu":
        binary = otsu_threshold(blurred)
    else:
        binary = adaptive_threshold(blurred, block_size, c)
    logger.debug("Preprocessing done — method=%s", method)
    return gray, binary
