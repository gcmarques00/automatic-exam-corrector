import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_ASPECT = 1.414


def compute_output_size(corners: np.ndarray, aspect_ratio: float = _DEFAULT_ASPECT) -> tuple[int, int]:
    """Compute output rectangle dimensions from the detected corners.

    Parameters
    ----------
    corners : np.ndarray
        Ordered (4, 2) corner array [TL, TR, BR, BL].
    aspect_ratio : float
        height / width ratio of the canonical output sheet.

    Returns
    -------
    tuple of int
        (width, height) in pixels.
    """
    tl, tr, br, bl = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))
    height = int(width * aspect_ratio)
    return width, height


def warp(
    img: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] | None = None,
    aspect_ratio: float = _DEFAULT_ASPECT,
) -> np.ndarray:
    """Rectify a perspective-distorted answer sheet to a frontal view.

    Computes a homography via cv2.getPerspectiveTransform (Direct Linear
    Transform, Hartley & Zisserman, 2003, §4.1) and warps the image.

    Parameters
    ----------
    img : np.ndarray
        BGR source image.
    corners : np.ndarray
        Ordered (4, 2) float32 corner array [TL, TR, BR, BL].
    output_size : tuple of int, optional
        (width, height) of the output. If None, derived from corners.
    aspect_ratio : float
        height / width ratio used when output_size is None.

    Returns
    -------
    np.ndarray
        Warped (rectified) BGR image.
    """
    if output_size is None:
        output_size = compute_output_size(corners, aspect_ratio)

    width, height = output_size

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (width, height))

    logger.info("Perspective warp applied — output size: %dx%d", width, height)
    return warped
