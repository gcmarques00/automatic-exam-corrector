"""Perspective correction: warp the detected answer grid to a frontal rectangle.

Adapted from:
  ua_computerVision / #06 - GeometricTransforms_Features (P. Dias, UA)
  Exercise 6.6 — homography estimation with cv2.findHomography and
  cv2.warpPerspective applied to correct oblique views of flat objects.

Academic references:
  - Projective (homography) transform: Hartley, R. & Zisserman, A. (2003).
    Multiple View Geometry in Computer Vision (2nd ed.). Cambridge UP.
    §2.3 — The 2D projective plane and transformations.
  - DLT algorithm for homography estimation: Hartley & Zisserman (2003), §4.1.
"""

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_ASPECT = 1.414  # A4 portrait ratio (ISO 216)


def compute_output_size(corners: np.ndarray, aspect_ratio: float = _DEFAULT_ASPECT) -> tuple[int, int]:
    """Compute output rectangle dimensions from the detected corners.

    Width is derived from the average of the top and bottom edge lengths of
    the detected quadrilateral; height is width × aspect_ratio.

    Args:
        corners: Ordered (4, 2) corner array [TL, TR, BR, BL].
        aspect_ratio: height / width ratio of the canonical output sheet.

    Returns:
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

    Follows ua_computerVision #06 exercise 6.6:
      M = cv2.getPerspectiveTransform(src_pts, dst_pts)
      warped = cv2.warpPerspective(img, M, (width, height))

    The homography H mapping the four detected corners to the four corners of a
    canonical rectangle is computed by cv2.getPerspectiveTransform, which
    solves the Direct Linear Transform (DLT) system (Hartley & Zisserman,
    2003, §4.1).

    Args:
        img: BGR source image.
        corners: Ordered (4, 2) float32 corner array [TL, TR, BR, BL].
        output_size: (width, height) of the output. If None, derived from corners.
        aspect_ratio: height / width ratio used when output_size is None.

    Returns:
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
