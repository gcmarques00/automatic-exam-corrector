"""Detect the answer-grid bounding quadrilateral in a thresholded image.

Adapted from:
  ua_computerVision / #04 - Edges_Lines (P. Dias, UA)
  ua_computerVision / #05 - Morph_Segmentation (P. Dias, UA)

Academic references:
  - Border-following contour algorithm: Suzuki, S. & Abe, K. (1985).
    "Topological Structural Analysis of Digitized Binary Images by Border
    Following." Computer Vision, Graphics, and Image Processing, 30(1), 32-46.
  - Douglas-Peucker polygon approximation: Douglas, D. & Peucker, T. (1973).
    "Algorithms for the Reduction of the Number of Points Required to
    Represent a Digitized Line or Its Caricature." Cartographica, 10(2).
"""

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left].

    Args:
        pts: Array of shape (4, 2) with (x, y) corner coordinates.

    Returns:
        Ordered array of shape (4, 2).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff

    return rect


def find_grid(
    binary: np.ndarray,
    min_area_ratio: float = 0.1,
) -> np.ndarray | None:
    """Find the four corners of the answer-grid rectangle.

    The approach follows ua_computerVision #04 and #05:
      1. cv2.findContours — border-following algorithm (Suzuki & Abe, 1985).
      2. Filter by area to discard noise (contour must cover at least
         `min_area_ratio` of the image).
      3. cv2.approxPolyDP — Douglas-Peucker simplification (Douglas &
         Peucker, 1973) to reduce the contour to its polygon vertices.
      4. Accept only quadrilaterals (4 vertices) as grid candidates.

    The answer grid is expected to be the largest quadrilateral in the image.

    Args:
        binary: Binary (thresholded) image — white foreground on black.
        min_area_ratio: Minimum fraction of image area a contour must cover.

    Returns:
        Ordered corner array of shape (4, 2) as float32, or None if not found.
    """
    image_area = binary.shape[0] * binary.shape[1]
    min_area = image_area * min_area_ratio

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No contours found in binary image")
        return None

    # Sort largest-first; the grid is the dominant rectangular region
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break  # remaining contours are smaller — stop early

        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) == 4:
            logger.info("Grid contour found — area=%.0f px²", area)
            return _order_corners(approx)

    logger.warning("No quadrilateral grid contour found")
    return None


def draw_grid_contour(
    img: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw the detected grid outline on a copy of the image.

    Args:
        img: BGR image to annotate.
        corners: Ordered (4, 2) corner array from find_grid().
        color: BGR line colour.
        thickness: Line thickness in pixels.

    Returns:
        Annotated copy of the image.
    """
    out = img.copy()
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out
