import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left].

    Parameters
    ----------
    pts : np.ndarray
        Array of shape (4, 2) with (x, y) corner coordinates.

    Returns
    -------
    np.ndarray
        Ordered array of shape (4, 2).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def find_grid(
    binary: np.ndarray,
    min_area_ratio: float = 0.1,
) -> np.ndarray | None:
    """Find the four corners of the answer-grid rectangle.

    Uses cv2.findContours (Suzuki & Abe, 1985), filters by area, then
    cv2.approxPolyDP (Douglas & Peucker, 1973) to reduce the contour to
    polygon vertices. Accepts only quadrilaterals as grid candidates.

    Parameters
    ----------
    binary : np.ndarray
        Binary (thresholded) image — white foreground on black.
    min_area_ratio : float
        Minimum fraction of image area a contour must cover.

    Returns
    -------
    np.ndarray or None
        Ordered corner array of shape (4, 2) as float32, or None if not found.
    """
    image_area = binary.shape[0] * binary.shape[1]
    min_area = image_area * min_area_ratio

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No contours found in binary image")
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break

        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) == 4:
            logger.info("Grid contour found — area=%.0f px²", area)
            return _order_corners(approx)

    logger.warning("No quadrilateral grid contour found")
    return None


def find_grid_canny(
    gray: np.ndarray,
    min_area_ratio: float = 0.1,
    canny_lo: int = 50,
    canny_hi: int = 150,
    hough_threshold: int = 80,
    min_line_len: int = 150,
    max_line_gap: int = 30,
) -> np.ndarray | None:
    """Find grid corners using Canny edge detection and Hough line transform.

    Edge-based alternative to find_grid(). Uses Canny (1986) edges and the
    probabilistic Hough transform (Matas et al., 2000) to find long straight
    line segments, then computes the outer bounding rectangle from extreme lines.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (single channel).
    min_area_ratio : float
        Minimum fraction of image area the detected rectangle must cover.
    canny_lo : int
        Lower hysteresis threshold for Canny.
    canny_hi : int
        Upper hysteresis threshold for Canny.
    hough_threshold : int
        Minimum accumulator vote count for Hough.
    min_line_len : int
        Minimum line segment length in pixels.
    max_line_gap : int
        Maximum gap to bridge between collinear segments.

    Returns
    -------
    np.ndarray or None
        Ordered corner array of shape (4, 2) as float32, or None if not found.
    """
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        logger.warning("Canny/Hough: no lines detected")
        return None

    horiz_ys: list[int] = []
    vert_xs: list[int] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if angle < 20:
            horiz_ys.extend([y1, y2])
        elif angle > 70:
            vert_xs.extend([x1, x2])

    if not horiz_ys or not vert_xs:
        logger.warning(
            "Canny/Hough: insufficient directional lines (horiz=%d, vert=%d)",
            len(horiz_ys), len(vert_xs),
        )
        return None

    top_y  = int(np.percentile(horiz_ys,  5))
    bot_y  = int(np.percentile(horiz_ys, 95))
    left_x = int(np.percentile(vert_xs,   5))
    right_x = int(np.percentile(vert_xs, 95))

    area = (bot_y - top_y) * (right_x - left_x)
    if area < h * w * min_area_ratio:
        logger.warning(
            "Canny/Hough: bounding rectangle too small (area=%d)", area
        )
        return None

    corners = np.array(
        [[left_x, top_y], [right_x, top_y], [right_x, bot_y], [left_x, bot_y]],
        dtype=np.float32,
    )
    logger.info("Canny/Hough grid found — area=%.0f px²", float(area))
    return _order_corners(corners)


def draw_grid_contour(
    img: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw the detected grid outline on a copy of the image.

    Parameters
    ----------
    img : np.ndarray
        BGR image to annotate.
    corners : np.ndarray
        Ordered (4, 2) corner array from find_grid().
    color : tuple of int
        BGR line colour.
    thickness : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        Annotated copy of the image.
    """
    out = img.copy()
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out
