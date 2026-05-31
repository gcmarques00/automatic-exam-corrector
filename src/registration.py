import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

_LOWE_RATIO = 0.75


def register(
    student_img: np.ndarray,
    model_img: np.ndarray,
    min_match_count: int = 10,
) -> np.ndarray | None:
    """Estimate homography that maps student-sheet pixel coordinates to model-sheet coordinates.

    Parameters
    ----------
    student_img : np.ndarray
        BGR or grayscale scan of the student answer sheet.
    model_img : np.ndarray
        BGR or grayscale scan of the model (reference) answer sheet.
    min_match_count : int
        Minimum number of good feature matches required to estimate a homography.
        Returns None if fewer matches are found.

    Returns
    -------
    np.ndarray or None
        3×3 float64 homography matrix M such that
        ``cv2.perspectiveTransform(pts_student, M)`` gives the corresponding
        points on the model sheet. Returns None on failure.

    References
    ----------
    - Lowe, D.G. (2004). Distinctive Image Features from Scale-Invariant
      Keypoints. IJCV 60(2).
    - Hartley, R. & Zisserman, A. (2003). Multiple View Geometry in Computer
      Vision. Cambridge UP.
    """
    gray_s = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY) if student_img.ndim == 3 else student_img
    gray_m = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY) if model_img.ndim == 3 else model_img

    sift = cv2.SIFT_create()
    kp_s, des_s = sift.detectAndCompute(gray_s, None)
    kp_m, des_m = sift.detectAndCompute(gray_m, None)

    if des_s is None or des_m is None or len(kp_s) < min_match_count or len(kp_m) < min_match_count:
        logger.warning("Not enough keypoints: student=%d model=%d", len(kp_s) if kp_s else 0, len(kp_m) if kp_m else 0)
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_s, des_m, k=2)

    good = [m for m, n in raw_matches if m.distance < _LOWE_RATIO * n.distance]
    logger.debug("SIFT matches: raw=%d good=%d", len(raw_matches), len(good))

    if len(good) < min_match_count:
        logger.warning("Too few good matches: %d (need %d)", len(good), min_match_count)
        return None

    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_m = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(pts_s, pts_m, cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None:
        logger.warning("Homography estimation failed")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    logger.debug("Homography inliers: %d / %d", inliers, len(good))
    return M


def transform_rects(
    rects: list[np.ndarray],
    M: np.ndarray,
) -> list[np.ndarray]:
    """Apply homography M to a list of per-question bbox arrays.

    Each bbox (x, y, w, h) is converted to its four corner points, transformed
    by M, then converted back to an axis-aligned (x, y, w, h) by taking the
    bounding box of the transformed corners.

    Parameters
    ----------
    rects : list of np.ndarray
        One (n_choices, 4) array per question — columns (x, y, w, h).
    M : np.ndarray
        3×3 homography matrix from ``register()``.

    Returns
    -------
    list of np.ndarray
        Transformed rects in the same structure as the input.
    """
    transformed: list[np.ndarray] = []
    for q_rects in rects:
        new_q: list[list[float]] = []
        for x, y, w, h in q_rects:
            corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
            tc = cv2.perspectiveTransform(corners, M).reshape(4, 2)
            nx, ny = tc[:, 0].min(), tc[:, 1].min()
            nw, nh = tc[:, 0].max() - nx, tc[:, 1].max() - ny
            new_q.append([nx, ny, nw, nh])
        transformed.append(np.array(new_q, dtype=np.float32))
    return transformed
