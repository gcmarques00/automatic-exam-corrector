"""Read marked bubbles from a perspective-corrected answer sheet.

Adapted from:
  ua_computerVision / #03 - Low Level Image Processing (P. Dias, UA)
  ua_computerVision / #05 - Morph_Segmentation (P. Dias, UA)

Academic references:
  - Pixel-based region analysis: Gonzalez, R.C. & Woods, R.E. (2018).
    Digital Image Processing (4th ed.). Pearson. §10.
  - Morphological operations: Serra, J. (1982). Image Analysis and
    Mathematical Morphology. Academic Press.
"""

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def _to_binary(img: np.ndarray, config: dict) -> np.ndarray:
    kernel = tuple(config["preprocessing"]["blur_kernel"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _cell_fill_ratio(
    binary: np.ndarray, row: int, col: int, cell_h: int, cell_w: int
) -> float:
    y0, y1 = row * cell_h, (row + 1) * cell_h
    x0, x1 = col * cell_w, (col + 1) * cell_w
    cell = binary[y0:y1, x0:x1]
    return cv2.countNonZero(cell) / cell.size


def read_marks_from_rects(
    img: np.ndarray,
    rects: list[np.ndarray],
    config: dict,
) -> dict[int, int | None]:
    """Read the marked choice for each question using pre-computed bounding boxes.

    For each question, computes the fill ratio of every choice bbox and returns
    the index of the maximum-ratio choice, provided its margin over the second-
    best exceeds config["bubble"]["min_spread"].  Returns None when the margin
    is too small (ambiguous or blank).

    Parameters
    ----------
    img : np.ndarray
        Full BGR exam sheet image (not warped).
    rects : list of np.ndarray
        One entry per question; each entry is an (n_choices, 4) float array of
        (x, y, w, h) bounding boxes — one bbox per answer choice.
    config : dict
        Project config dict (bubble.min_spread, preprocessing.*).

    Returns
    -------
    dict[int, int | None]
        Mapping of 0-based question index to 0-based choice index, or None.
    """
    kernel = tuple(config["preprocessing"]["blur_kernel"])
    min_spread = float(config["bubble"].get("min_spread", 0.03))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    marks: dict[int, int | None] = {}
    for q, q_rects in enumerate(rects):
        ratios = []
        for x, y, w, h in q_rects.astype(int):
            cell = binary[y : y + h, x : x + w]
            ratios.append(cv2.countNonZero(cell) / cell.size if cell.size > 0 else 0.0)

        best = int(np.argmax(ratios))
        sorted_r = sorted(ratios, reverse=True)
        spread = sorted_r[0] - sorted_r[1] if len(sorted_r) > 1 else sorted_r[0]

        marks[q] = best if spread >= min_spread else None
        if marks[q] is None:
            logger.debug("Question %d: spread=%.3f below min_spread=%.3f", q, spread, min_spread)

    return marks


def read_marks(warped: np.ndarray, config: dict) -> dict[int, int | None]:
    """Read the marked option for each question row in a warped answer sheet.

    Parameters
    ----------
    warped : np.ndarray
        BGR image output of perspective.warp().
    config : dict
        Project config dict (grid.questions, grid.options, bubble.fill_threshold,
        preprocessing.*).

    Returns
    -------
    dict[int, int | None]
        Mapping of 0-based question index to 0-based option index, or None if
        the row has zero or more than one marked bubble.
    """
    n_questions = config["grid"]["questions"]
    n_options = config["grid"]["options"]
    threshold = config["bubble"]["fill_threshold"]

    binary = _to_binary(warped, config)
    h, w = binary.shape
    cell_h = h // n_questions
    cell_w = w // n_options

    marks: dict[int, int | None] = {}
    for q in range(n_questions):
        marked = [
            col for col in range(n_options)
            if _cell_fill_ratio(binary, q, col, cell_h, cell_w) >= threshold
        ]
        if len(marked) == 1:
            marks[q] = marked[0]
        else:
            marks[q] = None
            if marked:
                logger.debug("Question %d: %d bubbles marked (ambiguous)", q, len(marked))

    return marks
