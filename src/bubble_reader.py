import cv2
import numpy as np

from src import cell_extractor
from src.utils import get_logger

logger = get_logger(__name__)


def _to_binary(img: np.ndarray, config: dict) -> np.ndarray:
    kernel = tuple(config["preprocessing"]["blur_kernel"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_size = int(config["bubble"].get("morph_open_kernel", 5))
    return _open_binary(binary, kernel_size)


def _open_binary(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _cell_fill_ratio(
    binary: np.ndarray, row: int, col: int, cell_h: int, cell_w: int, inset: int = 0
) -> float:
    y0, y1 = row * cell_h + inset, (row + 1) * cell_h - inset
    x0, x1 = col * cell_w + inset, (col + 1) * cell_w - inset
    if y1 <= y0 or x1 <= x0:
        return 0.0
    cell = binary[y0:y1, x0:x1]
    return cv2.countNonZero(cell) / cell.size


def _marks_from_ml(
    img: np.ndarray,
    rects: list[np.ndarray],
    classifier,
) -> dict[int, int | None]:
    """Select confirmed answer per question using ML classifier labels.

    Implements the grading logic of Afifi & Hussain (IJDAR 2019) Algorithm 1:
    - One confirmed → answer is that choice.
    - Zero confirmed, one crossed-out → treat crossed-out as the answer.
    - Otherwise → None (ambiguous or blank).
    """
    from src.classifier import LABEL_CONFIRMED, LABEL_CROSSEDOUT

    crops = cell_extractor.extract_cells(img, rects)
    labels = classifier.predict(crops)

    marks: dict[int, int | None] = {}
    idx = 0
    for q, q_rects in enumerate(rects):
        n = len(q_rects)
        q_labels = labels[idx: idx + n]
        idx += n

        confirmed = [i for i, lbl in enumerate(q_labels) if lbl == LABEL_CONFIRMED]
        crossedout = [i for i, lbl in enumerate(q_labels) if lbl == LABEL_CROSSEDOUT]

        if len(confirmed) == 1:
            marks[q] = confirmed[0]
        elif len(confirmed) == 0 and len(crossedout) == 1:
            marks[q] = crossedout[0]
        else:
            marks[q] = None

    return marks


def read_marks_from_rects(
    img: np.ndarray,
    rects: list[np.ndarray],
    config: dict,
    classifier=None,
) -> dict[int, int | None]:
    """Read the marked choice for each question using pre-computed bounding boxes.

    When a classifier is provided, delegates to _marks_from_ml. Otherwise uses
    fill-ratio: for each question, picks the choice whose binary pixel density
    exceeds the second-best by at least config["bubble"]["min_spread"], with a
    scratch-detection heuristic that selects the second-highest cell when the
    gap below it is larger than the gap above. Returns None when no clear mark
    is found.

    Parameters
    ----------
    img : np.ndarray
        Full BGR exam sheet image.
    rects : list of np.ndarray
        One entry per question; each entry is an (n_choices, 4) float array of
        (x, y, w, h) bounding boxes — one bbox per answer choice.
    config : dict
        Project config dict (bubble.min_spread, bubble.morph_open_kernel,
        bubble.cell_inset, preprocessing.*).
    classifier : optional
        If provided, a TwoStageClassifier used instead of fill-ratio.

    Returns
    -------
    dict[int, int | None]
        Mapping of 0-based question index to 0-based choice index, or None.
    """
    if classifier is not None:
        return _marks_from_ml(img, rects, classifier)

    kernel = tuple(config["preprocessing"]["blur_kernel"])
    min_spread = float(config["bubble"].get("min_spread", 0.03))
    inset = int(config["bubble"].get("cell_inset", 0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_size = int(config["bubble"].get("morph_open_kernel", 5))
    binary = _open_binary(binary, kernel_size)

    marks: dict[int, int | None] = {}
    for q, q_rects in enumerate(rects):
        ratios = []
        for x, y, w, h in q_rects.astype(int):
            y0, y1 = y + inset, y + h - inset
            x0, x1 = x + inset, x + w - inset
            if y1 <= y0 or x1 <= x0:
                ratios.append(0.0)
                continue
            cell = binary[y0:y1, x0:x1]
            ratios.append(cv2.countNonZero(cell) / cell.size if cell.size > 0 else 0.0)

        order = sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True)
        vals = [ratios[i] for i in order]

        gap_top = vals[0] - vals[1] if len(vals) > 1 else vals[0]
        gap_below = vals[1] - vals[2] if len(vals) > 2 else 0.0

        if len(vals) >= 3 and gap_below > gap_top and gap_top >= min_spread:
            marks[q] = order[1]
            logger.debug(
                "Question %d: scratch detected (gap_top=%.3f gap_below=%.3f), "
                "picking cell %d (fill=%.3f)",
                q, gap_top, gap_below, order[1], vals[1],
            )
            continue

        best = order[0]
        marks[q] = best if gap_top >= min_spread else None
        if marks[q] is None:
            logger.debug("Question %d: spread=%.3f below min_spread=%.3f", q, gap_top, min_spread)

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
    inset = int(config["bubble"].get("cell_inset", 0))

    binary = _to_binary(warped, config)
    h, w = binary.shape
    cell_h = h // n_questions
    cell_w = w // n_options

    marks: dict[int, int | None] = {}
    for q in range(n_questions):
        marked = [
            col for col in range(n_options)
            if _cell_fill_ratio(binary, q, col, cell_h, cell_w, inset) >= threshold
        ]
        if len(marked) == 1:
            marks[q] = marked[0]
        else:
            marks[q] = None
            if marked:
                logger.debug("Question %d: %d bubbles marked (ambiguous)", q, len(marked))

    return marks
