import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_GRAY = (160, 160, 160)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

_RESULT_COLOR = {"correct": _GREEN, "wrong": _RED, "unanswered": _GRAY}


def _draw_score(img: np.ndarray, results: dict) -> None:
    text = f"Score: {results['score']:.0f} / {results['max_score']:.0f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.2, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (10, 10), (20 + tw, 20 + th + 10), _WHITE, cv2.FILLED)
    cv2.putText(img, text, (15, 15 + th), font, scale, _BLACK, thickness, cv2.LINE_AA)


def annotate_rects(
    img: np.ndarray,
    rects: list[np.ndarray],
    marks: dict[int, int | None],
    results: dict,
    config: dict,
) -> np.ndarray:
    """Draw grading overlay on the original image using pre-computed bounding boxes.

    A circle is drawn at the centre of the detected answer choice bbox:
    green for correct, red for wrong, gray for unanswered/ambiguous.
    The score is rendered in the top-left corner.

    Parameters
    ----------
    img : np.ndarray
        Full BGR exam sheet image.
    rects : list of np.ndarray
        One entry per question; each (n_choices, 4) float array of (x,y,w,h) bboxes.
    marks : dict[int, int | None]
        Output of bubble_reader.read_marks_from_rects().
    results : dict
        Output of grader.grade().
    config : dict
        Project config dict (unused directly, kept for API consistency).

    Returns
    -------
    np.ndarray
        Annotated BGR copy of the image.
    """
    out = img.copy()
    detail_by_q = {d["q"]: d for d in results["details"]}

    for q, q_rects in enumerate(rects):
        detail = detail_by_q.get(q)
        result = detail["result"] if detail else "unanswered"
        color = _RESULT_COLOR.get(result, _GRAY)
        opt = marks.get(q)
        col = opt if opt is not None else 0
        x, y, w, h = q_rects[col].astype(int)
        cx, cy = x + w // 2, y + h // 2
        radius = min(w, h) // 3
        cv2.circle(out, (cx, cy), radius, color, thickness=2)

    _draw_score(out, results)
    logger.debug("Annotated image prepared (rects mode)")
    return out


def annotate(
    warped: np.ndarray,
    marks: dict[int, int | None],
    results: dict,
    config: dict,
) -> np.ndarray:
    """Draw grading overlay on a copy of the warped answer sheet.

    A circle is drawn at the centre of each detected answer cell:
    green for correct, red for wrong, gray for unanswered/ambiguous.
    The score is rendered in the top-left corner.

    Parameters
    ----------
    warped : np.ndarray
        BGR image output of perspective.warp().
    marks : dict[int, int | None]
        Output of bubble_reader.read_marks().
    results : dict
        Output of grader.grade().
    config : dict
        Project config dict (grid.questions, grid.options).

    Returns
    -------
    np.ndarray
        Annotated BGR copy of the warped image.
    """
    out = warped.copy()
    n_questions = config["grid"]["questions"]
    n_options = config["grid"]["options"]
    h, w = out.shape[:2]
    cell_h = h // n_questions
    cell_w = w // n_options
    radius = min(cell_h, cell_w) // 3

    detail_by_q = {d["q"]: d for d in results["details"]}

    for q in range(n_questions):
        detail = detail_by_q.get(q)
        color = _RESULT_COLOR.get(detail["result"] if detail else "unanswered", _GRAY)
        opt = marks.get(q)
        col = opt if opt is not None else 0
        cx = col * cell_w + cell_w // 2
        cy = q * cell_h + cell_h // 2
        cv2.circle(out, (cx, cy), radius, color, thickness=2)

    _draw_score(out, results)
    logger.debug("Annotated image prepared")
    return out
