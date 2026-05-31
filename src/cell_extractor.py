import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def extract_cells(
    img: np.ndarray,
    rects: list[np.ndarray],
    target_size: tuple[int, int] = (64, 64),
) -> list[np.ndarray]:
    """Return one resized grayscale crop per cell, ordered question-by-question.

    Unlike the fill-ratio approach, no binarization is applied — the raw
    grayscale intensity is preserved so that downstream classifiers can use
    texture and gradient information (Afifi & Hussain, IJDAR 2019).

    Parameters
    ----------
    img : np.ndarray
        Full BGR or grayscale exam sheet image.
    rects : list of np.ndarray
        One (n_choices, 4) float array per question — columns are (x, y, w, h).
    target_size : (height, width)
        All crops are resized to this shape using INTER_AREA (anti-aliasing
        when downscaling).

    Returns
    -------
    list of np.ndarray
        Flat list of (H, W) uint8 grayscale arrays, one per cell.
        Order: question 0 choice 0, question 0 choice 1, ..., question N choice M.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    cells: list[np.ndarray] = []
    for q_idx, q_rects in enumerate(rects):
        for c_idx, (x, y, w, h) in enumerate(q_rects.astype(int)):
            crop = gray[y : y + h, x : x + w]
            if crop.size == 0:
                logger.warning("Empty crop at q=%d c=%d rect=(%d,%d,%d,%d)", q_idx, c_idx, x, y, w, h)
                crop = np.full(target_size, 255, dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            cells.append(crop)

    return cells
