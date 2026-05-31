import cv2
import numpy as np
import pytest

from src.bubble_reader import read_marks

_CONFIG = {
    "grid": {"questions": 4, "options": 3},
    "bubble": {"fill_threshold": 0.45},
    "preprocessing": {"blur_kernel": [5, 5], "adaptive_block_size": 11, "adaptive_c": 2},
}

_H, _W = 400, 300


def _white_bgr() -> np.ndarray:
    return np.full((_H, _W, 3), 255, dtype=np.uint8)


def _fill_cell(img: np.ndarray, row: int, col: int, ratio: float = 0.8) -> np.ndarray:
    cell_h = _H // _CONFIG["grid"]["questions"]
    cell_w = _W // _CONFIG["grid"]["options"]
    y0, y1 = row * cell_h, (row + 1) * cell_h
    x0, x1 = col * cell_w, (col + 1) * cell_w
    pad = int(cell_h * (1 - ratio) / 2)
    out = img.copy()
    out[y0 + pad : y1 - pad, x0 + pad : x1 - pad] = 0
    return out


def test_blank_image_all_none():
    marks = read_marks(_white_bgr(), _CONFIG)
    assert all(v is None for v in marks.values())


def test_single_mark_per_row():
    img = _white_bgr()
    img = _fill_cell(img, row=0, col=1)
    img = _fill_cell(img, row=1, col=0)
    img = _fill_cell(img, row=2, col=2)
    marks = read_marks(img, _CONFIG)
    assert marks[0] == 1
    assert marks[1] == 0
    assert marks[2] == 2
    assert marks[3] is None


def test_ambiguous_row_returns_none():
    img = _fill_cell(_white_bgr(), row=0, col=0)
    img = _fill_cell(img, row=0, col=2)
    marks = read_marks(img, _CONFIG)
    assert marks[0] is None


def test_fill_threshold_respected():
    config_strict = {**_CONFIG, "bubble": {"fill_threshold": 0.95}}
    img = _fill_cell(_white_bgr(), row=0, col=0, ratio=0.5)
    marks = read_marks(img, config_strict)
    assert marks[0] is None


def test_scratch_pattern_not_detected():
    img = _white_bgr()
    cell_h = _H // _CONFIG["grid"]["questions"]
    cell_w = _W // _CONFIG["grid"]["options"]
    y0, y1 = 0, cell_h
    x0, x1 = cell_w, 2 * cell_w
    cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
    cv2.line(img, (x0, y1), (x1, y0), (0, 0, 0), 2)
    config = {**_CONFIG, "bubble": {"fill_threshold": 0.45, "morph_open_kernel": 5}}
    marks = read_marks(img, config)
    assert marks[0] is None


def test_solid_fill_survives_opening():
    img = _fill_cell(_white_bgr(), row=0, col=1)
    config = {**_CONFIG, "bubble": {"fill_threshold": 0.45, "morph_open_kernel": 5}}
    marks = read_marks(img, config)
    assert marks[0] == 1
