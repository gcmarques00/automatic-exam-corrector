import numpy as np
import pytest

from src.cell_extractor import extract_cells


def _make_img(h: int = 400, w: int = 300, color: bool = True) -> np.ndarray:
    img = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8) if color else np.random.randint(100, 200, (h, w), dtype=np.uint8)
    return img


def _make_rects(n_questions: int = 4, n_choices: int = 3, cell_h: int = 80, cell_w: int = 80) -> list[np.ndarray]:
    rects = []
    for q in range(n_questions):
        q_rects = []
        for c in range(n_choices):
            x = c * (cell_w + 5)
            y = q * (cell_h + 5)
            q_rects.append([x, y, cell_w, cell_h])
        rects.append(np.array(q_rects, dtype=np.float32))
    return rects


def test_extract_cells_count():
    img = _make_img()
    rects = _make_rects(n_questions=4, n_choices=3)
    cells = extract_cells(img, rects)
    assert len(cells) == 12


def test_extract_cells_output_size():
    img = _make_img()
    rects = _make_rects()
    target = (64, 64)
    cells = extract_cells(img, rects, target_size=target)
    for cell in cells:
        assert cell.shape == target, f"Expected {target}, got {cell.shape}"


def test_extract_cells_returns_grayscale():
    img = _make_img(color=True)
    rects = _make_rects()
    cells = extract_cells(img, rects)
    for cell in cells:
        assert cell.ndim == 2, "Expected 2D grayscale array"


def test_extract_cells_no_binarization():
    img = _make_img()
    rects = _make_rects()
    cells = extract_cells(img, rects)
    all_values = np.concatenate([c.ravel() for c in cells])
    assert len(np.unique(all_values)) > 2, "Output looks binarized — expected continuous grayscale values"


def test_extract_cells_accepts_grayscale_input():
    img = _make_img(color=False)
    rects = _make_rects()
    cells = extract_cells(img, rects)
    assert len(cells) == 12
    for cell in cells:
        assert cell.ndim == 2


def test_extract_cells_custom_target_size():
    img = _make_img()
    rects = _make_rects()
    cells = extract_cells(img, rects, target_size=(32, 32))
    for cell in cells:
        assert cell.shape == (32, 32)
