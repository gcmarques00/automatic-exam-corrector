"""Tests for src/grid_detector.py.

Verifies: finds grid in a synthetic binary image with a clear rectangle,
returns 4 ordered corners, returns None when no grid is present.
"""

import numpy as np
import pytest

from src.grid_detector import _order_corners, draw_grid_contour, find_grid


def _make_binary_with_rect(h: int, w: int, margin: int) -> np.ndarray:
    """Create a binary image with a single filled white rectangle."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[margin : h - margin, margin : w - margin] = 255
    return img


def test_find_grid_detects_rectangle() -> None:
    binary = _make_binary_with_rect(400, 300, 40)
    corners = find_grid(binary, min_area_ratio=0.1)
    assert corners is not None
    assert corners.shape == (4, 2)


def test_find_grid_returns_none_on_blank() -> None:
    blank = np.zeros((400, 300), dtype=np.uint8)
    result = find_grid(blank)
    assert result is None


def test_find_grid_returns_none_when_too_small() -> None:
    binary = _make_binary_with_rect(400, 300, 40)
    # Require at least 90 % of area — the rect covers ~72 % → should fail
    result = find_grid(binary, min_area_ratio=0.9)
    assert result is None


def test_order_corners_top_left_is_first() -> None:
    pts = np.array([[100, 0], [0, 0], [100, 100], [0, 100]], dtype=np.float32)
    ordered = _order_corners(pts)
    assert tuple(ordered[0]) == (0.0, 0.0)   # top-left
    assert tuple(ordered[2]) == (100.0, 100.0)  # bottom-right


def test_draw_grid_contour_returns_same_shape() -> None:
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    corners = np.array([[50, 50], [250, 50], [250, 350], [50, 350]], dtype=np.float32)
    out = draw_grid_contour(img, corners)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)  # lines were drawn
