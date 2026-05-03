"""Tests for src/preprocess.py.

Verifies: output is binary, correct dimensions preserved, both threshold
methods work.
"""

import numpy as np
import pytest

from src.preprocess import adaptive_threshold, apply_gaussian_blur, otsu_threshold, run, to_grayscale


@pytest.fixture()
def gray_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 300), dtype=np.uint8)


@pytest.fixture()
def bgr_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)


def test_to_grayscale_from_bgr(bgr_image: np.ndarray) -> None:
    gray = to_grayscale(bgr_image)
    assert gray.ndim == 2
    assert gray.shape == (200, 300)


def test_to_grayscale_passthrough(gray_image: np.ndarray) -> None:
    result = to_grayscale(gray_image)
    assert result.ndim == 2
    assert result.shape == gray_image.shape


def test_gaussian_blur_preserves_shape(gray_image: np.ndarray) -> None:
    blurred = apply_gaussian_blur(gray_image)
    assert blurred.shape == gray_image.shape
    assert blurred.dtype == np.uint8


def test_adaptive_threshold_is_binary(gray_image: np.ndarray) -> None:
    blurred = apply_gaussian_blur(gray_image)
    binary = adaptive_threshold(blurred)
    unique = set(np.unique(binary).tolist())
    assert unique.issubset({0, 255}), f"Non-binary values found: {unique}"


def test_adaptive_threshold_preserves_shape(gray_image: np.ndarray) -> None:
    blurred = apply_gaussian_blur(gray_image)
    binary = adaptive_threshold(blurred)
    assert binary.shape == gray_image.shape


def test_otsu_threshold_is_binary(gray_image: np.ndarray) -> None:
    blurred = apply_gaussian_blur(gray_image)
    binary = otsu_threshold(blurred)
    unique = set(np.unique(binary).tolist())
    assert unique.issubset({0, 255}), f"Non-binary values found: {unique}"


def test_run_adaptive_returns_correct_shapes(bgr_image: np.ndarray) -> None:
    gray, binary = run(bgr_image, method="adaptive")
    assert gray.shape == (200, 300)
    assert binary.shape == (200, 300)


def test_run_otsu_returns_correct_shapes(bgr_image: np.ndarray) -> None:
    gray, binary = run(bgr_image, method="otsu")
    assert gray.shape == (200, 300)
    assert binary.shape == (200, 300)
