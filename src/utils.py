"""Shared helpers: config loading, image I/O, logging setup."""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def read_image(path: Path) -> np.ndarray:
    """Read an image from disk in BGR format.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot decode the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_image(img: np.ndarray, path: Path) -> None:
    """Write an image to disk, creating parent directories as needed.

    Args:
        img: Image array to save.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
    return logger
