"""CLI entry point for the automatic exam corrector.

Usage:
  python -m src.main grade --image <path> --key <path> [--debug]
  python -m src.main grade-batch --dir <path> --key <path> [--debug]
  python -m src.main capture --key <path> [--debug]
"""

import argparse
import logging
import sys
from pathlib import Path

from src import capture, grid_detector, perspective, preprocess, utils

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "default.yaml"
_RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s [%(name)s] %(message)s")


def _run_pipeline(
    img,
    config: dict,
    debug: bool,
    output_stem: str,
) -> None:
    """Run the full pipeline on a single image."""
    logger = utils.get_logger(__name__)

    # Preprocessing
    kernel = tuple(config["preprocessing"]["blur_kernel"])
    block = config["preprocessing"]["adaptive_block_size"]
    c = config["preprocessing"]["adaptive_c"]
    gray, binary = preprocess.run(img, kernel_size=kernel, block_size=block, c=c)

    if debug:
        utils.save_image(binary, _RESULTS_DIR / "debug" / f"{output_stem}_binary.png")

    # Grid detection
    min_area = config["grid"]["min_area_ratio"]
    corners = grid_detector.find_grid(binary, min_area_ratio=min_area)

    if corners is None:
        logger.error("Grid not detected — skipping %s", output_stem)
        return

    if debug:
        annotated = grid_detector.draw_grid_contour(img, corners)
        utils.save_image(annotated, _RESULTS_DIR / "debug" / f"{output_stem}_grid.png")

    # Perspective correction
    warped = perspective.warp(img, corners)

    if debug:
        utils.save_image(warped, _RESULTS_DIR / "debug" / f"{output_stem}_warped.png")

    logger.info("Pipeline complete for %s", output_stem)


def cmd_grade(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    img = capture.from_file(args.image)
    _run_pipeline(img, config, args.debug, args.image.stem)


def cmd_grade_batch(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    images = capture.from_directory(args.dir)
    for path, img in images:
        _run_pipeline(img, config, args.debug, path.stem)


def cmd_capture(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    img = capture.from_webcam()
    _run_pipeline(img, config, args.debug, "webcam_capture")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic exam corrector")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_grade = sub.add_parser("grade", help="Grade a single image")
    p_grade.add_argument("--image", type=Path, required=True)
    p_grade.add_argument("--key", type=Path, required=True)
    p_grade.set_defaults(func=cmd_grade)

    p_batch = sub.add_parser("grade-batch", help="Grade all images in a directory")
    p_batch.add_argument("--dir", type=Path, required=True)
    p_batch.add_argument("--key", type=Path, required=True)
    p_batch.set_defaults(func=cmd_grade_batch)

    p_cap = sub.add_parser("capture", help="Capture from webcam and grade")
    p_cap.add_argument("--key", type=Path, required=True)
    p_cap.set_defaults(func=cmd_capture)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.debug)
    args.func(args)


if __name__ == "__main__":
    main()
