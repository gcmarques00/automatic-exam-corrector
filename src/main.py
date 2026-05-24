import argparse
import logging
import sys
from pathlib import Path

from src import bubble_reader, capture, grader, grid_detector, perspective, preprocess, utils, visualizer

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "default.yaml"
_RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
_DEFAULT_DATASET_DIR = Path(__file__).parent.parent / "data" / "dataset" / "answer_sheets"
_DEFAULT_GT = Path(__file__).parent.parent / "data" / "dataset" / "ground_truth" / "exams.mat"


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s [%(name)s] %(message)s")


def _run_pipeline(
    img,
    config: dict,
    key_path: Path,
    debug: bool,
    output_stem: str,
) -> None:
    """Run the full pipeline on a single image."""
    logger = utils.get_logger(__name__)

    kernel = tuple(config["preprocessing"]["blur_kernel"])
    block = config["preprocessing"]["adaptive_block_size"]
    c = config["preprocessing"]["adaptive_c"]
    gray, binary = preprocess.run(img, kernel_size=kernel, block_size=block, c=c)

    if debug:
        utils.save_image(binary, _RESULTS_DIR / "debug" / f"{output_stem}_binary.png")

    min_area = config["grid"]["min_area_ratio"]
    corners = grid_detector.find_grid(binary, min_area_ratio=min_area)

    if corners is None:
        logger.error("Grid not detected — skipping %s", output_stem)
        return

    if debug:
        annotated = grid_detector.draw_grid_contour(img, corners)
        utils.save_image(annotated, _RESULTS_DIR / "debug" / f"{output_stem}_grid.png")

    warped = perspective.warp(img, corners)

    if debug:
        utils.save_image(warped, _RESULTS_DIR / "debug" / f"{output_stem}_warped.png")

    marks = bubble_reader.read_marks(warped, config)

    key = grader.load_key(key_path)
    results = grader.grade(marks, key, config)

    annotated = visualizer.annotate(warped, marks, results, config)
    utils.save_image(annotated, _RESULTS_DIR / f"{output_stem}_result.png")

    logger.info("Pipeline complete for %s", output_stem)


def cmd_grade(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    img = capture.from_file(args.image)
    _run_pipeline(img, config, args.key, args.debug, args.image.stem)


def cmd_grade_batch(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    images = capture.from_directory(args.dir)
    for path, img in images:
        _run_pipeline(img, config, args.key, args.debug, path.stem)


def cmd_grade_dataset(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    img = capture.from_file(args.image)
    logger = utils.get_logger(__name__)

    record = utils.mat_load_record(args.ground_truth, args.image.name)
    rects = record["rects"]
    key = record["key"]

    marks = bubble_reader.read_marks_from_rects(img, rects, config)
    results = grader.grade(marks, key, config)
    annotated = visualizer.annotate_rects(img, rects, marks, results, config)

    output_stem = args.image.stem
    utils.save_image(annotated, _RESULTS_DIR / f"{output_stem}_result.png")

    if args.debug:
        for q, detail in enumerate(results["details"]):
            logger.debug(
                "Q%d: detected=%s expected=%s → %s",
                q + 1, detail["detected"], detail["expected"], detail["result"],
            )

    logger.info(
        "Score: %.1f / %.1f — saved to data/results/%s_result.png",
        results["score"], results["max_score"], output_stem,
    )


def cmd_evaluate_dataset(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    logger = utils.get_logger(__name__)

    images = sorted(args.dataset_dir.glob("**/*.png"))
    total_correct = 0
    total_questions = 0
    skipped = 0

    for img_path in images:
        try:
            img = capture.from_file(img_path)
            record = utils.mat_load_record(args.ground_truth, img_path.name)
            marks = bubble_reader.read_marks_from_rects(img, record["rects"], config)
            results = grader.grade(marks, record["key"], config)

            n_correct = results["correct"]
            n_total = len(record["key"])
            print(f"{img_path.stem}: {n_correct}/{n_total} correct answers")

            total_correct += n_correct
            total_questions += n_total
        except Exception as e:
            logger.warning("Skipped %s: %s", img_path.name, e)
            skipped += 1

    processed = len(images) - skipped
    print(f"\nProcessed {processed} images ({skipped} skipped)")
    print(f"Total: {total_correct}/{total_questions} correct answers")


def cmd_capture(args: argparse.Namespace) -> None:
    config = utils.load_config(args.config)
    img = capture.from_webcam()
    _run_pipeline(img, config, args.key, args.debug, "webcam_capture")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic exam corrector")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_grade = sub.add_parser("grade", help="Grade a single image")
    p_grade.add_argument("--image", type=Path, required=True)
    p_grade.add_argument("--key", type=Path, required=True)
    p_grade.add_argument("--debug", action="store_true")
    p_grade.set_defaults(func=cmd_grade)

    p_batch = sub.add_parser("grade-batch", help="Grade all images in a directory")
    p_batch.add_argument("--dir", type=Path, required=True)
    p_batch.add_argument("--key", type=Path, required=True)
    p_batch.add_argument("--debug", action="store_true")
    p_batch.set_defaults(func=cmd_grade_batch)

    p_ds = sub.add_parser("grade-dataset", help="Grade using exams.mat ground-truth coordinates")
    p_ds.add_argument("--image", type=Path, required=True)
    p_ds.add_argument("--ground-truth", type=Path, required=True)
    p_ds.add_argument("--debug", action="store_true")
    p_ds.set_defaults(func=cmd_grade_dataset)

    p_eval = sub.add_parser("evaluate-dataset", help="Evaluate accuracy across all dataset images")
    p_eval.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    p_eval.add_argument("--ground-truth", type=Path, default=_DEFAULT_GT)
    p_eval.set_defaults(func=cmd_evaluate_dataset)

    p_cap = sub.add_parser("capture", help="Capture from webcam and grade")
    p_cap.add_argument("--key", type=Path, required=True)
    p_cap.add_argument("--debug", action="store_true")
    p_cap.set_defaults(func=cmd_capture)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.debug)
    args.func(args)


if __name__ == "__main__":
    main()
