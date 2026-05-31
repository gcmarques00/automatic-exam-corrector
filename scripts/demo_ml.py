import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2
import numpy as np

from src import cell_extractor, classifier as clf_module, grid_detector, grader, utils, visualizer
from src.classifier import LABEL_CONFIRMED, LABEL_CROSSEDOUT, LABEL_EMPTY

_ROOT = Path(__file__).parent.parent
_DATASET = _ROOT / "data" / "dataset" / "answer_sheets"
_GT = _ROOT / "data" / "dataset" / "ground_truth" / "exams.mat"
_CONFIG = _ROOT / "config" / "default.yaml"
_MODELS = _ROOT / "models"
_MAX_H = 900

_COLOR_CONFIRMED  = (0, 200, 0)
_COLOR_CROSSEDOUT = (0, 140, 255)
_COLOR_EMPTY      = (130, 130, 130)
_COLOR_FILLED     = (0, 220, 220)


def _fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= _MAX_H:
        return img
    scale = _MAX_H / h
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _show(win: str, img: np.ndarray, title: str) -> bool:
    cv2.imshow(win, _fit(img))
    cv2.setWindowTitle(win, title)
    return (cv2.waitKey(0) & 0xFF) != ord("q")


def _gray_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _grid_corners_from_rects(rects: list) -> np.ndarray:
    all_pts = np.vstack([r[:, :2] for r in rects])
    all_br  = np.vstack([r[:, :2] + r[:, 2:4] for r in rects])
    pts = np.vstack([all_pts, all_br])
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _draw_grid(img: np.ndarray, binary: np.ndarray, config: dict, rects: list) -> np.ndarray:
    corners = grid_detector.find_grid(binary, min_area_ratio=config["grid"]["min_area_ratio"])
    if corners is None:
        corners = _grid_corners_from_rects(rects)
    return grid_detector.draw_grid_contour(img, corners, color=(0, 255, 0), thickness=4)


def _draw_rois(img: np.ndarray, rects: list) -> np.ndarray:
    out = img.copy()
    n = len(rects)
    for q, q_rects in enumerate(rects):
        hue = int(q / max(n - 1, 1) * 150)
        color = [int(c) for c in cv2.cvtColor(np.uint8([[[hue, 230, 220]]]), cv2.COLOR_HSV2BGR)[0][0]]
        for x, y, w, h in q_rects.astype(int):
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        x0 = int(min(r[0] for r in q_rects))
        y0 = int(min(r[1] for r in q_rects))
        cv2.putText(out, f"Q{q + 1}", (x0 - 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    return out


def _cell_mosaic(cells: list[np.ndarray], rects: list, cell_size: int = 64) -> np.ndarray:
    """Arrange cell crops in a question-row × choice-col grid."""
    n_choices = max(len(r) for r in rects)
    n_q = len(rects)
    pad = 3
    tile = cell_size + pad
    canvas = np.full((n_q * tile + pad, n_choices * tile + pad), 200, dtype=np.uint8)
    idx = 0
    for q, q_rects in enumerate(rects):
        for c in range(len(q_rects)):
            cell = cells[idx]
            y0 = pad + q * tile
            x0 = pad + c * tile
            canvas[y0: y0 + cell_size, x0: x0 + cell_size] = cell
            idx += 1
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _label_overlay(img: np.ndarray, rects: list, labels: list[int],
                   label_color: dict[int, tuple]) -> np.ndarray:
    """Draw filled semi-transparent rectangles coloured by label."""
    out = img.copy()
    overlay = out.copy()
    idx = 0
    for q_rects in rects:
        for x, y, w, h in q_rects.astype(int):
            color = label_color.get(labels[idx], _COLOR_EMPTY)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            idx += 1
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    return out


def _stage1_labels(classifier, cells: list[np.ndarray]) -> list[int]:
    """Run only Stage 1 (BoVW+SVM): returns LABEL_EMPTY or LABEL_FILLED (synthetic)."""
    from src.classifier import _sift_histogram
    _LABEL_FILLED = 0
    histograms = np.array(
        [_sift_histogram(c, classifier._sift, classifier._kmeans, classifier._k) for c in cells],
        dtype=np.float32,
    )
    stage1 = classifier._svm.predict(histograms)
    return [_LABEL_FILLED if p == 1 else LABEL_EMPTY for p in stage1]


def run_demo(image_name: str) -> None:
    config = utils.load_config(_CONFIG)
    img_path = _DATASET / image_name.split("_")[0] / image_name
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    record = utils.mat_load_record(_GT, image_name)
    rects, key = record["rects"], record["key"]

    kernel = tuple(config["preprocessing"]["blur_kernel"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    _, binary_raw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph_k = int(config["bubble"].get("morph_open_kernel", 5))
    k_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    binary = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, k_elem)

    classifier = clf_module.load_classifier_local(_MODELS)
    cells = cell_extractor.extract_cells(img, rects)

    s1_labels = _stage1_labels(classifier, cells)
    s2_labels = classifier.predict(cells)

    _LABEL_FILLED = 0
    s1_colors = {_LABEL_FILLED: _COLOR_FILLED, LABEL_EMPTY: _COLOR_EMPTY}
    s2_colors = {LABEL_CONFIRMED: _COLOR_CONFIRMED, LABEL_CROSSEDOUT: _COLOR_CROSSEDOUT, LABEL_EMPTY: _COLOR_EMPTY}

    win = "Demo — ML Pipeline"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    steps = [
        (img,                                          "1/8  Raw Answer Sheet"),
        (_gray_to_bgr(gray),                           "2/8  Grayscale Conversion"),
        (_draw_grid(img, binary, config, rects),       "3/8  Answer Grid Detection"),
        (_draw_rois(img, rects),                       "4/8  Regions of Interest (ROIs)"),
        (_cell_mosaic(cells, rects),                   "5/8  Cell Crops (64x64) — input to classifier"),
        (_label_overlay(img, rects, s1_labels, s1_colors), "6/8  Stage 1 — BoVW+SVM: empty vs filled"),
        (_label_overlay(img, rects, s2_labels, s2_colors), "7/8  Stage 2 — CNN: confirmed / crossed-out / empty"),
        (None,                                         "8/8  Grading Result"),
    ]

    for frame, title in steps:
        if frame is None:
            from src import bubble_reader
            marks = bubble_reader.read_marks_from_rects(img, rects, config, classifier=classifier)
            results = grader.grade(marks, key, config)
            frame = visualizer.annotate_rects(img, rects, marks, results, config)
        if not _show(win, frame, title):
            break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="exam0_1_1.png")
    args = parser.parse_args()
    run_demo(args.image)


if __name__ == "__main__":
    main()
