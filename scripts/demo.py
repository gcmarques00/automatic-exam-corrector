import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2
import numpy as np

from src import bubble_reader, grid_detector, grader, utils, visualizer

_ROOT = Path(__file__).parent.parent
_DATASET = _ROOT / "data" / "dataset" / "answer_sheets"
_GT = _ROOT / "data" / "dataset" / "ground_truth" / "exams.mat"
_CONFIG = _ROOT / "config" / "default.yaml"
_MAX_H = 900


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


def _draw_boxes(img: np.ndarray, rects: list) -> np.ndarray:
    out = img.copy()
    for q_rects in rects:
        for x, y, w, h in q_rects.astype(int):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
    return out


def _fill_heatmap(img: np.ndarray, binary: np.ndarray, rects: list, config: dict) -> np.ndarray:
    out = img.copy()
    inset = int(config["bubble"].get("cell_inset", 0))
    cell_data = []
    for q_rects in rects:
        for x, y, w, h in q_rects.astype(int):
            cell = binary[y + inset: y + h - inset, x + inset: x + w - inset]
            ratio = cv2.countNonZero(cell) / cell.size if cell.size > 0 else 0.0
            cell_data.append((x, y, w, h, ratio))
    max_r = max(d[4] for d in cell_data) if cell_data else 1.0
    for x, y, w, h, ratio in cell_data:
        norm = ratio / max_r if max_r > 0 else 0.0
        color = (int(255 * (1 - norm)), 40, int(255 * norm))
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{ratio * 100:.0f}%", (x + 3, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    return out


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

    win = "Demo — Threshold Pipeline"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    steps = [
        (img,                           "1/9  Raw Answer Sheet"),
        (_gray_to_bgr(gray),            "2/9  Grayscale Conversion"),
        (_gray_to_bgr(binary_raw),      "3/9  Otsu Threshold (inverted)"),
        (_gray_to_bgr(binary),          "4/9  Morphological Opening"),
        (_draw_grid(img, binary, config, rects), "5/9  Answer Grid Detection"),
        (_draw_rois(img, rects),        "6/9  Regions of Interest (ROIs)"),
        (_draw_boxes(img, rects),       "7/9  Answer Box Detection"),
        (_fill_heatmap(img, binary, rects, config), "8/9  Fill-Ratio Heatmap"),
        (None,                          "9/9  Grading Result"),
    ]

    for frame, title in steps:
        if frame is None:
            marks = bubble_reader.read_marks_from_rects(img, rects, config)
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
