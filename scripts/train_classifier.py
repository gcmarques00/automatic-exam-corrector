from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.classifier import _sift_histogram, LABEL_CONFIRMED, LABEL_CROSSEDOUT, LABEL_EMPTY

_BOW_K = 200
_SVM_C = 10.0


def _collect_sift_descriptors(cells: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create()
    pool: list[np.ndarray] = []
    n = len(cells)
    for i, img in enumerate(cells):
        if i % 2000 == 0:
            print(f"  SIFT extraction: {i}/{n}", flush=True)
        _, desc = sift.detectAndCompute(img, None)
        if desc is not None and len(desc) > 0:
            pool.append(desc)
    return np.vstack(pool).astype(np.float32) if pool else np.empty((0, 128), np.float32)


def train_stage1(cells: np.ndarray, labels: np.ndarray, models_dir: Path) -> None:
    """Train BoVW codebook + binary SVM (empty=0 vs filled=1)."""
    print("\n=== Stage 1: SIFT BoVW + SVM (filled vs empty) ===")

    binary = (labels != LABEL_EMPTY).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        cells, binary, test_size=0.2, random_state=42, stratify=binary
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"  empty={int((y_tr == 0).sum())}  filled={int((y_tr == 1).sum())} (train)")

    print("Extracting SIFT descriptors from training cells ...")
    desc_pool = _collect_sift_descriptors(X_tr)
    print(f"  Descriptor pool: {len(desc_pool):,} vectors")

    print(f"Training KMeans codebook (k={_BOW_K}) ...")
    t0 = time.time()
    kmeans = KMeans(n_clusters=_BOW_K, n_init=10, random_state=42)
    kmeans.fit(desc_pool)
    print(f"  Done in {time.time() - t0:.1f}s")

    sift = cv2.SIFT_create()
    print("Computing BoVW histograms ...")
    X_tr_hist = np.array(
        [_sift_histogram(c, sift, kmeans, _BOW_K) for c in X_tr], dtype=np.float32
    )
    X_te_hist = np.array(
        [_sift_histogram(c, sift, kmeans, _BOW_K) for c in X_te], dtype=np.float32
    )

    print("Training SVM ...")
    t0 = time.time()
    svm = SVC(C=_SVM_C, kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_tr_hist, y_tr)
    print(f"  Done in {time.time() - t0:.1f}s")

    y_pred = svm.predict(X_te_hist)
    print("\nStage 1 test results:")
    print(classification_report(y_te, y_pred, target_names=["empty", "filled"]))

    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "bovw_svm.pkl"
    joblib.dump({"kmeans": kmeans, "svm": svm}, path)
    print(f"Saved: {path}")


def train_stage2(cells: np.ndarray, labels: np.ndarray, models_dir: Path) -> None:
    """Train CNN to distinguish confirmed (class 0) from crossed-out (class 1)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("\nSkipping Stage 2 — torch not installed.")
        print("  Install with:  pip install torch  (see requirements-train.txt)")
        return

    print("\n=== Stage 2: CNN (confirmed vs crossed-out) ===")

    mask = labels != LABEL_EMPTY
    cells_f = cells[mask]
    labels_f = labels[mask]

    cnn_labels = np.where(labels_f == LABEL_CONFIRMED, 0, 1).astype(np.int64)
    n_conf = int((cnn_labels == 0).sum())
    n_cross = int((cnn_labels == 1).sum())
    print(f"Filled cells — confirmed: {n_conf:,}  crossed-out: {n_cross:,}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        cells_f, cnn_labels, test_size=0.2, random_state=42, stratify=cnn_labels
    )

    n_tr = len(y_tr)
    w = torch.tensor(
        [n_tr / (2 * (y_tr == 0).sum()), n_tr / (2 * (y_tr == 1).sum())],
        dtype=torch.float32,
    )
    print(f"Loss weights: confirmed={w[0]:.3f}  crossed-out={w[1]:.3f}")

    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x[:, np.newaxis], dtype=torch.float32) / 255.0

    tr_loader = DataLoader(
        TensorDataset(to_tensor(X_tr), torch.tensor(y_tr)),
        batch_size=64, shuffle=True,
    )
    te_loader = DataLoader(
        TensorDataset(to_tensor(X_te), torch.tensor(y_te)),
        batch_size=256,
    )

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.features(x))

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_state: dict | None = None
    n_epochs = 30
    print(f"Training for {n_epochs} epochs ...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        correct = n_total = 0
        with torch.no_grad():
            for xb, yb in te_loader:
                preds = model(xb).argmax(1)
                correct += int((preds == yb).sum().item())
                n_total += len(yb)
        acc = correct / n_total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == n_epochs:
            print(f"  Epoch {epoch:3d}/{n_epochs}  val_acc={acc:.4f}  best={best_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    all_pred: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for xb, yb in te_loader:
            all_pred.extend(model(xb).argmax(1).tolist())
            all_true.extend(yb.tolist())
    print("\nStage 2 test results:")
    print(classification_report(all_true, all_pred, target_names=["confirmed", "crossed-out"]))

    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "cnn_classifier.onnx"
    dummy = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"Saved: {onnx_path}")


def train(dataset_path: Path, models_dir: Path) -> None:
    """Train both classifier stages and write models to models_dir."""
    print(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    cells: np.ndarray = data["images"]
    labels: np.ndarray = data["labels"].astype(int)
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  {len(cells):,} cells  label counts: {counts}")

    stage1_path = models_dir / "bovw_svm.pkl"
    if stage1_path.exists():
        print(f"\nStage 1 model already exists ({stage1_path}) — skipping. Delete it to retrain.")
    else:
        train_stage1(cells, labels, models_dir)

    train_stage2(cells, labels, models_dir)
    print(f"\nAll models written to: {models_dir}/")


if __name__ == "__main__":
    train(
        Path("data/cell_dataset/cells.npz"),
        Path("models"),
    )
