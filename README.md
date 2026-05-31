# automatic-exam-corrector

Automatically grade multiple-choice exam answer sheets using computer vision.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip

## Installation

### With uv (recommended)

```bash
git clone <repo-url>
cd automatic-exam-corrector
uv sync
```

### With pip

```bash
git clone <repo-url>
cd automatic-exam-corrector
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Running on the included dataset

The repository ships with a full dataset under `data/dataset/` and pre-trained models under `models/`.

**Evaluate all exam sheets at once** (runs every image, prints per-image scores):

```bash
# uv
uv run exam-corrector evaluate-dataset

# pip venv
exam-corrector evaluate-dataset
```

**Grade a single sheet from the dataset** (useful for debugging):

```bash
exam-corrector grade-dataset \
  --image data/dataset/answer_sheets/exam0/exam0_1_1.png \
  --ground-truth data/dataset/ground_truth/exams.mat
```

Add `--debug` to save intermediate processing images alongside the result:

```bash
exam-corrector evaluate-dataset --debug
```

All output images are automatically saved to **`data/results/`** — one `<stem>_result.png` per processed sheet. Debug images go to `data/results/debug/`.

## Grading a custom exam image

Provide your own image and an answer-key JSON:

```bash
exam-corrector grade \
  --image path/to/exam.jpg \
  --key config/answer_keys/sample_key.json
```

### Answer key format

```json
{
  "exam_id": "midterm",
  "answers": {
    "1": "A",
    "2": "C",
    "3": "B"
  }
}
```

**Grade a whole folder of custom images:**

```bash
exam-corrector grade-batch \
  --dir path/to/folder/ \
  --key config/answer_keys/sample_key.json
```

## Pipeline overview

The pipeline has two distinct paths depending on the command used:

| Commands | Grid detection | How cell locations are found |
|---|---|---|
| `grade`, `grade-batch`, `capture` | **Yes — runs every time** | Detected from the raw image via contour analysis + perspective warp |
| `grade-dataset`, `evaluate-dataset` | **No** | Pre-computed bounding boxes loaded from `exams.mat` |

Grid detection is not cached between runs. Every call to `grade` or `grade-batch` re-detects the grid from scratch.

## Configuration

Edit `config/default.yaml` to match your exam format. Key fields:

| Field | Default | Description |
|---|---|---|
| `grid.questions` | 20 | Number of questions |
| `grid.options` | 3 | Choices per question (3 → A/B/C) |
| `bubble.fill_threshold` | 0.45 | Minimum fill ratio to count a bubble as marked |
| `bubble.ml_classifier.enabled` | true | Use the trained BoVW+SVM/CNN classifier |

## Run tests

```bash
# uv
uv run pytest tests/ -v

# pip venv
pytest tests/ -v
```
