# automatic-exam-corrector

Automatically grade multiple-choice exam answer sheets using computer vision.

## Requirements

- Python 3.10+
- pip

## Installation

```bash
git clone <repo-url>
cd automatic-exam-corrector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

**Grade a single image** (provide your own exam sheet and answer key):
```bash
exam-corrector grade --image path/to/exam.jpg --key config/answer_keys/sample_key.json
```

**Evaluate accuracy across the full dataset:**
```bash
exam-corrector evaluate-dataset
```

Add `--debug` to any command to save intermediate processing images to `data/results/debug/`.

## Answer key format

A sample key is at `config/answer_keys/sample_key.json`. To grade your own exam, create a JSON file in the same format:

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

## Configuration

Edit `config/default.yaml` to match your exam format. The most relevant fields are `grid.questions` (number of questions) and `grid.options` (choices per question, e.g. 3 for A/B/C).

## Run tests

```bash
pytest tests/ -v
```
