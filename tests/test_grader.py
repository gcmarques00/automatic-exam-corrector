import json
import tempfile
from pathlib import Path

import pytest

from src.grader import grade, load_key

_CONFIG = {
    "scoring": {"correct": 1.0, "wrong": 0.0, "unanswered": 0.0},
}

_KEY = {0: "A", 1: "B", 2: "C"}
_MARKS_ALL_CORRECT = {0: 0, 1: 1, 2: 2}
_MARKS_ALL_WRONG = {0: 1, 1: 2, 2: 0}
_MARKS_ALL_NONE = {0: None, 1: None, 2: None}


def test_all_correct():
    r = grade(_MARKS_ALL_CORRECT, _KEY, _CONFIG)
    assert r["score"] == 3.0
    assert r["correct"] == 3
    assert r["wrong"] == 0
    assert r["unanswered"] == 0


def test_all_wrong_zero_penalty():
    r = grade(_MARKS_ALL_WRONG, _KEY, _CONFIG)
    assert r["score"] == 0.0
    assert r["wrong"] == 3


def test_all_unanswered():
    r = grade(_MARKS_ALL_NONE, _KEY, _CONFIG)
    assert r["score"] == 0.0
    assert r["unanswered"] == 3


def test_mixed_counts_sum_to_total():
    marks = {0: 0, 1: 2, 2: None}  # correct, wrong, unanswered
    r = grade(marks, _KEY, _CONFIG)
    assert r["correct"] + r["wrong"] + r["unanswered"] == len(_KEY)


def test_negative_penalty():
    config = {"scoring": {"correct": 1.0, "wrong": -0.25, "unanswered": 0.0}}
    marks = {0: 0, 1: 2, 2: None}
    r = grade(marks, _KEY, config)
    assert r["score"] == pytest.approx(1.0 - 0.25)


def test_max_score():
    r = grade(_MARKS_ALL_CORRECT, _KEY, _CONFIG)
    assert r["max_score"] == 3.0


def test_load_key(tmp_path):
    key_file = tmp_path / "key.json"
    key_file.write_text(json.dumps({"exam_id": "test", "answers": {"0": "A", "1": "b", "2": "C"}}))
    key = load_key(key_file)
    assert key == {0: "A", 1: "B", 2: "C"}


def test_load_key_missing_file():
    with pytest.raises(FileNotFoundError):
        load_key(Path("/nonexistent/key.json"))
