import json
from pathlib import Path
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)

_INDEX_TO_LETTER = dict(enumerate("ABCDE"))


def load_key(path: Path) -> dict[int, str]:
    """Load a JSON answer key file.

    Parameters
    ----------
    path : Path
        Path to JSON with schema {"exam_id": str, "answers": {"0": "A", ...}}.
        Keys are 0-based question indices (as strings); values are answer letters.

    Returns
    -------
    dict[int, str]
        Mapping of 0-based question index to expected answer letter (uppercase).

    Raises
    ------
    FileNotFoundError
        If the key file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Answer key not found: {path}")
    with path.open() as f:
        data = json.load(f)
    return {int(k): v.upper() for k, v in data["answers"].items()}


def grade(
    marks: dict[int, int | None],
    key: dict[int, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare detected marks against the answer key and compute the score.

    Parameters
    ----------
    marks : dict[int, int | None]
        Output of bubble_reader.read_marks() — 0-based question → 0-based
        option index, or None if the mark was absent or ambiguous.
    key : dict[int, str]
        Output of load_key() — 0-based question → expected answer letter.
    config : dict
        Project config dict (scoring.correct, scoring.wrong, scoring.unanswered).

    Returns
    -------
    dict
        score, max_score, correct, wrong, unanswered, details.
        details is a list of per-question dicts with keys q, detected, expected, result.
    """
    pts_correct = float(config["scoring"]["correct"])
    pts_wrong = float(config["scoring"]["wrong"])
    pts_unanswered = float(config["scoring"]["unanswered"])

    score = 0.0
    correct = wrong = unanswered = 0
    details = []

    for q, expected in key.items():
        option_idx = marks.get(q)
        detected = _INDEX_TO_LETTER.get(option_idx) if option_idx is not None else None

        if detected is None:
            result = "unanswered"
            score += pts_unanswered
            unanswered += 1
        elif detected == expected:
            result = "correct"
            score += pts_correct
            correct += 1
        else:
            result = "wrong"
            score += pts_wrong
            wrong += 1

        details.append({"q": q, "detected": detected, "expected": expected, "result": result})

    max_score = pts_correct * len(key)
    logger.info(
        "Score: %.1f / %.1f  (correct=%d wrong=%d unanswered=%d)",
        score, max_score, correct, wrong, unanswered,
    )

    return {
        "score": score,
        "max_score": max_score,
        "correct": correct,
        "wrong": wrong,
        "unanswered": unanswered,
        "details": details,
    }
