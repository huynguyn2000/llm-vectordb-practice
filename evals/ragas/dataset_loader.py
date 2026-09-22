"""Load and validate the hand-labeled Ragas eval dataset."""

import json
from pathlib import Path

DEFAULT_PATH = "evals/ragas/dataset.json"


def load_eval_dataset(path: str = DEFAULT_PATH) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    for i, row in enumerate(rows):
        if not row.get("question", "").strip():
            raise ValueError(f"{path} row {i}: empty 'question'")
        if not row.get("ground_truth", "").strip():
            raise ValueError(f"{path} row {i}: empty 'ground_truth'")
    return rows
