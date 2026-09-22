import pytest

from evals.ragas.dataset_loader import load_eval_dataset


def test_dataset_loads_and_is_well_formed():
    rows = load_eval_dataset()
    assert len(rows) >= 10
    assert all(r["question"].strip() and r["ground_truth"].strip() for r in rows)


def test_loader_rejects_empty_ground_truth(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('[{"question": "q", "ground_truth": ""}]', encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_dataset(str(bad))
