import json

import pytest

from scripts.run_ragas import load_dataset


def test_ragas_workflow_loads_labeled_jsonl_rows(tmp_path):
    path = tmp_path / "ragas.jsonl"
    row = {"question": "What is the policy?", "answer": "The policy is local.", "contexts": ["The policy is local."], "ground_truth": "The policy is local."}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_dataset(path) == [row]


def test_ragas_workflow_rejects_incomplete_rows(tmp_path):
    path = tmp_path / "ragas.jsonl"
    path.write_text(json.dumps({"question": "Missing answer"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ground_truth"):
        load_dataset(path)
