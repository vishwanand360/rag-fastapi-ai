import json
import math

import pytest

from scripts.benchmark_retrieval import load_queries, retrieval_metrics


def test_retrieval_metrics_calculate_recall_mrr_and_dcg_at_k():
    results = [{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}]
    metrics = retrieval_metrics(results, {"d1", "d2", "d3"}, "doc_id", 3)
    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == pytest.approx(1.5 / (1 + 1 / math.log2(3) + 1 / math.log2(4)))


def test_benchmark_dataset_requires_relevance_labels(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps({"question": "Find policy"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relevant_doc_ids"):
        load_queries(path)


def test_benchmark_dataset_accepts_document_labels(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps({"question": "Find policy", "relevant_doc_ids": ["doc-1"]}) + "\n", encoding="utf-8")
    row = load_queries(path)[0]
    assert row["_relevance_field"] == "doc_id"
    assert row["_relevant_ids"] == {"doc-1"}
