"""Measure semantic, hybrid, and reranked retrieval against labeled JSONL queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from backend.config import settings
from backend.rag_service import get_service

STRATEGIES = ("semantic", "hybrid", "reranked")


def load_queries(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Line {line_number} needs a non-empty question")
        doc_ids = row.get("relevant_doc_ids", [])
        chunk_ids = row.get("relevant_chunk_ids", [])
        if bool(doc_ids) == bool(chunk_ids):
            raise ValueError(f"Line {line_number} needs either relevant_doc_ids or relevant_chunk_ids")
        ids = doc_ids or chunk_ids
        if not isinstance(ids, list) or not ids or not all(isinstance(value, str) and value for value in ids):
            raise ValueError(f"Line {line_number} relevance IDs must be a non-empty list of strings")
        row["_relevance_field"] = "doc_id" if doc_ids else "id"
        row["_relevant_ids"] = set(ids)
        owner_id = row.get("owner_id", "default")
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError(f"Line {line_number} owner_id must be a non-empty string")
        rows.append(row)
    if not rows:
        raise ValueError("The benchmark dataset is empty")
    return rows


def retrieval_metrics(results: list[dict], relevant_ids: set[str], relevance_field: str, k: int) -> dict[str, float]:
    seen: set[str] = set()
    gains = []
    for result in results[:k]:
        item_id = result.get(relevance_field)
        gain = int(item_id in relevant_ids and item_id not in seen)
        if item_id:
            seen.add(item_id)
        gains.append(gain)
    recall = len(seen & relevant_ids) / len(relevant_ids)
    reciprocal_rank = next((1.0 / rank for rank, gain in enumerate(gains, start=1) if gain), 0.0)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    return {"recall_at_k": recall, "mrr_at_k": reciprocal_rank, "ndcg_at_k": dcg / ideal if ideal else 0.0}


def run_benchmark(rows: list[dict], *, k: int, repeats: int) -> dict:
    service = get_service()
    for owner_id in {row.get("owner_id", "default") for row in rows}:
        documents = service.list_documents(owner_id)
        if not documents:
            raise ValueError(f"No indexed documents are available for owner {owner_id!r}")
        chunks = [chunk for chunk in service._all_chunks() if service._belongs_to_owner(chunk, owner_id)]
        document_ids = {document["id"] for document in documents}
        chunk_ids = {chunk["id"] for chunk in chunks}
        for row in rows:
            if row.get("owner_id", "default") != owner_id:
                continue
            available = document_ids if row["_relevance_field"] == "doc_id" else chunk_ids
            missing = row["_relevant_ids"] - available
            if missing:
                raise ValueError(f"Benchmark labels for owner {owner_id!r} do not match indexed documents/chunks")
    # Warm model/runtime paths once per method, excluding initial downloads/loading from timings.
    for strategy in STRATEGIES:
        service.retrieve(rows[0]["question"], top_k=k, owner_id=rows[0].get("owner_id", "default"), strategy=strategy)

    aggregates = {name: {"metrics": [], "latencies_ms": []} for name in STRATEGIES}
    for row in rows:
        owner_id = row.get("owner_id", "default")
        for strategy in STRATEGIES:
            runs = []
            results = []
            for _ in range(repeats):
                start = time.perf_counter()
                results = service.retrieve(row["question"], top_k=k, owner_id=owner_id, strategy=strategy)
                runs.append((time.perf_counter() - start) * 1000)
            metrics = retrieval_metrics(results, row["_relevant_ids"], row["_relevance_field"], k)
            aggregates[strategy]["metrics"].append(metrics)
            aggregates[strategy]["latencies_ms"].append(statistics.median(runs))

    methods = {}
    for strategy, values in aggregates.items():
        methods[strategy] = {
            metric: statistics.mean(row[metric] for row in values["metrics"])
            for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k")
        }
        methods[strategy]["median_latency_ms"] = statistics.median(values["latencies_ms"])
    return {
        "samples": len(rows),
        "k": k,
        "repeats": repeats,
        "retrieval_candidates": settings.retrieval_candidates,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL with question and relevant doc/chunk IDs")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional path for the measured JSON report")
    args = parser.parse_args()
    if args.k < 1 or args.repeats < 1:
        parser.error("--k and --repeats must be positive")
    try:
        report = run_benchmark(load_queries(args.dataset), k=args.k, repeats=args.repeats)
        report["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
