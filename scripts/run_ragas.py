"""Submit a labeled JSONL dataset to the authenticated RAGAS API endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} must be a JSON object")
        if (not isinstance(row.get("question"), str) or not row["question"].strip() or
                not isinstance(row.get("answer"), str) or not row["answer"].strip() or
                not isinstance(row.get("contexts"), list) or not row["contexts"] or
                not all(isinstance(text, str) and text.strip() for text in row["contexts"]) or
                not isinstance(row.get("ground_truth"), str) or not row["ground_truth"].strip()):
            raise ValueError(f"Line {line_number} needs question, answer, non-empty string contexts, and ground_truth")
        rows.append(row)
    if not 1 <= len(rows) <= 100:
        raise ValueError("RAGAS dataset must contain between 1 and 100 rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL with question, answer, contexts, ground_truth")
    parser.add_argument("--output", type=Path, help="Optional path for measured RAGAS results")
    parser.add_argument("--api-url", default=None, help="Defaults to RAG_API_URL or http://localhost:8000")
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("RAG_EVAL_API_KEY") or os.getenv("RAG_API_KEY")
    if not api_key:
        print("Set RAG_EVAL_API_KEY (or RAG_API_KEY) to an authorized API key.", file=sys.stderr)
        return 2
    try:
        rows = load_dataset(args.dataset)
        dataset_bytes = args.dataset.read_bytes()
        response = requests.post(
            f"{(args.api_url or os.getenv('RAG_API_URL', 'http://localhost:8000')).rstrip('/')}/evaluation",
            headers={"Authorization": f"Bearer {api_key}"}, json={"rows": rows}, timeout=3600,
        )
    except (OSError, ValueError, requests.RequestException) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not response.ok:
        try:
            detail = response.json().get("detail", "Evaluation failed")
        except ValueError:
            detail = "Evaluation failed"
        print(f"HTTP {response.status_code}: {detail}", file=sys.stderr)
        return 1
    report = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "samples": len(rows),
        "results": response.json(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
