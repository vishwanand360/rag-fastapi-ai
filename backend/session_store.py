"""SQLite-backed conversation history, scoped to an authenticated owner."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ConversationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "owner_id TEXT NOT NULL, session_id TEXT NOT NULL, history TEXT NOT NULL, "
                "PRIMARY KEY (owner_id, session_id))"
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def get_history(self, owner_id: str, session_id: str) -> list[dict]:
        with self._connect() as db:
            row = db.execute(
                "SELECT history FROM conversations WHERE owner_id=? AND session_id=?",
                (owner_id, session_id),
            ).fetchone()
        return json.loads(row[0]) if row else []

    def append_turn(self, owner_id: str, session_id: str, question: str, answer: str, max_turns: int) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT history FROM conversations WHERE owner_id=? AND session_id=?",
                (owner_id, session_id),
            ).fetchone()
            history = json.loads(row[0]) if row else []
            history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            history = history[-max_turns * 2:]
            db.execute(
                "INSERT INTO conversations(owner_id,session_id,history) VALUES(?,?,?) "
                "ON CONFLICT(owner_id,session_id) DO UPDATE SET history=excluded.history",
                (owner_id, session_id, json.dumps(history, ensure_ascii=False)),
            )
