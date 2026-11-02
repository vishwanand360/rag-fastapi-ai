from concurrent.futures import ThreadPoolExecutor

from backend.session_store import ConversationStore


def test_sqlite_sessions_serialize_concurrent_writes(tmp_path):
    store = ConversationStore(tmp_path / "history.sqlite3")

    def add_turn(index):
        store.append_turn("alice", "chat", f"q-{index}", f"a-{index}", max_turns=50)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add_turn, range(20)))

    history = store.get_history("alice", "chat")
    assert len(history) == 40
    assert {item["content"] for item in history if item["role"] == "user"} == {f"q-{i}" for i in range(20)}
