import json

from fastapi.testclient import TestClient

import app as api
from backend.session_store import ConversationStore


class FakeService:
    def __init__(self):
        self.documents = {}
        self.histories = []

    def list_documents(self, owner_id="default"):
        return list(self.documents.get(owner_id, {}).values())

    def add_chunks(self, chunks, owner_id="default"):
        docs = self.documents.setdefault(owner_id, {})
        for chunk in chunks:
            docs[chunk["doc_id"]] = {"id": chunk["doc_id"], "filename": chunk["source"], "chunks": 1}
        return {"documents": [{"id": d, "filename": v["filename"]} for d, v in docs.items()], "chunks": len(chunks)}

    def delete_document(self, doc_id, owner_id="default"):
        return self.documents.get(owner_id, {}).pop(doc_id, None) is not None

    def retrieve(self, query, top_k, owner_id="default"):
        return [{"id": "chunk-1", "source": "guide.pdf", "page": 3, "text": "Grounded context."}]

    def answer(self, query, contexts, history):
        self.histories.append(list(history))
        return f"Prior turns: {len(history) // 2} [Source: guide.pdf, p. 3]"


def test_chat_persists_history_and_scopes_it_to_owner(monkeypatch, tmp_path):
    service = FakeService()
    store_path = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "conversation_store", ConversationStore(store_path))
    monkeypatch.setenv("RAG_USERS", json.dumps({"alice": "alice-test-key", "bob": "bob-test-key"}))
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    client = TestClient(api.app)
    alice = {"Authorization": "Bearer alice-test-key"}
    bob = {"Authorization": "Bearer bob-test-key"}
    first = client.post("/chat", json={"question": "What?", "session_id": "shared"}, headers=alice)
    monkeypatch.setattr(api, "conversation_store", ConversationStore(store_path))  # simulate API restart
    bob_reply = client.post("/chat", json={"question": "Why?", "session_id": "shared"}, headers=bob)
    alice_reply = client.post("/chat", json={"question": "Again?", "session_id": "shared"}, headers=alice)
    assert first.status_code == bob_reply.status_code == alice_reply.status_code == 200
    assert first.json()["citations"] == [{"source": "guide.pdf", "page": 3, "chunk_id": "chunk-1"}]
    assert "Prior turns: 0" in bob_reply.json()["answer"]
    assert "Prior turns: 1" in alice_reply.json()["answer"]
    assert service.histories[-1][0]["content"] == "What?"


def test_chat_rejects_blank_question(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client = TestClient(api.app)
    assert client.post("/chat", json={"question": "   "}, headers={"Authorization": "Bearer test-secret"}).status_code == 422


def test_evaluation_validates_dataset(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client = TestClient(api.app)
    response = client.post("/evaluation", json={"rows": [{"question": "incomplete"}]}, headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 422


def test_document_routes_require_bearer_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client = TestClient(api.app)
    assert client.get("/documents").status_code == 401
    assert client.get("/health/gemini").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/docs").status_code == 401
    assert client.post("/evaluation", json={"rows": []}).status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_missing_api_key_fails_closed(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    client = TestClient(api.app)
    response = client.get("/documents")
    assert response.status_code == 503
    assert response.json()["detail"] == "RAG_API_KEY is not configured"


def test_duplicate_user_keys_fail_closed(monkeypatch):
    monkeypatch.setenv("RAG_USERS", json.dumps({"alice": "same-key", "bob": "same-key"}))
    response = TestClient(api.app).get("/documents", headers={"Authorization": "Bearer same-key"})
    assert response.status_code == 503
    assert "distinct API key" in response.json()["detail"]


def test_documents_are_isolated_by_authenticated_owner(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "parse_pdfs", lambda *args, **kwargs: [{"doc_id": "doc-a", "source": "private.pdf", "text": "private", "id": "chunk-a"}])
    monkeypatch.setenv("RAG_USERS", json.dumps({"alice": "alice-test-key", "bob": "bob-test-key"}))
    client = TestClient(api.app)
    alice = {"Authorization": "Bearer alice-test-key"}
    bob = {"Authorization": "Bearer bob-test-key"}

    uploaded = client.post("/documents/upload", headers=alice, files={"files": ("private.pdf", b"pdf bytes", "application/pdf")})
    doc_id = uploaded.json()["documents"][0]["id"]
    assert uploaded.status_code == 200
    assert client.get("/documents", headers=bob).json()["documents"] == []
    assert client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.get("/documents", headers=alice).json()["documents"][0]["id"] == doc_id
    assert client.delete(f"/documents/{doc_id}", headers=alice).status_code == 200


def test_gemini_credential_error_is_clear_and_secret_safe(monkeypatch):
    class InvalidGeminiService(FakeService):
        def validate_gemini_credentials(self):
            raise RuntimeError("Gemini rejected GEMINI_API_KEY. Check key configuration.")

    monkeypatch.setattr(api, "get_service", lambda: InvalidGeminiService())
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    response = TestClient(api.app).get("/health/gemini", headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 503
    assert "Gemini rejected GEMINI_API_KEY" in response.json()["detail"]
    assert "test-secret" not in response.text


def test_chat_withholds_answer_with_unknown_citation(monkeypatch, tmp_path):
    class BadCitationService(FakeService):
        def answer(self, query, contexts, history):
            return "Unsupported claim [Source: another.pdf, p. 1]."

    store = ConversationStore(tmp_path / "sessions.sqlite3")
    monkeypatch.setattr(api, "conversation_store", store)
    monkeypatch.setattr(api, "get_service", lambda: BadCitationService())
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    response = TestClient(api.app).post(
        "/chat", json={"question": "What?", "session_id": "citation-check"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 503
    assert store.get_history("default", "citation-check") == []
