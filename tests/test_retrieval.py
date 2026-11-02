from types import SimpleNamespace

from qdrant_client import QdrantClient, models

from backend.rag_service import RAGService


class FakeClient:
    def __init__(self, chunks):
        self.chunks = chunks

    def collection_exists(self, collection):
        return True

    def scroll(self, *args, **kwargs):
        return ([SimpleNamespace(payload=chunk) for chunk in self.chunks], None)

    def query_points(self, **kwargs):
        chunk = next(chunk for chunk in self.chunks if chunk["id"] == "chunk-a")
        return SimpleNamespace(points=[SimpleNamespace(payload=chunk)])


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return [FakeVector([0.1, 0.2]) for _ in texts]


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeReranker:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def test_hybrid_retrieval_reranks_and_preserves_page_metadata(monkeypatch):
    chunks = [
        {"id": "chunk-a", "text": "retrieval augmented generation document context", "source": "guide.pdf", "page": 7},
        {"id": "chunk-b", "text": "different topic entirely", "source": "other.pdf", "page": 2},
    ]
    service = RAGService.__new__(RAGService)
    service.client = FakeClient(chunks)
    service.collection = "documents"
    monkeypatch.setattr("backend.rag_service.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("backend.rag_service.get_reranker", lambda: FakeReranker())
    results = service.retrieve("retrieval generation", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "chunk-a"
    assert results[0]["source"] == "guide.pdf"
    assert results[0]["page"] == 7


def test_retrieval_benchmark_modes_compare_each_pipeline_stage(monkeypatch):
    chunks = [
        {"id": "chunk-a", "text": "semantic result", "source": "a.pdf", "page": 1},
        {"id": "chunk-b", "text": "hybrid signal and lexical evidence", "source": "b.pdf", "page": 2},
    ]
    service = RAGService.__new__(RAGService)
    service.client = FakeClient(chunks)
    service.collection = "documents"
    monkeypatch.setattr("backend.rag_service.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("backend.rag_service.get_reranker", lambda: FakeReranker())

    semantic = service.retrieve("hybrid signal", top_k=2, strategy="semantic")
    hybrid = service.retrieve("hybrid signal", top_k=2, strategy="hybrid")
    reranked = service.retrieve("hybrid signal", top_k=2, strategy="reranked")
    assert [item["id"] for item in semantic] == ["chunk-a"]
    assert {item["id"] for item in hybrid} == {"chunk-a", "chunk-b"}
    assert reranked[0]["id"] == "chunk-b"


def test_qdrant_retrieval_and_delete_are_scoped_to_owner(monkeypatch, tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection("documents", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    client.upsert("documents", points=[
        models.PointStruct(id="00000000-0000-0000-0000-000000000001", vector=[1.0, 0.0], payload={
            "id": "same-chunk", "doc_id": "same-document", "owner_id": "alice",
            "text": "private project alpha evidence", "source": "private.pdf", "page": 1,
        }),
        models.PointStruct(id="00000000-0000-0000-0000-000000000002", vector=[0.99, 0.01], payload={
            "id": "same-chunk", "doc_id": "same-document", "owner_id": "bob",
            "text": "private project beta evidence", "source": "private.pdf", "page": 1,
        }),
    ])
    service = RAGService.__new__(RAGService)
    service.client = client
    service.collection = "documents"
    service._chunks_cache = None
    service._bm25_cache = None
    monkeypatch.setattr("backend.rag_service.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("backend.rag_service.get_reranker", lambda: FakeReranker())

    alice_results = service.retrieve("private alpha evidence", top_k=3, owner_id="alice")
    assert alice_results and {chunk["owner_id"] for chunk in alice_results} == {"alice"}
    assert [doc["id"] for doc in service.list_documents("bob")] == ["same-document"]
    assert service.delete_document("same-document", owner_id="alice")
    assert service.list_documents("alice") == []
    assert [doc["id"] for doc in service.list_documents("bob")] == ["same-document"]
    client.close()
