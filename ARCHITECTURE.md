# Architecture

This diagram reflects the current FastAPI/Streamlit application. Local Qdrant and SQLite both persist under the Compose `rag_data` volume. Only `GET /health` bypasses bearer authentication; when `RAG_USERS` is configured, the authenticated key maps to the owner ID used for document and conversation scoping.

```mermaid
flowchart TD
    User[User] --> UI[Streamlit dashboard]
    UI -->|Bearer key and request| Auth[FastAPI bearer middleware]
    Auth -->|RAG_USERS identity or default principal| API[FastAPI routes]

    API -->|PDF upload| Extract[PyMuPDF page extraction]
    Extract --> Chunk[Page-aware recursive chunking]
    Chunk --> EmbedDocs[Sentence Transformers document embeddings]
    EmbedDocs -->|Vectors and owner/source/page payload| Qdrant[(Qdrant local collection)]

    API -->|Chat question and owner ID| Chat[Chat request]
    Chat --> Query[Sentence Transformers query embedding]
    Query -->|Owner-filtered dense search| Qdrant
    Chat --> BM25[BM25 over cached chunks for that owner]
    Qdrant --> RRF[Reciprocal Rank Fusion]
    BM25 --> RRF
    RRF --> Reranker[Cross-Encoder reranking]
    Reranker --> Gemini[Gemini answer generation]
    Gemini --> CitationCheck[Validate citation source/page against retrieved context]
    CitationCheck -->|Answer and cited pages| API
    API -->|Read/write by owner ID and session ID| SQLite[(SQLite conversation history)]
    API --> UI

    API -->|POST /evaluation with labeled rows| RAGAS[RAGAS evaluation]
    RAGAS --> Gemini
    RAGAS -->|Faithfulness, answer relevancy, context precision/recall| API

    Qdrant -. persisted in DATA_DIR .-> Volume[(Compose rag_data volume)]
    SQLite -. persisted in DATA_DIR .-> Volume
```

## Retrieval details

- **Semantic stage:** the query embedding searches Qdrant with an owner filter.
- **Hybrid stage:** BM25 ranks the same owner's cached chunk payloads. Reciprocal Rank Fusion combines dense and sparse candidate ranks.
- **Reranked stage:** a Cross-Encoder scores the fused candidates; normal chat uses the reranked results.
- **Citations:** Gemini is prompted to cite source and page labels. The application rejects missing, malformed, or unrecognized references and returns only labels present in the retrieved contexts. This checks reference identity, not whether a cited passage entails each claim.

The benchmark CLI can measure the three retrieval stages against supplied relevance labels. It must use the same Qdrant store and owner IDs as the indexed corpus; local Qdrant cannot be opened by the benchmark and API processes at the same time.
