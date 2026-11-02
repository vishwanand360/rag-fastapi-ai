# Advanced PDF RAG Platform

> A PDF question-answering application with FastAPI, Streamlit, Qdrant hybrid retrieval, Cross-Encoder reranking, Gemini answers, page citations, and RAGAS evaluation.

The platform ingests multiple PDFs, retrieves page-aware evidence, and returns answers with source references. It is built on the existing PDF RAG project and runs locally or with Docker Compose.

## Features

- Multi-PDF ingestion with PyMuPDF, page-aware chunking, and document metadata.
- Sentence Transformers embeddings stored in persistent local Qdrant.
- Hybrid retrieval: Qdrant semantic search plus BM25, merged with Reciprocal Rank Fusion.
- Cross-Encoder reranking of the fused candidates.
- Gemini responses constrained to retrieved excerpts; citations are checked against retrieved source/page metadata.
- Owner-scoped document operations and SQLite conversation history when `RAG_USERS` is configured.
- FastAPI endpoints and a Streamlit dashboard for documents, chat, citations, and RAGAS evaluation.
- Reproducible retrieval benchmark and RAGAS runner using user-supplied labeled datasets.

Citation validation checks that a source and page were retrieved. It does not prove that the cited text entails every answer claim.

## Technology stack

| Area | Technology |
| --- | --- |
| API | Python 3.11, FastAPI, Uvicorn |
| Dashboard | Streamlit |
| PDF processing | PyMuPDF, LangChain text splitters |
| Embeddings and reranking | Sentence Transformers, PyTorch CPU |
| Vector and lexical retrieval | Qdrant local storage, BM25, Reciprocal Rank Fusion |
| Answer generation and evaluation | Gemini API, RAGAS |
| Conversation history | SQLite |
| Deployment and tests | Docker Compose, pytest |

## System architecture

The Streamlit dashboard calls the authenticated FastAPI service. Uploads are parsed into page-aware chunks and indexed in Qdrant. Chat requests combine owner-filtered semantic retrieval with BM25, fuse the rankings, rerank candidates, and send selected excerpts to Gemini. Validated citations and answers return to the dashboard; session history is persisted by owner and session ID in SQLite.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the Mermaid component and data-flow diagram.

## RAG pipeline

1. **Upload:** `POST /documents/upload` accepts one or more PDFs.
2. **Extract and chunk:** PyMuPDF extracts each page; `RecursiveCharacterTextSplitter` creates overlapping chunks while retaining source, page, document ID, and owner ID.
3. **Embed and index:** Sentence Transformers creates normalized embeddings; Qdrant stores vectors and payload metadata.
4. **Retrieve:** The query is embedded and searched in Qdrant with an owner filter. BM25 ranks that owner's indexed chunks.
5. **Fuse and rerank:** Reciprocal Rank Fusion combines dense and sparse ranks. A Cross-Encoder scores the fused candidates; the top results become context.
6. **Generate and cite:** Gemini answers from those excerpts. The service rejects unsupported or malformed source/page citation labels before returning the answer.
7. **Persist:** Valid chat turns are written to SQLite under `(owner_id, session_id)`.

The semantic, hybrid, and reranked retrieval stages can be measured separately with the benchmark script below. Normal chat uses the reranked stage.

## Project structure

```text
app.py                         FastAPI routes and bearer authentication
dashboard.py                  Streamlit document, chat, and evaluation UI
backend/
  citations.py                 Citation reference validation
  config.py                    Environment-backed settings
  gemini_errors.py             Safe provider error messages
  ingestion.py                 PyMuPDF extraction and page-aware chunking
  rag_service.py               Qdrant, BM25, fusion, reranking, Gemini
  session_store.py             Owner-scoped SQLite conversation history
scripts/
  benchmark_retrieval.py       Semantic/hybrid/reranked benchmark CLI
  run_ragas.py                 Authenticated RAGAS evaluation CLI
tests/                         Ingestion, retrieval, API, citations, and storage tests
Dockerfile                     Python 3.11 image
compose.yaml                   API, dashboard, and persistent data volume
requirements.txt               Runtime dependencies
requirements-dev.txt           Runtime plus test dependencies
requirements-legacy.txt        Dependencies for the retained legacy FAISS UI
.env.example                   Configuration template
```

`chatbot.py`, `rag.py`, and the older FAISS modules are retained as legacy code. The FastAPI application uses `backend.rag_service`.

## Installation

Use Python 3.11. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
[ -f .env ] || cp .env.example .env
```

Edit the newly created `.env` with your provider and API configuration. The copy commands preserve an existing `.env` file.

## Environment configuration

The `.env.example` file lists all supported settings. Important values:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini generation and RAGAS evaluator access. Required for answers and evaluation. |
| `RAG_API_KEY` | Single-principal bearer key and dashboard default. |
| `RAG_USERS` | Optional JSON object mapping user IDs to distinct bearer keys; when set, it takes precedence over `RAG_API_KEY`. |
| `RAG_API_URL` | Dashboard/evaluation client API base URL. Defaults to `http://localhost:8000`. |
| `DATA_DIR`, `QDRANT_PATH` | SQLite and local Qdrant persistence locations. |
| `QDRANT_COLLECTION` | Qdrant collection name. |
| `EMBEDDING_MODEL`, `RERANKER_MODEL`, `GEMINI_MODEL` | Model identifiers. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_UPLOAD_MB` | Ingestion limits. |
| `RETRIEVAL_CANDIDATES`, `TOP_K`, `MAX_HISTORY_TURNS` | Retrieval and memory limits. |

Example multi-user configuration (use real, unique, high-entropy keys in your local environment):

```dotenv
RAG_USERS={"alice":"<alice-key>","bob":"<bob-key>"}
```

With only `RAG_API_KEY`, all callers share the `default` identity. Per-user document and conversation isolation requires `RAG_USERS` with a distinct key per principal. Never commit `.env` or paste credentials into logs, screenshots, issues, or documentation.

## Run locally

In one terminal, activate the environment and start the API:

```bash
uvicorn app:app --reload
```

In a second terminal, run the dashboard:

```bash
streamlit run dashboard.py
```

Open the dashboard at `http://localhost:8501` and enter an API key if prompted. The API listens at `http://localhost:8000`. The first use downloads the embedding and reranker models. FastAPI's `/docs` route is also behind bearer authentication.

## Docker deployment

Create `.env` from `.env.example` only if it does not already exist, then set `GEMINI_API_KEY` and either `RAG_API_KEY` or `RAG_USERS`:

```bash
docker compose up --build
```

The dashboard is at `http://localhost:8501`; the API is at `http://localhost:8000`. Qdrant and SQLite data persist in the `rag_data` named volume mounted at `/app/data`. The dashboard asks for a bearer key rather than receiving the API container's environment file.

## API endpoints

All routes except `GET /health` require `Authorization: Bearer <key>`.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Public status-only API/vector-store health check. |
| `GET` | `/health/gemini` | Validate Gemini credentials without returning the key. |
| `GET` | `/documents` | List documents visible to the authenticated owner. |
| `POST` | `/documents/upload` | Upload one or more PDFs as multipart `files` fields. |
| `DELETE` | `/documents/{doc_id}` | Delete a document owned by the caller. |
| `POST` | `/chat` | Ask a question; accepts `question`, optional `session_id`, and `top_k`. |
| `POST` | `/evaluation` | Evaluate 1–100 labeled RAGAS rows. |

Example document listing (bash):

```bash
curl http://localhost:8000/documents \
  -H "Authorization: Bearer $RAG_API_KEY"
```

PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/documents -Headers @{ Authorization = "Bearer $env:RAG_API_KEY" }
```

Example chat body:

```json
{"question":"What does the document say about retention?","session_id":"review-1","top_k":5}
```

The response includes the answer, only validated source/page citations, and retrieved contexts. The API enforces the configured total PDF upload limit (25 MB by default).

## Tests

```bash
python -m pytest -q
```

The suite covers PDF parsing and metadata, hybrid/reranked retrieval, owner isolation, citation checks, API validation/authentication, and SQLite persistence/concurrent writes. Tests mock external generation/evaluation services and do not produce live Gemini or RAGAS results.

## Retrieval benchmarking

Prepare a JSONL file from judged queries and relevant IDs from your indexed corpus. Use either `relevant_doc_ids` or `relevant_chunk_ids` per row; add `owner_id` when benchmarking a `RAG_USERS` principal:

```json
{"question":"<held-out question>","relevant_doc_ids":["<document-id>"]}
```

The local Qdrant store is single-process: stop the API before running the benchmark against the same store. For a local Python installation:

```bash
python scripts/benchmark_retrieval.py evaluation/queries.jsonl --k 5 --repeats 3 --output evaluation/retrieval-results.json
```

For the Docker named volume, create the `evaluation` directory and dataset first, then run from PowerShell:

```powershell
docker compose stop api
docker compose run --rm --no-deps -v "${PWD}/evaluation:/evaluation" api python scripts/benchmark_retrieval.py /evaluation/queries.jsonl --k 5 --repeats 3 --output /evaluation/retrieval-results.json
docker compose start api
```

The output reports Recall@k, MRR@k, nDCG@k, median warm latency, model IDs, retrieval candidate count, and dataset SHA-256. Results depend on the supplied relevance judgments; no benchmark scores are included in this repository.

## RAGAS evaluation

Prepare 1–100 held-out examples with the answer, retrieved context text, and reference answer. Save one JSON object per line:

```json
{"question":"<question>","answer":"<generated answer>","contexts":["<retrieved text>","<another retrieved passage>"],"ground_truth":"<reference answer>"}
```

Run the API with a valid Gemini key, then set `RAG_EVAL_API_KEY` in the client environment to a principal allowed by the API and run:

```powershell
$env:RAG_EVAL_API_KEY = "<your API key>"
python scripts/run_ragas.py evaluation/ragas.jsonl --output evaluation/ragas-results.json
```

The API evaluates faithfulness, answer relevancy, context precision, and context recall. The runner records the dataset hash and actual API results only after success; invalid credentials or provider failures do not produce a results file. Do not treat scores from a small or unrepresentative dataset as platform-wide performance claims.

## Security considerations

- Keep Gemini and bearer credentials in environment variables or a secret manager. `.gitignore` ignores new `.env` files, but ignore rules do not untrack files already committed; verify repository history and rotate any real credential that was published.
- Use `RAG_USERS` for per-user ownership. A shared `RAG_API_KEY` intentionally acts as one shared principal.
- All API routes except `/health` require bearer authentication. `/health` returns status only.
- The default Compose ports use HTTP. Put any remotely accessible deployment behind TLS and network access controls.
- Protect `DATA_DIR`: SQLite conversation history is stored unencrypted.
- Prompt-injection instructions treat document text and user input as untrusted; citation validation checks reference membership, not factual entailment.

## Known limitations

- Qdrant local mode supports a single API process. Use a Qdrant server for multiple workers or replicas.
- SQLite is persistent on the same volume/host, but is not a shared multi-host session database. PostgreSQL is the next step for multi-instance deployments.
- Image-only PDFs need OCR, which is not included.
- Cross-Encoder and embedding models download on first use; model loading and reranking can add latency, especially on CPU.
- Citation checks verify source/page existence, not semantic support for each statement. Live RAGAS evaluation requires a valid Gemini key and a representative labeled dataset.
- `RAG_USERS` is a static environment mapping, not an identity provider. There is no rate limiting or per-user quota management.

## Future improvements

- Add OIDC/JWT identity and managed key rotation for larger deployments.
- Move vector and conversation persistence to shared Qdrant and PostgreSQL services when scaling across processes/hosts.
- Add OCR for scanned documents and semantic entailment checks for citations.
- Add a curated, versioned evaluation dataset and run RAGAS/IR benchmarks in CI without publishing unverified scores.
