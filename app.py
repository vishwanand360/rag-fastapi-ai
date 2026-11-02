"""FastAPI interface for the PDF RAG platform."""
from __future__ import annotations

import logging
import json
import os
import secrets
import uuid
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load the project's local configuration before settings and API auth are read.
load_dotenv()

from backend.config import settings
from backend.ingestion import parse_pdfs
from backend.rag_service import get_service
from backend.session_store import ConversationStore
from backend.gemini_errors import gemini_error_message, invalid_gemini_credentials
from backend.citations import citations_for_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Advanced PDF RAG", version="1.0.0")
conversation_store = ConversationStore(settings.data_dir / "conversations.sqlite3")


def _configured_users() -> dict[str, str]:
    raw = os.getenv("RAG_USERS", "").strip()
    if raw:
        try:
            users = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("RAG_USERS must be a JSON object mapping user IDs to API keys") from exc
        if (not isinstance(users, dict) or not users or
                any(not isinstance(user, str) or not user.strip() or len(user) > 100 or
                    not isinstance(key, str) or not key for user, key in users.items())):
            raise ValueError("RAG_USERS must be a non-empty JSON object mapping user IDs to API keys")
        if len(set(users.values())) != len(users):
            raise ValueError("Each RAG_USERS principal must have a distinct API key")
        return users
    key = os.getenv("RAG_API_KEY", "")
    return {"default": key} if key else {}


@app.middleware("http")
async def require_api_key(request, call_next):
    # Keep one unauthenticated health probe for Docker; all document and model
    # operations require a shared bearer token configured outside source code.
    if request.url.path == "/health":
        return await call_next(request)
    try:
        users = _configured_users()
    except ValueError as exc:
        logger.error("Invalid API identity configuration: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    if not users:
        return JSONResponse(status_code=503, content={"detail": "RAG_API_KEY is not configured"})
    supplied = request.headers.get("authorization", "")
    token = supplied[7:] if supplied.startswith("Bearer ") else ""
    owner_id = next((user for user, key in users.items() if secrets.compare_digest(token, key)), None)
    if owner_id is None:
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={"detail": "Invalid or missing API key"},
        )
    request.state.owner_id = owner_id
    return await call_next(request)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


class EvaluationRow(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12000)
    contexts: list[str] = Field(min_length=1, max_length=20)
    ground_truth: str = Field(min_length=1, max_length=12000)


class EvaluationRequest(BaseModel):
    rows: list[EvaluationRow] = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    try:
        get_service().list_documents()
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Vector store unavailable") from exc


@app.get("/health/gemini")
def gemini_health():
    try:
        get_service().validate_gemini_credentials()
        return {"status": "ok", "provider": "Gemini"}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.get("/documents")
def documents(request: Request):
    try:
        return {"documents": get_service().list_documents(request.state.owner_id)}
    except Exception as exc:
        logger.exception("Could not list documents")
        raise HTTPException(status_code=503, detail="Document store unavailable") from exc


@app.post("/documents/upload")
async def upload_documents(files: Annotated[list[UploadFile], File(...)], request: Request) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one PDF")
    total_bytes = 0
    file_data = []
    try:
        for upload in files:
            filename = upload.filename or "document.pdf"
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=415, detail=f"{filename}: only PDF files are supported")
            data = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
            total_bytes += len(data)
            if total_bytes > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"Total upload exceeds {settings.max_upload_mb} MB")
            file_data.append((filename, data))
        chunks = parse_pdfs(file_data, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        result = get_service().add_chunks(chunks, owner_id=request.state.owner_id)
        return {"message": "Documents indexed", **result}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=503, detail="Document ingestion failed") from exc
    finally:
        for upload in files:
            await upload.close()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, request: Request):
    if not doc_id or len(doc_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    try:
        if not get_service().delete_document(doc_id, request.state.owner_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": doc_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=503, detail="Document deletion failed") from exc


@app.post("/chat")
def chat(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank")
    session_id = payload.session_id or str(uuid.uuid4())
    owner_id = request.state.owner_id
    history = conversation_store.get_history(owner_id, session_id)
    try:
        service = get_service()
        contexts = service.retrieve(question, payload.top_k, owner_id=owner_id)
        if not contexts:
            return {"session_id": session_id, "answer": "No indexed document content is available to answer this question.", "citations": [], "contexts": []}
        answer = service.answer(question, contexts, history)
        try:
            citations = citations_for_answer(answer, contexts)
        except ValueError as exc:
            logger.warning("Answer citation validation failed; withholding response")
            raise HTTPException(status_code=503, detail="Answer citations could not be verified; please retry.") from exc
        conversation_store.append_turn(owner_id, session_id, question, answer, settings.max_history_turns)
        return {"session_id": session_id, "answer": answer, "citations": citations, "contexts": contexts}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=503, detail="Retrieval service is temporarily unavailable") from exc


@app.post("/evaluation")
def evaluate(payload: EvaluationRequest):
    """Evaluate supplied RAG rows: question, answer, contexts, ground_truth."""
    rows = [row.model_dump() for row in payload.rows]
    try:
        get_service().validate_gemini_credentials()
        from datasets import Dataset
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas import evaluate as ragas_evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        dataset = Dataset.from_list(rows)
        result = ragas_evaluate(
            dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
            llm=LangchainLLMWrapper(ChatGoogleGenerativeAI(model=settings.gemini_model)),
            embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
        )
        return {"scores": result.to_pandas().mean(numeric_only=True).to_dict(), "samples": len(rows)}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception as exc:
        logger.error("RAGAS evaluation failed (%s)", type(exc).__name__)
        detail = gemini_error_message(exc) if invalid_gemini_credentials(exc) else (
            "RAGAS evaluation failed. Check evaluator logs, dependency configuration, and provider limits."
        )
        raise HTTPException(status_code=503, detail=detail) from None
