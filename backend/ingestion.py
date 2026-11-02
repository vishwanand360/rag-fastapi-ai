"""PDF parsing and page-aware chunking."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf(data: bytes, filename: str, *, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    if not data.startswith(b"%PDF"):
        raise ValueError("File is not a valid PDF")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF could not be opened") from exc
    doc_id = hashlib.sha256(data).hexdigest()[:24]
    source = Path(filename.replace("\\", "/")).name
    source = re.sub(r"[^\w.() -]", "_", source).strip()[:200] or "document.pdf"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_index, page in enumerate(document, start=1):
        text = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if not text:
            continue
        for chunk_index, text_chunk in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": f"{doc_id}:{page_index}:{chunk_index}", "doc_id": doc_id,
                "source": source, "page": page_index,
                "text": text_chunk, "metadata": {"page_count": len(document)},
            })
    document.close()
    if not chunks:
        raise ValueError("PDF contains no extractable text")
    return chunks


def parse_pdfs(files: Iterable[tuple[str, bytes]], **kwargs) -> list[dict]:
    result = []
    for filename, data in files:
        result.extend(parse_pdf(data, filename, **kwargs))
    return result
