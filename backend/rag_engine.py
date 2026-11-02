import os
import faiss
import numpy as np
from pypdf import PdfReader

# Embeddings & similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Load Single PDF
# =========================================================
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            pages.append({
                "text": text,
                "page": i + 1
            })

    return pages


# =========================================================
# Load Multiple PDFs
# =========================================================
def load_multiple_pdfs(folder_path):
    all_pages = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(folder_path, filename)

            pages = load_pdf(pdf_path)

            for p in pages:
                p["source"] = filename
                all_pages.append(p)

    return all_pages


# =========================================================
# Chunk Text
# =========================================================
def chunk_text(pages, chunk_size=500, overlap=100):

    chunks = []

    for p in pages:

        text = p["text"]

        for i in range(
            0,
            len(text),
            chunk_size - overlap
        ):

            chunks.append({
                "text": text[i:i + chunk_size],
                "page": p["page"],
                "source": p["source"]
            })

    return chunks


# =========================================================
# Load Embedding Model
# =========================================================
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# Create Embeddings
# =========================================================
def create_embeddings(chunks):

    texts = [
        c["text"]
        for c in chunks
    ]

    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


# =========================================================
# Old Retrieval Method
# Cosine Similarity
# =========================================================
def retrieve(
    query,
    embeddings,
    chunks,
    top_k=3
):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    return [
        chunks[i]
        for i in top_indices
    ]


# =========================================================
# Create FAISS Index
# =========================================================
def create_faiss_index(embeddings):

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# FAISS Retrieval
# =========================================================
def retrieve_faiss(
    query,
    index,
    chunks,
    top_k=3
):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(
            chunks[idx]
        )

    return results


# =========================================================
# Build Complete Pipeline
# =========================================================
def build_rag_pipeline(pdf_folder="pdfs"):

    pages = load_multiple_pdfs(
        pdf_folder
    )

    chunks = chunk_text(
        pages
    )

    embeddings = create_embeddings(
        chunks
    )

    faiss_index = create_faiss_index(
        embeddings
    )

    return (
        chunks,
        embeddings,
        faiss_index
    )