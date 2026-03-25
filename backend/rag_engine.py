import os
from pypdf import PdfReader

# 🔹 Embeddings & similarity (ZERO COST)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Load a single PDF
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
# Load multiple PDFs from a folder
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
# Chunk text (ACCURACY IMPROVEMENT)
# =========================================================
def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []

    for p in pages:
        text = p["text"]

        for i in range(0, len(text), chunk_size - overlap):
            chunks.append({
                "text": text[i:i + chunk_size],
                "page": p["page"],
                "source": p["source"]
            })

    return chunks


# =========================================================
# Create embeddings (ZERO COST - LOCAL MODEL)
# =========================================================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    return embed_model.encode(texts, convert_to_numpy=True)


# =========================================================
# Retrieve relevant chunks (Semantic Search)
# =========================================================
def retrieve(query, embeddings, chunks, top_k=3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices]
