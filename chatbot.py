import streamlit as st
from backend.rag_engine import (
    load_multiple_pdfs,
    chunk_text,
    create_embeddings,
    retrieve
)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    layout="centered"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("📄 RAG-Based PDF Chatbot")
st.write(
    "Ask questions from your PDF documents using "
    "**Retrieval-Augmented Generation (RAG)**."
)

# -------------------------------
# CACHE HEAVY OPERATIONS
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_rag_pipeline():
    pages = load_multiple_pdfs("pdfs")
    chunks = chunk_text(pages)
    embeddings = create_embeddings(chunks)
    return chunks, embeddings

# -------------------------------
# LOAD DATA (ONLY ONCE)
# -------------------------------
with st.spinner("Loading documents and building embeddings..."):
    chunks, embeddings = load_rag_pipeline()

st.success("✅ Documents loaded successfully!")

# -------------------------------
# USER INPUT
# -------------------------------
query = st.text_input(
    "💬 Ask a question from the PDFs:",
    placeholder="e.g. What is Retrieval-Augmented Generation?"
)

# -------------------------------
# RETRIEVE & DISPLAY ANSWER
# -------------------------------
if query:
    results = retrieve(query, embeddings, chunks)

    st.subheader("📌 Retrieved Context")

    if results:
        for i, res in enumerate(results, 1):
            st.markdown(
                f"**Source {i}:** `{res['source']}` | Page {res['page']}"
            )
            st.write(res["text"])
            st.divider()
    else:
        st.warning("No relevant information found in the PDFs.")
