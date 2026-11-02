"""Previous FAISS dashboard retained for compatibility with the local prototype."""
import streamlit as st

from backend.rag_engine import build_rag_pipeline, retrieve_faiss

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("📄 RAG-Based PDF Chatbot")
st.write("Ask questions from your PDF documents using semantic search powered by FAISS.")


@st.cache_resource(show_spinner=True)
def load_rag_pipeline_cached():
    return build_rag_pipeline("pdfs")


with st.spinner("Loading PDFs and building FAISS index..."):
    chunks, embeddings, index = load_rag_pipeline_cached()
st.success("✅ PDFs loaded and FAISS index ready!")

query = st.text_input("💬 Ask a question:", placeholder="What is Retrieval-Augmented Generation?")
if query:
    results = retrieve_faiss(query, index, chunks)
    st.subheader("📌 Retrieved Context")
    for i, result in enumerate(results, start=1):
        st.markdown(f"**Source {i}:** `{result['source']}` | Page {result['page']}")
        st.write(result["text"])
        st.divider()
