import streamlit as st
from backend.rag_engine import (
    load_multiple_pdfs,
    chunk_text,
    create_embeddings,
    retrieve
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG PDF Chatbot")
st.write("Ask questions from your PDFs using Retrieval-Augmented Generation")

@st.cache_resource
def prepare_data():
    pages = load_multiple_pdfs("pdfs")
    chunks = chunk_text(pages)
    embeddings = create_embeddings(chunks)
    return chunks, embeddings

with st.spinner("Loading PDFs and building index..."):
    chunks, embeddings = prepare_data()

query = st.text_input("Ask a question")

if query:
    results = retrieve(query, embeddings, chunks)

    st.subheader("🔍 Retrieved Context")
    for r in results:
        st.markdown(
            f"""
            **Source:** {r['source']}  
            **Page:** {r['page']}  

            {r['text'][:400]}
            ---
            """
        )
