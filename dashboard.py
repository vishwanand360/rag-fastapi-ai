"""Streamlit dashboard for document administration, chat, and RAGAS."""
import json
import os
from html import escape

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("RAG_API_KEY", "")
st.set_page_config(page_title="Atlas | Document Intelligence", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1280px}.hero{padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#101b31,#21395b);color:#fff;margin-bottom:1.2rem}
.hero h1{margin:0}.hero p{color:#cbd5e1;margin:.3rem 0 0}.citation{border-left:3px solid #5b8def;padding:.55rem .9rem;margin:.4rem 0;background:#f4f7fb;border-radius:4px}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Atlas Document Intelligence</h1><p>Search, cite, and evaluate answers across your PDF library.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.text_input(
        "API access key", value=API_KEY, type="password", key="api_access_key",
        help="Use your personal key when RAG_USERS is configured on the API.",
    )

def api(method, path, **kwargs):
    try:
        headers = dict(kwargs.pop("headers", {}))
        api_key = st.session_state.get("api_access_key", API_KEY)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.request(method, f"{API}{path}", headers=headers, timeout=120, **kwargs)
        if response.ok:
            return response.json()
        st.error(response.json().get("detail", response.text))
    except requests.RequestException as exc:
        st.error(f"API unavailable: {exc}")
    return None

with st.sidebar:
    st.subheader("Platform")
    health = api("GET", "/health")
    st.success("Connected" if health else "API unavailable")
    st.caption(f"API: {API}")
    page = st.radio("Workspace", ["Chat", "Documents", "Evaluation"], label_visibility="collapsed")

if page == "Documents":
    st.header("Document library")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index selected PDFs", type="primary", disabled=not uploads):
        with st.spinner("Extracting, chunking, and indexing documents…"):
            result = api("POST", "/documents/upload", files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in uploads])
        if result:
            st.success(f"Indexed {len(result['documents'])} document(s), {result['chunks']} chunks.")
            st.rerun()
    result = api("GET", "/documents")
    docs = result.get("documents", []) if result else []
    if docs:
        for doc in docs:
            left, mid, right = st.columns([5, 2, 1])
            left.write(f"**{doc['filename']}**")
            mid.caption(f"{doc['chunks']} chunks · {doc['id'][:8]}")
            if right.button("Delete", key=f"del-{doc['id']}"):
                if api("DELETE", f"/documents/{doc['id']}"):
                    st.success("Document deleted.")
                    st.rerun()
    else:
        st.info("No documents indexed yet.")
elif page == "Chat":
    st.header("Ask your documents")
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.sidebar:
        if st.button("New conversation"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
        top_k = st.slider("Sources per answer", 1, 10, 5)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for cite in msg.get("citations", []):
                st.markdown(f"<div class='citation'>📄 <b>{escape(cite['source'])}</b> · page {cite['page']}</div>", unsafe_allow_html=True)
    if question := st.chat_input("Ask a question about your PDFs"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and composing a cited answer…"):
                result = api("POST", "/chat", json={"question": question, "session_id": st.session_state.session_id, "top_k": top_k})
            if result:
                st.session_state.session_id = result["session_id"]
                st.markdown(result["answer"])
                for cite in result["citations"]:
                    st.markdown(f"<div class='citation'>📄 <b>{escape(cite['source'])}</b> · page {cite['page']}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "citations": result["citations"]})
elif page == "Evaluation":
    st.header("RAGAS evaluation")
    st.caption("Score a labeled dataset with faithfulness, answer relevancy, context precision, and context recall.")
    sample = [{"question":"What is RAG?","answer":"...","contexts":["..."],"ground_truth":"..."}]
    raw = st.text_area("Dataset JSON", value=json.dumps(sample, indent=2), height=240)
    if st.button("Run evaluation", type="primary"):
        try:
            rows = json.loads(raw)
            with st.spinner("Running RAGAS with Gemini…"):
                result = api("POST", "/evaluation", json={"rows": rows})
            if result:
                cols = st.columns(4)
                labels = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                for col, label in zip(cols, labels):
                    col.metric(label.replace("_", " ").title(), f"{result['scores'].get(label, float('nan')):.3f}")
                st.caption(f"Evaluated {result['samples']} labeled examples. Scores are dataset results, not platform benchmarks.")
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid dataset JSON: {exc}")
