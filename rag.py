import google.generativeai as genai
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------- TEXT EXTRACTION ----------------------
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return text

# ---------------------- CHUNKING ----------------------
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------------- INDEXING ----------------------
def build_index(chunks):
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vectors = vectorizer.transform(chunks)
    return vectorizer, chunk_vectors

# ---------------------- RETRIEVAL ----------------------
def retrieve(query, vectorizer, chunk_vectors, chunks):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, chunk_vectors)[0]
    top_indices = scores.argsort()[-3:][::-1]
    return [chunks[i] for i in top_indices]

# ---------------------- GENERATION (GEMINI 2.5 FLASH) ----------------------
def answer_query(query, context_chunks):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the user question ONLY using this context:\n\n"
        f"{context}\n\n"
        f"Question: {query}"
    )
    
    response = model.generate_content(prompt)
    return response.text

# ---------------------- CHAT LOOP ----------------------
def chat():
    pdf_path = input("Enter PDF file path: ")
    full_text = extract_text(pdf_path)
    print("Extracting text...")

    chunks = split_text(full_text)
    print("Building TF-IDF index...")

    vectorizer, chunk_vectors = build_index(chunks)

    print("\nRAG Chatbot Ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        context = retrieve(query, vectorizer, chunk_vectors, chunks)
        answer = answer_query(query, context)

        print("\nAI:", answer, "\n")

# ---------------------- RUN ----------------------
chat()
