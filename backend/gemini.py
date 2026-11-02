import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(
        prompt
    )

    return response.text