# qa_agent.py
"""
Simple QA agent module for Streamlit RAG Document Analyzer.
Provides summarization, insight generation, MCQ creation, and QA answering
using OpenAI GPT models.
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------------------------
# 🔹 Load environment variables and initialize client
# -------------------------------------------------------------------
load_dotenv()  # Loads .env file from project root
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file. Please set it properly.")

client = OpenAI(api_key=api_key)

# -------------------------------------------------------------------
# 🔹 Utility: Extract text from PDF
# -------------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"[Error extracting text: {e}]"

# -------------------------------------------------------------------
# 1️⃣ Generate Summary
# -------------------------------------------------------------------
def generate_summary(file_path, model="gpt-4o-mini", temperature=0.3):
    text = extract_text_from_pdf(file_path)
    if not text:
        return "No text extracted from the document."

    prompt = f"Summarize this financial or technical document in bullet points:\n\n{text[:6000]}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert financial data summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Summary generation error: {e}]"

# -------------------------------------------------------------------
# 2️⃣ Generate Insights
# -------------------------------------------------------------------
def generate_insights(file_path, model="gpt-4o-mini", chunk_size=1024):
    text = extract_text_from_pdf(file_path)
    if not text:
        return "No text extracted."

    prompt = f"From the following financial document, derive deep insights, patterns, and anomalies:\n\n{text[:8000]}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst who identifies trends, risks, and opportunities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Insight generation error: {e}]"

# -------------------------------------------------------------------
# 3️⃣ Generate MCQs
# -------------------------------------------------------------------
def generate_mcq(file_path, num_questions=5, model="gpt-4o-mini"):
    text = extract_text_from_pdf(file_path)
    if not text:
        return "No text extracted."

    prompt = (
        f"Generate {num_questions} multiple choice questions with 4 options each, "
        "and provide the correct answer at the end. Use this document:\n\n"
        f"{text[:8000]}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a teacher creating concept-checking MCQs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[MCQ generation error: {e}]"

# -------------------------------------------------------------------
# 4️⃣ Answer Question
# -------------------------------------------------------------------
def answer_question(file_path, question, model="gpt-4o-mini"):
    text = extract_text_from_pdf(file_path)
    if not text:
        return "No text available to answer from."

    prompt = f"Use the following document to answer the question:\n\nDocument:\n{text[:6000]}\n\nQuestion: {question}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert assistant that answers accurately from context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Answer generation error: {e}]"

# -------------------------------------------------------------------
# 5️⃣ Dummy Build Retrieval Index
# -------------------------------------------------------------------
def build_retrieval_index(file_path, chunk_size=1024):
    """Placeholder RAG index builder (for UI only)."""
    text = extract_text_from_pdf(file_path)
    return f"✅ Indexed document '{Path(file_path).name}' with {len(text)//chunk_size + 1} chunks."
