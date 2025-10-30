from rag_pipeline import get_chat_completion, build_rag, load_pdf

def generate_mcq(pdf_path):
    documents = load_pdf(pdf_path)
    qa_chain = build_rag(documents)

    messages = [
        {"role": "system", "content": "You create multiple-choice questions."},
        {"role": "user", "content": "Generate 5 MCQs from this document."}
    ]

    response = get_chat_completion(messages, qa_chain)
    return response
