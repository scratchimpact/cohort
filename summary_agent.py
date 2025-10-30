from rag_pipeline import get_chat_completion, build_rag, load_pdf

def summarize_document(pdf_path):
    documents = load_pdf(pdf_path)
    qa_chain = build_rag(documents)

    messages = [
        {"role": "system", "content": "You summarize financial documents."},
        {"role": "user", "content": "Provide a concise summary of this document."}
    ]

    response = get_chat_completion(messages, qa_chain)
    return response
