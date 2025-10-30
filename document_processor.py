from langchain.schema import Document

def create_documents(pdf_text):
    return [Document(page_content=pdf_text, metadata={"type": "pdf"})]
