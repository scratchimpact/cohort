from loader import load_pdf
from rag_pipeline import build_documents, get_insights

PDF_PATH = "data/financial_report.pdf"

def main():
    pdf_text = load_pdf(PDF_PATH)
    documents = build_documents(pdf_text)
    insights = get_insights(documents)
    
    print("\n===== Insights Extracted =====\n")
    for idx, insight in enumerate(insights, 1):
        print(f"Insight {idx}:\n{insight}\n{'-'*50}")

if __name__ == "__main__":
    main()
