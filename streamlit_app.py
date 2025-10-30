# streamlit_app.py
# Advanced Streamlit app for RAG Document Analyzer by Piyush
# Modern UI + Interactive Features + Professional Design

import streamlit as st
from pathlib import Path
import tempfile
import base64
from qa_agent import generate_summary, generate_insights, generate_mcq, answer_question, build_retrieval_index

# -------------------------- Page Config --------------------------
st.set_page_config(
    page_title="RAG Document Analyzer — Piyush",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# -------------------------- Custom CSS --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    body {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
        color: #222222;
    }
    .main {
        background-color: #f7f7f7;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        transition: background 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #357ABD 0%, #2A5EAA 100%);
        transform: scale(1.05);
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        transition: background 0.3s ease, transform 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #357ABD 0%, #2A5EAA 100%);
        transform: scale(1.05);
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #111111;
        font-weight: 700;
    }
    .emoji {
        font-size: 1.3em;
    }
    .stSidebar {
        background-color: #f7f7f7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------- Helper Functions --------------------------
def save_uploaded_file(uploaded_file, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.read())
    return dst_path

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Customize and Explore!")

    valid_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    model = st.selectbox("🧠 Select Model", valid_models, index=0)

    temp = st.slider("🔥 Creativity (Temperature)", 0.0, 1.0, 0.3, 0.05)
    chunk_size = st.number_input("📄 Chunk Size (tokens)", 256, 4096, 1024, 64)

    st.markdown("---")
    reindex = st.button("🔁 Rebuild Retrieval Index")

    st.markdown("---")
    st.info("💡 Tip: Upload your file → Rebuild Index → Ask Smart Questions")

# -------------------------- Header --------------------------
col1, col2 = st.columns([8, 2])
with col1:
    st.title("🚀 RAG Document Analyzer — Piyush")
    st.caption("Empowering teams with AI-driven document intelligence 💬")
with col2:
    st.image("https://static.streamlit.io/images/brand/streamlit-mark-color.png", width=70)

st.progress(100)

# -------------------------- File Upload --------------------------
uploaded_file = st.file_uploader("📤 Upload Your PDF or CSV File", type=["pdf", "csv"], accept_multiple_files=False)

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

if uploaded_file:
    tmp = Path(tempfile.gettempdir()) / "uploaded_document"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / uploaded_file.name
    save_uploaded_file(uploaded_file, dst)
    st.session_state.uploaded_path = str(dst)
    st.success(f"✅ File Uploaded: {uploaded_file.name}")
    st.balloons()

# -------------------------- Tabs --------------------------
tabs = st.tabs(["📘 Document Overview", "🤖 Actions", "💬 Chat & QA", "📥 Exports"])

# -------------------------- Document Tab --------------------------
with tabs[0]:
    st.subheader("📄 Document Information")
    if st.session_state.uploaded_path:
        file_name = Path(st.session_state.uploaded_path).name
        size_kb = Path(st.session_state.uploaded_path).stat().st_size // 1024

        c1, c2, c3 = st.columns(3)
        c1.metric("📁 File Name", file_name)
        c2.metric("💾 Size", f"{size_kb} KB")
        c3.metric("⚙️ Model", model)

        with st.expander("🔍 Text Preview"):
            try:
                with open(st.session_state.uploaded_path, "rb") as f:
                    _b = f.read()
                if file_name.endswith(".pdf"):
                    st.info("PDF content extraction happens during processing.")
                else:
                    st.text(_b[:2000].decode(errors="replace"))
            except Exception:
                st.error("❌ Preview unavailable.")
    else:
        st.info("Please upload a file to analyze.")

# -------------------------- Actions Tab --------------------------
with tabs[1]:
    st.subheader("⚡ Intelligent Actions")

    if not st.session_state.uploaded_path:
        st.warning("⚠️ Upload a document first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🧾 Generate Summary"):
                with st.spinner("✨ Summarizing document..."):
                    summary = generate_summary(st.session_state.uploaded_path, model=model, temperature=temp)
                    st.session_state.last_summary = summary
                    st.success("🎉 Summary Generated Successfully!")
                    st.write(summary)
        with c2:
            if st.button("🔮 Generate Insights"):
                with st.spinner("Analyzing for insights..."):
                    insights = generate_insights(st.session_state.uploaded_path, model=model, chunk_size=chunk_size)
                    st.session_state.last_insights = insights
                    st.success("✅ Insights Ready!")
                    st.write(insights)
        with c3:
            if st.button("🎯 Generate MCQs"):
                with st.spinner("Generating interactive questions..."):
                    mcqs = generate_mcq(st.session_state.uploaded_path, num_questions=10, model=model)
                    st.session_state.last_mcqs = mcqs
                    st.success("✅ MCQs Generated!")
                    st.write(mcqs)

        st.markdown("---")
        if reindex:
            with st.spinner("Rebuilding retrieval index..."):
                result = build_retrieval_index(st.session_state.uploaded_path, chunk_size=chunk_size)
                st.success(result)

# -------------------------- Chat & QA Tab --------------------------
with tabs[2]:
    st.subheader("🧠 Smart Document Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.uploaded_path:
        st.info("Upload a file to start chatting.")
    else:
        query = st.text_input("💬 Ask a question about your document:")
        if st.button("🚀 Ask") and query:
            with st.spinner("Thinking..."):
                ans = answer_question(st.session_state.uploaded_path, query, model=model)
                st.session_state.chat_history.append({"q": query, "a": ans})
                st.toast("AI has answered your question!", icon="🤖")
        for turn in reversed(st.session_state.chat_history[-5:]):
            with st.chat_message("user"):
                st.markdown(f"**🧍‍♂️ You:** {turn['q']}")
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 AI:** {turn['a']}")

# -------------------------- Downloads Tab --------------------------
with tabs[3]:
    st.subheader("📦 Export Results")

    if st.session_state.get("last_summary"):
        st.download_button("⬇️ Download Summary", st.session_state.last_summary, file_name="summary.txt")
    if st.session_state.get("last_insights"):
        st.download_button("⬇️ Download Insights", st.session_state.last_insights, file_name="insights.txt")
    if st.session_state.get("last_mcqs"):
        st.download_button("⬇️ Download MCQs", st.session_state.last_mcqs, file_name="mcqs.txt")

    if not (st.session_state.get("last_summary") or st.session_state.get("last_mcqs")):
        st.info("No downloadable content yet. Run an action first!")

st.markdown("---")
st.caption("💡 Created by **Piyush** | ✨ Powered by OpenAI | 🧩 Enhanced with Streamlit Interactive UI")