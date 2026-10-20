import streamlit as st
from pathlib import Path

# --------------------------------------------------------------
# 1. Initialize chroma DB if it does not already exist
# --------------------------------------------------------------
CHROMA_DIR = Path("chroma_db")

if not CHROMA_DIR.exists():
    st.info("First time setup: builiding the insurance documents knowledge base...")

    from src.ingest import main as ingest_documents

    ingest_documents()

# --------------------------------------------------------------
# 2. Import RAG pipeline after chroma db is ready
# --------------------------------------------------------------
from src.rag import ask_rag

# --------------------------------------------------------------
# 3. Sreamlit page configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Claims Copilot",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------------------
# 4. Application title
# --------------------------------------------------------------
st.title("Insurance Claims Copilot")

st.write(
    "RAG powered application for querying insurance "
    "Policies, claims guidelines and SOPs."
)

# --------------------------------------------------------------
# 5. User question
# --------------------------------------------------------------
question = st.text_input(
    "Ask a question about insurance documents:"
)

# --------------------------------------------------------------
# 6. RAG response
# --------------------------------------------------------------
if question:
    st.subheader("Your Quesiion")
    st.write(question)

    with st.spinner(
        "Searching insurance documents and generating an answer..."
    ):
        answer, documents = ask_rag(question)

    st.write(answer)

    # --------------------------------------------------------------
    # 7. Source documents
    # --------------------------------------------------------------
    st.subheader("Sources")

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("Source", "Unknow source")

        with st.expander(f"Source {i}: {source}"):
            st.write(doc.page_content)