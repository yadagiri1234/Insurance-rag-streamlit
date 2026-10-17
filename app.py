import streamlit as st

st.set_page_config(
    page_title="Insurance Claims Copilot",
    page_icon="📄",
    layout="wide"
)

st.title("Insurance Claims Copilot")

st.write(
    "RAG-powered application for querying insurance "
    "policies, claims guidelines and SOPs."
)

question = st.text_input(
    "Ask a question about the insurance documents:"
)

if question:
    st.subheader("Your Question")
    st.write(question)

    st.subheader("Response")
    st.info("RAG pipeline will be connected here.")
