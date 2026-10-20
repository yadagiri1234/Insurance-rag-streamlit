import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # Load environment variables from .env file

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

def load_documents():
    documents = []

    for path in DATA_DIR.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text, 
                metadata={"source": str(path.relative_to(BASE_DIR))}
                )
            )

    return documents

def build_vector_store():
    raw_docs = load_documents()
    chunks = text_splitter.split_documents(raw_docs)
    
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embeddings=embeddings, 
        #persist_directory=CHROMA_DIR,
        collection_name="insurance_documents"
    )

    vector_store.add_documents(chunks)  # Add the chunks to the vector store

    return vector_store

def get_vector_store():
    # if not Path(CHROMA_DIR).exists():
    #     print("Chroma database not found. Building vector store...")
    #     return build_vectorstore()
    # else:
    #     print("Loading existing Chroma database...")
    #     return Chroma(
    #         persist_directory=CHROMA_DIR,
    #         embedding_function=embeddings,
    #         collection_name="insurance_documents"
    #     )

    return Chroma(
        persist_directory="CHROMA_DIR",
        embedding_function=embeddings,
        collection_name="insurance_documents"
    )

def answer_question(question: str):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}\n"
        for doc in retrieved_docs
    )

    prompt = f"""
    You are an insurance claims assistant.

    Answer the user's question using ONLY the supplied documents.

    if the document does not contain enough information,
    say: "I cannot determine that from the provided documents."

    Do not invent new policy terms, prices, coverage limits, claim outcomes, or requirements.

    USER QUESTION:
    {question}

    RETRIEVED DOCUMENTS:
    {context}
    """
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
        max_retries=3,
    )

    response = llm.invoke(prompt)

    return response.content, retrieved_docs
