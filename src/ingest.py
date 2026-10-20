from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

def load_documents():
    documents = []

    for file_path in DATA_DIR.rglob("*.txt"):
        loader = TextLoader(
            str(file_path), 
            encoding="utf-8"
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = str(file_path)
            documents.append(doc)
        
        documents.extend(docs)
    
    return documents

def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,  
        persist_directory=CHROMA_DIR,
        embedding=embeddings,
        collection_name="insurance_documents"
    )

    print("Chroma database created successfully.")

    #print(f"stored chunks: {vectorstore._collection.count()}")
    print(f"stored chunks: {len(chunks)}")

    #vectorstore.persist()
    #vectorstore = None

    if __name__ == "__main__":
        main()