from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="insurance_documents"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

question = input("\nWhat is your question? ")

documents = retriever.invoke(question)

print("\n===================== RETRIEVED DOCUMENTS =====================")

for i, doc in enumerate(documents, start=1):

    print(f"=== RESULT {i} ===")

    print("Source:", doc.metadata.get("source", "Unknown"))

    print(doc.page_content)
    # print(f"Content: {doc.page_content[:500]}...")  # Print first 500 characters of content

    print()
    