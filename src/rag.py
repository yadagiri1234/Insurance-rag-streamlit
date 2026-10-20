from contextvars import Context
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

CHROMA_DIR = "chroma_db"
collection_name = "insurance_documents"

# --------------------------------------------------
# 1. Embedding Model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 2. Connect to Vector Store / database
# --------------------------------------------------
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# --------------------------------------------------
# 3. Create a retriever from the vector database
# --------------------------------------------------
retriever = vectorstore.as_retriever(
    #search_type="similarity",
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# 4. Gemini LLM
# --------------------------------------------------
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")
else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
        max_retries=3,
    )

# --------------------------------------------------
# 5. Grounded RAG function
# --------------------------------------------------
def ask_rag(question):
    documents = retriever.invoke(question)

    # Retrieve relevant documents from the vector store
    #retrieved_docs = retriever.get_relevant_documents(query)

    # Combine the retrieved documents into a single context string
    context = "\n\n".join(
        [
            f"SOURCE: {doc.metadata.get('source')}\n"
            f"Content: {doc.page_content}\n"
            for doc in documents
        ])

    # Create a prompt that includes the context and the user's query
    prompt = f"""
You are an insurance claims assistant. 

Answer the user's question using ONLY the information provided in the context below.

Do not use outside Knowledge or make any assumptions."

if the answer cannot be determined from the context,
say

"I cannot determine the answer from the provided documents."

Do not invent policy conditions, claim amounts, coverage rules, or any other facts.

Always provide the source documents used.

CONTEXT
-------
{context}

QUESTION
--------
{question}

ANSWER
------
"""
    
    
    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        answer = response.content
    else:
        answer_parts = []
        for part in response.content:
            if isinstance(part, dict) and  part.get("text"):
                answer_parts.append(part["text"])
        
        answer = "\n".join(answer_parts) #if answer_parts else "No answer generated."

    return answer, documents

# --------------------------------------------------
# 6. Command-line test
# --------------------------------------------------
if __name__ == "__main__":
    question = input("Enter your insurance related question: ")
    answer, docs = ask_rag(question)
    print("\n================ GROUNDED ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")

    for i, doc in enumerate(docs, start=1):
        print(f"{i}. "
              f"{doc.metadata.get('source', 'Unknown')}"
        )
    