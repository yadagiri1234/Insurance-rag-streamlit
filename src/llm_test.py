import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
    max_retries=3,
)

prompt = """
Explain Reetrieval Augmented Generation (RAG) in one short paragraph.
Focus on the relationship between retrieval, context, and genetraion.
"""

for attempt in range(1, 4):
    try:
        print(f"\nAttempt {attempt}/3...")

        response = llm.invoke(prompt)

        print("\n===================== GEMINI RESPONSE =====================")
        print(response.content)

        break  # Exit the loop if the request was successful
    except Exception as e:
        print(f"\nRequest failed: {e}")
        if attempt < 3:
            print("Waiting before retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            print("Max retries reached. Exiting.")
            #print("\nGemini is currently unavailable. Please try again later.")