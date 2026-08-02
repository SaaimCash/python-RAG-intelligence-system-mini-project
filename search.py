# IMPORTS

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# RE-LOAD the embedding model
# same embedding model used to save the vectors!

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# CONNECTS TO THE EXISTING DATABASE
# passed 'persist_directory' pointing to our './chroma_db' folder.
# This means we do NOT need to re-process our raw files!
print("Connecting to local ChromaDB...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# DEFINED A USER QUESTION
# Question uses DIFFERENT words than the original text!
# Original text: "Payment processing supports Stripe and PayPal. Refunds take 5-7 business days..."

user_query = "What is the timeline for getting my money back?"

print(f"\nUser Query: '{user_query}'")
print("Searching vector database for top matching chunks...\n")

# RUN SIMILARITY SEARCH
# Then returns the TOP 2 most relevant chunks.

results = vectorstore.similarity_search(user_query, k=2)

# Prints the context

for idx, doc in enumerate(results):
    print(f"=== RETRIEVED CHUNK {idx + 1} ===")
    print(f"Content: {doc.page_content}\n")