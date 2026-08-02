
# IMPORTS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# OUR DOCUMENTS (Knowledge Base)

documents = [
    "Feature A: User authentication allows login via OAuth2, Google, and GitHub. Passwords must be at least 12 characters.",
    "Feature B: Dark mode automatically toggles based on system preferences or manual overrides in user settings.",
    "Feature C: Data exports can be generated in CSV or JSON formats up to 50,000 rows per batch export.",
    "Feature D: Payment processing supports Stripe and PayPal. Refunds take 5-7 business days to process.",
]


# CHUNKS THE TEXT

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.create_documents(documents)
print(f"Created {len(chunks)} text chunks.")


# LOAD THE EMBEDDING MODEL
print("Loading open-source embedding model onto your CPU...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STORE CHUNKS IN CHROMADB
# This command takes every text chunk, converts it into numbers using
# the model above, and saves the data to a folder called 'chroma_db'.

print("Converting text chunks into vectors and saving to database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db" 
)

print("\nSuccess! Your vector database is saved in 'chroma_db' folder.")