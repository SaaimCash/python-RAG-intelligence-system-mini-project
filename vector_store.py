import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def process_and_store_document(file_path: str):
    """
    Loads a PDF or TXT file, splits it into overlapping chunks,
    generates embeddings, and persists them into ChromaDB.

    Re-running this on the same file_path replaces that file's old
    chunks instead of duplicating them, so ingestion is idempotent.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\n1. Loading document from: {file_path}")

    # Select loader based on file extension
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw page(s)/document(s).")

    # 2. CHUNKING CONFIGURATION
    # For real PDFs, 500 characters per chunk with 50 overlap works well
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"2. Created {len(chunks)} text chunks.")

    print("3. Loading embedding model & connecting to ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    # DE-DUP GUARD
    # PyPDFLoader/TextLoader stamp each chunk's metadata with "source" =
    # the file_path passed in. Before adding new chunks, remove any
    # chunks already in the store from this same source so re-running
    # ingestion on the same file doesn't pile up duplicates over time.
    existing = vectorstore.get(where={"source": file_path})
    if existing["ids"]:
        print(f"   Found {len(existing['ids'])} existing chunk(s) from this file — replacing them.")
        vectorstore.delete(ids=existing["ids"])

    vectorstore.add_documents(chunks)

    print("Success! Document processed and saved into ChromaDB.\n")
    return vectorstore


if __name__ == "__main__":
    # Sample test
    sample_file = "sample.pdf"

    if os.path.exists(sample_file):
        process_and_store_document(sample_file)
    else:
        print(f"Place a sample PDF named '{sample_file}' in this directory to test manually!")