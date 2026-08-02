# Import the text splitter module from LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter


# OUR RAW DATASET (The Document Library)
documents = [
    "Feature A: User authentication allows login via OAuth2, Google, and GitHub. Passwords must be at least 12 characters.",
    "Feature B: Dark mode automatically toggles based on system preferences or manual overrides in user settings.",
    "Feature C: Data exports can be generated in CSV or JSON formats up to 50,000 rows per batch export.",
    "Feature D: Payment processing supports Stripe and PayPal. Refunds take 5-7 business days to process.",
]

print(f"Total raw documents loaded: {len(documents)}")

# CHUNKER CONFIGURATION
# Each chunk will be around 100 characters max.
# The last 20 characters of chunk N repeat in chunk N+1. (can be changed based on file type)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)


# Converts raw string lists into structured Document objects
chunks = text_splitter.create_documents(documents)


# Output
print(f"Total chunks created: {len(chunks)}\n")

for idx, chunk in enumerate(chunks):
    print(f"--- CHUNK {idx + 1} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Character Length: {len(chunk.page_content)}\n")