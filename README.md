# 🧠 AI RAG Lab

A mini project demo to learn how to build a local **Retrieval-Augmented Generation (RAG)** pipeline and REST API with LangChain, ChromaDB, LiteLLM, and FastAPI.

Ask questions about your documents — the AI answers using **only** what's in your knowledge base, and refuses to hallucinate.

---

## 📁 Project Structure

```
ai_rag_lab/
├── main.py          # FastAPI server with endpoints for upload & Q&A
├── ingest.py        # Inspect how documents are chunked
├── vector_store.py  # Embed documents & save to ChromaDB
├── search.py        # Test semantic search against ChromaDB
├── rag_app.py       # Full RAG pipeline (retrieval + LLM answer)
├── requirements.txt # Python dependencies
└── .env             # Your API keys (NOT uploaded to GitHub)
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/python-rag-intelligence-system-mini-project.git
cd python-rag-intelligence-system-mini-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file
Create a file called `.env` in the root folder:
```env
# Add whichever key matches the model you want to use
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 4. Build the vector database (Optional for manual testing)
```bash
python vector_store.py
```
This reads your documents, chunks them, embeds them, and saves the ChromaDB database locally into `./chroma_db/`.

---

## 🚀 Running the Project

### Option A: Run the FastAPI REST Server
Start the backend server:
```bash
uvicorn main:app --reload --port 8000
```
- **Interactive Swagger UI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

#### Available API Endpoints:
- `GET /api/v1/health`: Checks service status.
- `POST /api/v1/upload`: Upload `.pdf` or `.txt` files to embed and store in ChromaDB.
- `POST /api/v1/ask`: Ask questions against the vectorized knowledge base.

---

### Option B: Run the Standalone RAG Script
```bash
python rag_app.py
```

---

## 🔄 Switching AI Models

Edit the `MODEL` variable at the top of `rag_app.py`:

```python
MODEL = "groq/llama-3.1-8b-instant"   # Groq free tier
MODEL = "gemini/gemini-2.0-flash"     # Google Gemini
MODEL = "openai/gpt-4o-mini"          # OpenAI
MODEL = "ollama/llama3.2"              # Local (no API key needed)
```

LiteLLM automatically picks the right API key from your `.env`.

---

## 🧩 How It Works

1. **Chunk** — Documents are split into small overlapping pieces
2. **Embed** — Each chunk is converted to a vector (list of numbers) by a local HuggingFace model
3. **Store** — Vectors are saved in ChromaDB on your disk
4. **Retrieve** — User question is converted to a vector; closest chunks are fetched
5. **Generate** — Chunks + question are sent to an LLM with strict instructions to only use the provided context

### Full Pipeline at a Glance
```
Your Document (PDF/TXT)
        │
        ▼
  [Load]  PyPDFLoader / TextLoader
        │
        ▼
  [Chunk]  RecursiveCharacterTextSplitter (500 chars, 50 overlap)
        │
        ▼
  [Embed]  HuggingFace all-MiniLM-L6-v2  ──► 384-dim vectors
        │
        ▼
  [Store]  ChromaDB (persisted to ./chroma_db/)
        │
   ─────┴───── (indexing done — run once) ─────
        │
   User asks a question
        │
        ▼
  [Embed Question]  same HuggingFace model
        │
        ▼
  [Retrieve]  cosine similarity search → top-k chunks
        │
        ▼
  [Generate]  chunks + question → strict prompt → LiteLLM → LLM → Answer
```

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com) | High-performance REST API backend |
| [Uvicorn](https://www.uvicorn.org) | ASGI web server |
| [LangChain](https://langchain.com) | Orchestration framework |
| [ChromaDB](https://trychroma.com) | Local vector database |
| [LiteLLM](https://litellm.ai) | Universal LLM API wrapper |
| [HuggingFace](https://huggingface.co) | Local embedding model (`all-MiniLM-L6-v2`) |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loads `.env` secrets |
