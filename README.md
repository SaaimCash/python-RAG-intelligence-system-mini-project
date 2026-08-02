# 🧠 AI RAG Lab

A local **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, ChromaDB, and LiteLLM.

Ask questions about your documents — the AI answers using **only** what's in your knowledge base, and refuses to hallucinate.

---

## 📁 Project Structure

```
ai_rag_lab/
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
> Get a free Groq key at: https://console.groq.com

### 4. Build the vector database
```bash
python vector_store.py
```
This reads your documents, chunks them, embeds them, and saves the ChromaDB database locally.

---

## 🚀 Running the App

```bash
python rag_app.py
```

---

## 🔄 Switching AI Models

Edit the `MODEL` variable at the top of `rag_app.py`:

```python
MODEL = "ollama/llama3.2"              # Local (no API key needed)
MODEL = "groq/llama-3.1-8b-instant"   # Groq free tier
MODEL = "gemini/gemini-2.0-flash"     # Google Gemini
MODEL = "openai/gpt-4o-mini"          # OpenAI
```

LiteLLM automatically picks the right API key from your `.env`.

---

## 🧩 How It Works

1. **Chunk** — Documents are split into small overlapping pieces
2. **Embed** — Each chunk is converted to a vector (list of numbers) by a local HuggingFace model
3. **Store** — Vectors are saved in ChromaDB on your disk
4. **Retrieve** — User question is converted to a vector; closest chunks are fetched
5. **Generate** — Chunks + question are sent to an LLM with strict instructions to only use the provided context

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| [LangChain](https://langchain.com) | Orchestration framework |
| [ChromaDB](https://trychroma.com) | Local vector database |
| [LiteLLM](https://litellm.ai) | Universal LLM API wrapper |
| [HuggingFace](https://huggingface.co) | Local embedding model |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loads `.env` secrets |
