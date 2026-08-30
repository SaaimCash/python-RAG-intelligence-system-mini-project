import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vector_store import process_and_store_document
from rag_app import retriever, prompt_template, llm

app = FastAPI(
    title="RAG Lab API",
    description="Backend service for document vectorization and grounded Q&A.",
    version="1.0.0"
)

# Enable CORS so the browser frontend can talk to FastAPI without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "online", "message": "RAG API server is running."}


@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to receive a PDF or TXT file over HTTP, save it temporarily,
    and process its vectors into ChromaDB.
    """
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file stream to disk temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Vectorize and embed into ChromaDB
        process_and_store_document(temp_file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "message": "File successfully embedded and saved into ChromaDB."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
    finally:
        # Clean up temporary upload file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/api/v1/ask")
def ask_question(request: QueryRequest):
    """
    Endpoint to query the RAG pipeline. Searches ChromaDB and returns
    grounded LLM response + retrieved context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # 1. Retrieve matching chunks from ChromaDB
        matching_docs = retriever.invoke(request.question)
        context_text = "\n\n".join([doc.page_content for doc in matching_docs])

        # 2. Format prompt template
        formatted_prompt = prompt_template.format(
            context=context_text,
            question=request.question
        )

        # 3. Generate answer via LLM
        response = llm.invoke(formatted_prompt)

        return {
            "question": request.question,
            "answer": response.content,
            "context_retrieved": [doc.page_content for doc in matching_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation error: {str(e)}")