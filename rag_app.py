# IMPORTS

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM

load_dotenv()

# ---------------------------------------------------------
# MODEL CONFIGURATION
# Change this one string to switch to any provider/model.
# LiteLLM auto-detects the right API key from your .env file.
#
# Examples:
#   "ollama/llama3.2"                       -> local, no key needed
#   "openai/gpt-4o-mini"                    -> needs OPENAI_API_KEY
#   "anthropic/claude-3-5-sonnet-20241022"  -> needs ANTHROPIC_API_KEY
#   "groq/llama-3.1-8b-instant"             -> needs GROQ_API_KEY
#   "gemini/gemini-2.0-flash"               -> needs GEMINI_API_KEY
#   "mistral/mistral-large-latest"          -> needs MISTRAL_API_KEY
# ---------------------------------------------------------
MODEL = "groq/llama-3.1-8b-instant"



# CONNECTS TO CHROMADB VECTOR STORE

print("1. Loading embedding model & connecting to local ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Convert vector store to a Retriever (fetches top 2 matching chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# CREATES STRICT PROMPT TEMPLATE FOR AI

prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant. Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, strictly say: "I cannot find that in the documents."

Context:
{context}

Question:
{question}

Answer:
""")

# INSTANTIATES THE LLM VIA LITELLM
llm = ChatLiteLLM(model=MODEL, temperature=0)



# DEFINES THE COMPLETE RAG EXECUTION PIPELINE

def ask_rag(user_question: str):
    print(f"\n==========================================")
    print(f"USER QUESTION: '{user_question}'")
    print(f"==========================================")
    
    # Retrieves relevant text chunks from ChromaDB
    matching_docs = retriever.invoke(user_question)
    context_text = "\n\n".join([doc.page_content for doc in matching_docs])
    
    print("\n[1. RETRIEVED CONTEXT FROM CHROMADB]")
    print(context_text)
    
    # Fills prompt template with retrieved context + user question
    formatted_messages = prompt_template.format_prompt(
        context=context_text,
        question=user_question
    ).to_messages()

    # Passes formatted messages into LLM via the standard .invoke() interface
    print(f"\n[2. GENERATING ANSWER VIA {MODEL.upper()}...]")
    response = llm.invoke(formatted_messages)
    
    print("\n[3. FINAL GROUNDED AI ANSWER]")
    print(response.content)


if __name__ == "__main__":
    # RUNS RAG PIPELINE TESTS (Targeting sample.pdf content)
    ask_rag("What is the main subject of this document?")
    ask_rag("Can you summarize the core conclusions of the paper?")
    ask_rag("What is Srisawad Corporation Public Company Limited (SAWAD)")