# Agentic AI RAG Chatbot

Document-grounded RAG chatbot using Python, LangGraph, Pinecone, Google Gemini, and FastAPI.

## Features
- PDF ingestion and chunking
- Gemini embeddings
- Pinecone vector search
- LangGraph RAG workflow
- Grounded answers
- Confidence scoring
- FastAPI Swagger API

## Run

Install dependencies:
``n
pip install -r requirements.txt
``n
Ingest the PDF:
``n
python -m src.ingestion
``n
Start the API:
``n
python -m uvicorn app:app --reload
``n
Open http://127.0.0.1:8000/docs

## API Response
``json
{
  "query": "What is Agentic AI?",
  "final_answer": "Answer based on the provided document...",
  "retrieved_context_chunks": [
    "Relevant document chunk..."
  ],
  "confidence_score": 0.88
}
``
