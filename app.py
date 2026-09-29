from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask_question


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Document-grounded RAG chatbot using LangGraph, Gemini and Pinecone",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG Chatbot API is running"
    }


@app.post("/query")
def query_rag(request: QueryRequest):
    return ask_question(request.query)
