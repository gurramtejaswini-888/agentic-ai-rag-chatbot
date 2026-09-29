import re

from typing import TypedDict

from google import genai
from pinecone import Pinecone

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)

from src.ingestion import get_embedding

from langgraph.graph import StateGraph, START, END


gemini = genai.Client(api_key=GEMINI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)


class RAGState(TypedDict):
    query: str
    retrieved_context_chunks: list[str]
    scores: list[float]
    final_answer: str
    confidence_score: float


def retrieve(state: RAGState):
    query = state["query"]

    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    chunks = []
    scores = []

    for match in results.matches:
        metadata = match.metadata or {}
        text = metadata.get("text", "")

        if text:
            page = metadata.get("page", "unknown")
            chunks.append(f"[Page {page}] {text}")
            scores.append(float(match.score or 0))

    return {
        "retrieved_context_chunks": chunks,
        "scores": scores,
    }


def generate_answer(state: RAGState):
    query = state["query"]
    chunks = state["retrieved_context_chunks"]
    scores = state["scores"]

    if not chunks:
        return {
            "final_answer": "I could not find relevant information in the provided document.",
            "confidence_score": 0.0,
        }

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""
You are a document-grounded RAG assistant.

Answer the user's question ONLY using the provided document context.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not supported by the context, say:
   "I could not find this information in the provided document."
4. Keep the answer clear and concise.
5. When possible, explain the answer using information from the retrieved context.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    answer = response.text.strip()

    top_score = max(scores) if scores else 0.0

    # Retrieval-based confidence score.
    # Pinecone cosine similarity is converted into a bounded 0-1 score.
    confidence = max(0.0, min(0.99, top_score))

    # Reduce confidence when the model explicitly says the information
    # is not available in the document.
    refusal_phrases = [
        "could not find this information",
        "not found in the provided document",
        "not provided in the document",
        "not mentioned in the document",
        "does not contain this information",
    ]

    if any(phrase in answer.lower() for phrase in refusal_phrases):
        confidence = min(confidence, 0.20)

    return {
        "final_answer": answer,
        "confidence_score": round(confidence, 2),
    }


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_answer)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


rag_graph = build_graph()


def ask_question(query: str):
    result = rag_graph.invoke(
        {
            "query": query,
            "retrieved_context_chunks": [],
            "scores": [],
            "final_answer": "",
            "confidence_score": 0.0,
        }
    )

    return {
        "query": query,
        "final_answer": result["final_answer"],
        "retrieved_context_chunks": result["retrieved_context_chunks"],
        "confidence_score": result["confidence_score"],
    }


if __name__ == "__main__":
    result = ask_question(
        "What is Agentic AI?"
    )

    print(result)
