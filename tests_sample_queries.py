from src.graph import ask_question

queries = [
    "What is Agentic AI?",
    "What are the key components of Agentic AI?",
    "What are the applications of Agentic AI?",
    "How does Agentic AI differ from traditional AI?",
    "What are the challenges of Agentic AI?",
    "What is the capital of France?",
]

for i, query in enumerate(queries, start=1):
    print("\n" + "=" * 70)
    print(f"TEST {i}: {query}")
    print("=" * 70)

    result = ask_question(query)

    print("Answer:")
    print(result["final_answer"])

    print("\nConfidence:", result["confidence_score"])
    print("Retrieved chunks:", len(result["retrieved_context_chunks"]))

print("\nALL 6 VALIDATION QUERIES COMPLETED")
