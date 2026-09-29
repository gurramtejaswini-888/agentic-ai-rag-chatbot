from pathlib import Path
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PDF_PATH,
)


gemini = genai.Client(api_key=GEMINI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)


def get_embedding(text: str):
    result = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        ),
    )
    return result.embeddings[0].values


def read_pdf():
    reader = PdfReader(str(PDF_PATH))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))

    return pages


def create_chunks():
    pages = read_pdf()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_number, text in pages:
        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:
            if chunk.strip():
                chunks.append(
                    {
                        "text": chunk.strip(),
                        "page": page_number,
                    }
                )

    return chunks


def get_index():
    existing_indexes = pinecone.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")

        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Waiting for Pinecone index...")
        while True:
            description = pinecone.describe_index(PINECONE_INDEX_NAME)
            status = getattr(description, "status", None)
            ready = getattr(status, "ready", False)

            if ready:
                break

            time.sleep(2)

    return pinecone.Index(PINECONE_INDEX_NAME)


def ingest():
    print("Reading PDF...")
    chunks = create_chunks()

    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        raise RuntimeError("No text could be extracted from the PDF.")

    index = get_index()

    print("Creating embeddings and uploading to Pinecone...")

    vectors = []

    for i, chunk in enumerate(chunks):
        print(f"Processing {i + 1}/{len(chunks)}", end="\r")

        embedding = get_embedding(chunk["text"])

        vectors.append(
            {
                "id": f"chunk-{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "page": chunk["page"],
                },
            }
        )

        if len(vectors) >= 50:
            index.upsert(vectors=vectors)
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)

    print()
    print("INGESTION COMPLETE")
    print(f"Total chunks uploaded: {len(chunks)}")
    print(f"Pinecone index: {PINECONE_INDEX_NAME}")


if __name__ == "__main__":
    ingest()
