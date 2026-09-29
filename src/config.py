from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

GEMINI_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
TOP_K = 4

PDF_PATH = ROOT_DIR / "data" / "Ebook-Agentic-AI.pdf"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing from .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing from .env")
