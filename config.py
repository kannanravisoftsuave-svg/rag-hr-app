import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# LLM
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"

# Vector store
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Langfuse
LANGFUSE_SECRET_KEY: str = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY: str = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST: str = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# Embeddings / reranking
EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIM: int = 384
QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# RAG defaults
CONFIDENCE_THRESHOLD: float = 0.45
DEFAULT_COLLECTION: str = "documents"
DEFAULT_TOP_K: int = 8

# Ingestion
MAX_FILE_SIZE_MB: int = 10
SUPPORTED_EXTENSIONS: set = {".pdf", ".docx", ".md", ".txt"}
UPLOAD_DIR: str = str(Path(__file__).parent / "uploads")

# UI
API_URL: str = os.environ.get("API_URL", "http://localhost:8000")
