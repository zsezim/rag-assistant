from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # Vector storage
    persist_dir: str = os.getenv("RAG_CHROMA_DIR", "chroma_store")
    collection_name: str = os.getenv("RAG_COLLECTION", "docs")

    # Embeddings
    embedding_model: str = os.getenv(
        "RAG_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Generator
    llm_model: str = os.getenv(
        "RAG_LLM_MODEL",
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    # Chunking
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
settings = Settings()
