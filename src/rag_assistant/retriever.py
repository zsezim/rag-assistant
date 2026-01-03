from __future__ import annotations
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .settings import settings


class Retriever:
    """
    Wraps a vector store retriever.
    Uses Chroma as the vector store and 
    HuggingFaceEmbeddings for embeddings.
    """
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.persist_dir
        self.collection_name = collection_name or settings.collection_name

        self._emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        self._vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self._emb,
        )

        self._retriever = self._vs.as_retriever(search_kwargs={"k": settings.top_k})

    def retrieve(self, query: str) -> List[Document]:
        """Return top-k documents for a query."""
        return self._retriever.invoke(query)
