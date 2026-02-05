from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_assistant.settings import settings


@dataclass
class Retriever:
    vectordb: Chroma

    @classmethod
    def from_settings(cls) -> "Retriever":
        emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        vectordb = Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.persist_dir,
            embedding_function=emb,
        )
        return cls(vectordb=vectordb)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filename: Optional[str] = None,
        page: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> List[Document]:
        where = {}
        if filename:
            where["filename"] = filename
        if page is not None:
            where["page"] = page
        if doc_id:
            where["doc_id"] = doc_id

        search_kwargs = {"k": k or settings.top_k}
        if where:
            search_kwargs["filter"] = where  

        retriever = self.vectordb.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)
