from __future__ import annotations
from rag_assistant.retriever import Retriever
from rag_assistant.generator import Generator
from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document


@dataclass
class RAGResult:
    answer: str
    docs: List[Document]


class RAG:
    def __init__(self):
        self.retriever = Retriever.from_settings()
        self.generator = Generator()

    def _has_index(self) -> bool:
        # if collection empty, count is 0
        try:
            return self.retriever.vectordb._collection.count() > 0
        except Exception:
            return True 

    def ask(
        self,
        question: str,
        filename: Optional[str] = None,
        page: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> RAGResult:
        if not self._has_index():
            raise RuntimeError(
                "No documents ingested yet. Run:\n"
                "  python scripts/ingest_pdfs.py --paths <pdf_or_folder>\n"
                "Then try asking again."
            )

        docs = self.retriever.retrieve(
            question,
            filename=filename,
            page=page,
            doc_id=doc_id,
        )

        answer = self.generator.generate(question, docs)
        return RAGResult(answer=answer, docs=docs)
