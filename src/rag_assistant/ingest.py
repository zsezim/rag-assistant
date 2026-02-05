from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_assistant.settings import settings


def file_sha256_id(path: str) -> str:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _format_source_metadata(docs: list[Document], filename: str) -> list[Document]:
    out = []
    for d in docs:
        md = dict(d.metadata or {})
        md["source"] = filename
        md["filename"] = filename
        out.append(Document(page_content=d.page_content, metadata=md))
    return out


def _add_chunk_ids(chunks: list[Document], doc_id: str, filename: str) -> list[Document]:
    out: list[Document] = []
    for idx, d in enumerate(chunks):
        md = dict(d.metadata or {})
        md["doc_id"] = doc_id
        md["chunk_id"] = f"{doc_id}:{idx:05d}"
        md["filename"] = filename
        out.append(Document(page_content=d.page_content, metadata=md))
    return out


def ingest_pdf(pdf_path: str, overwrite: bool = False) -> dict:
    pdf_path = str(Path(pdf_path).expanduser().resolve())
    p = Path(pdf_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF not found (or not a .pdf): {pdf_path}")

    filename = p.name
    doc_id = file_sha256_id(pdf_path)

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectordb = Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.persist_dir,
        embedding_function=emb,
    )

    if overwrite:
        vectordb._collection.delete(where={"doc_id": doc_id})

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  
    pages = _format_source_metadata(pages, filename=filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=getattr(settings, "chunk_size", 1000),
        chunk_overlap=getattr(settings, "chunk_overlap", 150),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    chunks = _add_chunk_ids(chunks, doc_id=doc_id, filename=filename)

    vectordb.add_documents(chunks)
    vectordb.persist()

    return {
        "filename": filename,
        "doc_id": doc_id,
        "pages": len(pages),
        "chunks": len(chunks),
        "persist_dir": settings.persist_dir,
        "collection": settings.collection_name,
        "overwrite": overwrite,
    }


def ingest_many(paths: Iterable[str], overwrite: bool = False) -> list[dict]:
    results: list[dict] = []
    for path in paths:
        results.append(ingest_pdf(path, overwrite=overwrite))
    return results


def collection_count() -> int:
    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectordb = Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.persist_dir,
        embedding_function=emb,
    )
    return int(vectordb._collection.count())
