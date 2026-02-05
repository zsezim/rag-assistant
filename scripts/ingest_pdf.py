from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag_assistant.settings import settings


def stable_doc_id(path: str) -> str:
    """Stable ID based on absolute path + file mtime + size."""
    p = Path(path).expanduser().resolve()
    stat = p.stat()
    raw = f"{p}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def add_ids(chunks: list[Document], doc_id: str, filename: str) -> list[Document]:
    """Add doc_id + chunk_id + filename"""
    out: list[Document] = []
    for idx, d in enumerate(chunks):
        md = dict(d.metadata or {})
        md["doc_id"] = doc_id
        md["chunk_id"] = f"{doc_id}:{idx:05d}"
        md["filename"] = filename
        out.append(Document(page_content=d.page_content, metadata=md))
    return out


def load_pdf_docs(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load() 
    filename = Path(pdf_path).name
    for d in docs:
        md = dict(d.metadata or {})
        md["source"] = filename
        md["filename"] = filename
        d.metadata = md
    return docs


def ingest_paths(paths: Iterable[str], overwrite: bool = False) -> None:
    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    vectordb = Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.persist_dir,
        embedding_function=emb,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=getattr(settings, "chunk_size", 1000),
        chunk_overlap=getattr(settings, "chunk_overlap", 150),
        separators=["\n\n", "\n", " ", ""],
    )

    total_added = 0
    for p in paths:
        pdf_path = str(Path(p).expanduser().resolve())
        if not Path(pdf_path).exists() or Path(pdf_path).suffix.lower() != ".pdf":
            print(f"Skipping (not a PDF / missing): {pdf_path}")
            continue

        filename = Path(pdf_path).name
        doc_id = stable_doc_id(pdf_path)

        if overwrite:
            # delete existing chunks
            deleted = vectordb._collection.delete(where={"doc_id": doc_id})
            print(f"Overwrite enabled: removed existing entries for doc_id={doc_id} file={filename}")

        pages = load_pdf_docs(pdf_path)
        chunks = splitter.split_documents(pages)
        chunks = add_ids(chunks, doc_id=doc_id, filename=filename)

        vectordb.add_documents(chunks)
        total_added += len(chunks)

        print(f"Ingested {filename}: pages={len(pages)} chunks={len(chunks)} doc_id={doc_id}")

    vectordb.persist()
    print(f"\nDone. Added total chunks: {total_added}")
    print(f"Chroma: dir={settings.persist_dir} collection={settings.collection_name}")


def expand_inputs(inputs: list[str]) -> list[str]:
    """Accept files and/or directories; expand dirs -> all PDFs within."""
    out: list[str] = []
    for x in inputs:
        p = Path(x).expanduser()
        if p.is_dir():
            out.extend([str(f) for f in sorted(p.glob("*.pdf"))])
        else:
            out.append(str(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True, help="PDF files and/or directories containing PDFs")
    ap.add_argument("--overwrite", action="store_true", help="Re-ingest PDFs (delete prior chunks for same doc_id)")
    args = ap.parse_args()

    ingest_paths(expand_inputs(args.paths), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
