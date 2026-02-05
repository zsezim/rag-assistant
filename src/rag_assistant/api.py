from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from rag_assistant.ingest import ingest_pdf, collection_count
from rag_assistant.rag import RAG


app = FastAPI(title="RAG Assistant API", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    filename: Optional[str] = None
    page: Optional[int] = None
    doc_id: Optional[str] = None
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": collection_count(),
        "collection": os.getenv("RAG_COLLECTION", "docs"),
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), overwrite: bool = True):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to a temp path
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            result = ingest_pdf(str(tmp_path), overwrite=overwrite)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return result


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    rag = RAG()

    try:
        res = rag.ask(
            req.question,
            filename=req.filename,
            page=req.page,
            doc_id=req.doc_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {e}")

    sources = []
    for i, d in enumerate(res.docs, start=1):
        md = d.metadata or {}
        sources.append(
            {
                "rank": i,
                "filename": md.get("filename", md.get("source", "unknown")),
                "page": md.get("page"),
                "doc_id": md.get("doc_id"),
                "chunk_id": md.get("chunk_id"),
                "snippet": d.page_content[:220].replace("\n", " "),
            }
        )

    return AskResponse(answer=res.answer, sources=sources)
