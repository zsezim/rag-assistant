from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
import shutil
import tempfile
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from rag_assistant.ingest import ingest_pdf, collection_count
from rag_assistant.rag import RAG


def ingest_ui(pdf_file) -> str:
    if pdf_file is None:
        return "Upload a PDF first."

    src_path = getattr(pdf_file, "name", None)
    if not src_path or not str(src_path).lower().endswith(".pdf"):
        return "Please upload a PDF file."

    with tempfile.TemporaryDirectory() as td:
        dst_path = Path(td) / Path(src_path).name
        shutil.copyfile(src_path, dst_path)

        try:
            result = ingest_pdf(str(dst_path), overwrite=True)
        except Exception as e:
            return f"Ingest failed: {e}"

    return (
        f"Ingested {result['filename']} | pages={result['pages']} | "
        f"chunks={result['chunks']} | doc_id={result['doc_id']} | "
        f"indexed_chunks_total={collection_count()}"
    )


def chat_ui(message: str, history):
    message = (message or "").strip()
    if not message:
        return "Ask a question."

    if collection_count() == 0:
        return "No documents indexed yet. Upload a PDF and click **Ingest / Re-ingest** first."

    try:
        rag = RAG()
        res = rag.ask(message)
    except Exception as e:
        return f"{e}"

    cites = []
    for i, d in enumerate(res.docs, start=1):
        md = d.metadata or {}
        fname = md.get("filename", md.get("source", "unknown"))
        page = md.get("page", "?")
        chunk_id = md.get("chunk_id", "")
        cites.append(f"[{i}] {fname} p{page} ({chunk_id})")

    out = res.answer
    if cites:
        out += "\n\nSources:\n" + "\n".join(cites)

    return out


with gr.Blocks(title="RAG Assistant (PDF)") as demo:
    gr.Markdown(
        "# RAG Assistant (PDF)\n"
        "Upload a PDF, click **Ingest / Re-ingest**, then ask questions grounded in that document."
    )

    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
        btn = gr.Button("Ingest / Re-ingest")

    status = gr.Textbox(label="Status", interactive=False)
    btn.click(fn=ingest_ui, inputs=[pdf], outputs=[status])

    gr.ChatInterface(chat_ui)


demo.launch()
