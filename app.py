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

import traceback
from pathlib import Path
import shutil
import tempfile

def ingest_ui(pdf_path: str) -> str:
    try:
        if not pdf_path:
            return "Upload a PDF first."
        if not str(pdf_path).lower().endswith(".pdf"):
            return "Please upload a PDF file."

        # copy into temp to avoid any gradio temp-file edge cases
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / Path(pdf_path).name
            shutil.copyfile(pdf_path, dst)

            result = ingest_pdf(str(dst), overwrite=True)

        # Return ONLY plain text
        return (
            f"Ingested {result['filename']} | pages={result['pages']} | "
            f"chunks={result['chunks']} | doc_id={result['doc_id']} | "
            f"indexed_chunks_total={collection_count()}"
        )

    except Exception as e:
        print("INGEST ERROR:", repr(e))
        traceback.print_exc()
        return f"Ingest failed: {e!r}"


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
    gr.Markdown("# RAG Assistant (PDF)\nUpload a PDF, then ask questions grounded in it.")

    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        btn = gr.Button("Ingest / Re-ingest")

    status = gr.Textbox(label="Status", interactive=False)
    btn.click(fn=ingest_ui, inputs=[pdf], outputs=[status])

    gr.ChatInterface(chat_ui)


demo.launch()
