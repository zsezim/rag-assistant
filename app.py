from __future__ import annotations

import sys
from pathlib import Path

# To make src importable on HF Spaces
sys.path.insert(0, str(Path(__file__).parent / "src"))

import shutil
import tempfile
import traceback

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from rag_assistant.ingest import ingest_pdf, collection_count
from rag_assistant.rag import RAG


def ingest_ui(pdf_path: str):
    """
    Returns: (status_text, doc_id_state_value)
    """
    try:
        if not pdf_path:
            return "Upload a PDF first.", None
        if not str(pdf_path).lower().endswith(".pdf"):
            return "Please upload a PDF file.", None

        # copy into a temp path
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / Path(pdf_path).name
            shutil.copyfile(pdf_path, dst)

            # Ingest
            result = ingest_pdf(str(dst), overwrite=True)

        status = (
            f"Ingested {result['filename']} | pages={result['pages']} | "
            f"chunks={result['chunks']} | doc_id={result['doc_id']} | "
            f"indexed_chunks_total={collection_count()}"
        )
        return status, result["doc_id"]

    except Exception as e:
        print("INGEST ERROR:", repr(e))
        traceback.print_exc()
        return f"Ingest failed: {e!r}", None


def chat_ui(message: str, history, current_doc_id: str | None):
    """
    ChatInterface passes: (message, history, *additional_inputs)
    """
    message = (message or "").strip()
    if not message:
        return "Ask a question."

    if not current_doc_id:
        return "Upload a PDF and click **Ingest / Re-ingest** first."

    # (Optional) still useful as a sanity check
    if collection_count() == 0:
        return "No documents indexed yet. Upload a PDF and click **Ingest / Re-ingest** first."

    try:
        rag = RAG()
        # IMPORTANT: filter retrieval to this session's uploaded doc only
        res = rag.ask(message, doc_id=current_doc_id)
    except Exception as e:
        return f"{e!r}"

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

def clear_ui(current_doc_id: str | None):
    if not current_doc_id:
        return "No document to clear.", None

    try:
        from rag_assistant.settings import settings
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        vectordb = Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.persist_dir,
            embedding_function=emb,
        )

        # Delete only this user's doc
        vectordb._collection.delete(where={"doc_id": current_doc_id})

        return "Cleared your document from the index.", None

    except Exception as e:
        return f"Failed to clear document: {e!r}", current_doc_id

with gr.Blocks(title="RAG Assistant (PDF)") as demo:
    current_doc_id = gr.State(value=None)

    gr.Markdown(
        "# RAG Assistant (PDF)\n"
        "Upload a PDF, click **Ingest / Re-ingest**, then ask questions grounded in *that* document.\n\n"
        "**Note:** Each user session searches only the most recently uploaded document in that session."
    )

    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        ingest_btn = gr.Button("Ingest / Re-ingest")
        clear_btn = gr.Button("Clear My Document")
    status = gr.Textbox(label="Status", interactive=False)

    # Update both the status box and the session doc_id state
    #btn.click(fn=ingest_ui, inputs=[pdf], outputs=[status, current_doc_id])
    
    ingest_btn.click(fn=ingest_ui, inputs=[pdf], outputs=[status, current_doc_id])
    clear_btn.click(fn=clear_ui, inputs=[current_doc_id], outputs=[status, current_doc_id])
    # Pass state into chat via additional_inputs
    gr.ChatInterface(fn=chat_ui, additional_inputs=[current_doc_id])

demo.launch()
