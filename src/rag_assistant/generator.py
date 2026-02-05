from __future__ import annotations
import os
from typing import List
from together import Together
from langchain_core.documents import Document


def _format_context(docs: List[Document]) -> str:
    """Format retrieved docs into a numbered context block with sources for citation."""
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = (d.metadata or {}).get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(parts)


class Generator:
    """
    Generator backed by Together AI (hosted open-source LLMs).
    
    """

    def __init__(self) -> None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing TOGETHER_API_KEY. Set it in your environment or .env file."
            )

        self.model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        self.client = Together(api_key=api_key)

        # System prompt: keep it strict about grounding + citations.
        self.system_prompt = (
            "You are a helpful assistant. Answer only using the provided context. "
            "If the answer isn't in the context, say you don't know. "
            "When you use facts from the context, add brief citations like [1], [2]."
        )

    def generate(self, question: str, docs: List[Document]) -> str:
        context = _format_context(docs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer (with brief citations like [1], [2]):"
                ),
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )

        return resp.choices[0].message.content.strip()
