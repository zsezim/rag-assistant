from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from __future__ import annotations
from settings import settings
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(parts)


class Generator:
    """
    Uses a local HuggingFace model via the transformers pipeline.
    """
    def __init__(self):
        tok = AutoTokenizer.from_pretrained(settings.llm_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model,
            device_map="auto",     
            torch_dtype="auto",
        )

        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.2,
            return_full_text=False,
        )

        self.llm = HuggingFacePipeline(pipeline=gen_pipe)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer ONLY using the provided context. "
             "If the answer isn't in the context, say you don't know."),
            ("human",
             "Question:\n{question}\n\nContext:\n{context}\n\nAnswer (with brief citations like [1], [2]):")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, docs: List[Document]) -> str:
        context = _format_context(docs)
        return self.chain.invoke({"question": question, "context": context})
