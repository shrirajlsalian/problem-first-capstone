from __future__ import annotations

import re
import json
from typing import Any, Callable, List, Dict, Protocol
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class RetrievalStrategy(Protocol):
    def retrieve(self, query: str, k: int = 12) -> List[Document]:
        ...


class NLIClassifier(Protocol):
    def compare(self, a: str, b: str) -> Dict[str, Any]:
        ...


class Reporter(Protocol):
    def build(self, conflicts: List[Dict[str, Any]], upload_name: str) -> str:
        ...


class HybridRetrieval:
    """Default retrieval strategy: dense (Chroma) + sparse (BM25)."""
    def __init__(self, vector_store: Any, bm25_corpus_supplier: Callable[[], List[Document]]):
        self.vector_store = vector_store
        self.bm25_corpus_supplier = bm25_corpus_supplier

    def retrieve(self, query: str, k: int = 12) -> List[Document]:
        dense_docs = self.vector_store.similarity_search(query, k=k)
        corpus = self.bm25_corpus_supplier()
        if corpus:
            bm25 = BM25Retriever.from_documents(corpus)
            sparse_docs = bm25.get_relevant_documents(query)[:k]
        else:
            sparse_docs = []
        return dense_docs + sparse_docs


class LLMNLI:
    """Default NLI via the configured Chat LLM and prompt."""
    def __init__(self, llm: Any, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def compare(self, a: str, b: str) -> Dict[str, Any]:
        prompt_msg = self.prompt.format_prompt(a=a, b=b)
        out = self.llm.invoke(prompt_msg.to_messages())
        try:
            return json.loads(out.content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", out.content)
            return json.loads(m.group(0)) if m else {
                "relation": "unrelated",
                "confidence": 0.3,
                "rationale": "Parse error",
            }


class LLMReporter:
    """Default report builder via the configured Chat LLM and prompt."""
    def __init__(self, llm: Any, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def build(self, conflicts: List[Dict[str, Any]], upload_name: str) -> str:
        prompt_msg = self.prompt.format_prompt(
            conflicts_json=json.dumps(conflicts, ensure_ascii=False),
            upload_name=upload_name,
        )
        out = self.llm.invoke(prompt_msg.to_messages())
        return out.content


