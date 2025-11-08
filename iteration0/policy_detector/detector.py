from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import PolicyConfig
from .types import Clause, Conflict, PolicyState
from .hooks import Hooks
from .strategies import (
    RetrievalStrategy,
    NLIClassifier,
    Reporter,
    HybridRetrieval,
    LLMNLI,
    LLMReporter,
)


class PolicyConflictDetector:
    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        nli_classifier: Optional[NLIClassifier] = None,
        reporter: Optional[Reporter] = None,
        hooks: Optional[Hooks] = None,
    ) -> None:
        self.config = config or PolicyConfig()

        self.embeddings = OpenAIEmbeddings(model=self.config.embed_model)
        self.vector_store = Chroma(
            collection_name="policies",
            embedding_function=self.embeddings,
            persist_directory=self.config.chroma_dir,
        )

        self.bm25_corpus: List[Document] = []

        self.llm = ChatOpenAI(model=self.config.openai_model, temperature=0)

        self.nli_prompt = PromptTemplate(
            input_variables=["a", "b"],
            template=(
                "You are a policy contradiction checker.\n"
                "Given two clauses, determine their logical relation:\n"
                "- CONTRADICTION: They cannot both be true / they impose incompatible requirements.\n"
                "- DUPLICATION: They say the same thing.\n"
                "- ENTailment: A entails B or B entails A (not a conflict).\n"
                "- UNRELATED: About different topics.\n\n"
                "Return JSON with fields relation in {contradiction, duplication, entailment, unrelated},"
                " confidence in [0,1], and rationale (1-2 sentences).\n\n"
                "Clause A:\n{{a}}\n\nClause B:\n{{b}}\n\n"
                "JSON:"
            ),
        )

        self.report_prompt = PromptTemplate(
            input_variables=["conflicts_json", "upload_name"],
            template=(
                "You are generating a concise report for a policy inspector.\n"
                "Input is a JSON array of conflict objects with keys new_text, existing_text, relation, confidence, source.\n"
                "Write a tight markdown report with sections: Summary, Conflicts (table), Details, Suggested Remediations.\n"
                "Keep it action-oriented.\n"
                "\nUpload: {{upload_name}}\nConflicts JSON: {{conflicts_json}}\n"
            ),
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "],
        )

        self.hooks = hooks or Hooks()

        self._retrieval_strategy = retrieval_strategy
        self._nli_classifier = nli_classifier
        self._reporter = reporter

        self._create_tools()
        self._graph = None

    # Utilities
    def load_policy_to_documents(self, path: str) -> List[Document]:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext in (".txt", ".csv"):
            return TextLoader(path, encoding="utf-8").load()
        if ext in (".docx", ".doc"):
            return Docx2txtLoader(path).load()
        raise ValueError(f"Unsupported file type: {path}")

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(docs)

    def to_clauses(self, docs: List[Document], source_id: str) -> List[Clause]:
        clauses: List[Clause] = []
        for i, d in enumerate(docs):
            text = d.page_content.strip()
            if not text:
                continue
            clauses.append(
                Clause(id=f"{source_id}:{i}", text=text, source_id=source_id, meta=d.metadata or {})
            )
        return clauses

    # Index
    def add_existing_policies(self, paths: List[str], *, force_rebuild: bool = False) -> None:
        if force_rebuild:
            self.vector_store.delete_collection()
            self.bm25_corpus.clear()
        for p in paths:
            docs = self.load_policy_to_documents(p)
            chunks = self.chunk_documents(docs)
            self.bm25_corpus.extend(chunks)
            self.vector_store.add_documents(chunks)

    # Tool implementations
    def _tool_upload_and_parse_impl(self, path: str) -> Dict[str, Any]:
        docs = self.load_policy_to_documents(path)
        chunks = self.chunk_documents(docs)
        clauses = self.to_clauses(chunks, source_id=os.path.basename(path))
        try:
            self.hooks.on_upload_parsed(clauses)
        except Exception:
            pass
        return {"clauses": [c.__dict__ for c in clauses]}

    def _tool_candidate_retrieval_impl(self, query_clause: str, k: int = 12) -> Dict[str, Any]:
        docs = self.retrieval_strategy.retrieve(query_clause, k=k)
        seen = set()
        candidates: List[Clause] = []
        for d in docs:
            key = (d.page_content[:50], str(d.metadata))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Clause(
                    id=d.metadata.get("source", d.metadata.get("source_id", d.metadata.get("document_id", "unknown")))
                    + ":"
                    + str(d.metadata.get("page", d.metadata.get("chunk", 0))),
                    text=d.page_content,
                    source_id=str(d.metadata.get("source", d.metadata.get("source_id", "unknown"))),
                    meta=d.metadata or {},
                )
            )
        return {"candidates": [c.__dict__ for c in candidates]}

    def _tool_nli_compare_impl(self, a: str, b: str) -> Dict[str, Any]:
        return self.nli_classifier.compare(a, b)

    def _tool_build_report_impl(self, conflicts: List[Dict[str, Any]], upload_name: str) -> str:
        report_md = self.reporter.build(conflicts, upload_name)
        try:
            self.hooks.on_report_built(report_md)
        except Exception:
            pass
        return report_md

    # LangChain tools
    def _create_tools(self) -> None:
        @tool
        def tool_upload_and_parse(path: str) -> Dict[str, Any]:
            return self._tool_upload_and_parse_impl(path)

        @tool
        def tool_candidate_retrieval(query_clause: str, k: int = 12) -> Dict[str, Any]:
            return self._tool_candidate_retrieval_impl(query_clause, k)

        @tool
        def tool_nli_compare(a: str, b: str) -> Dict[str, Any]:
            return self._tool_nli_compare_impl(a, b)

        @tool
        def tool_build_report(conflicts: List[Dict[str, Any]], upload_name: str) -> str:
            return self._tool_build_report_impl(conflicts, upload_name)

        self.tool_upload_and_parse_tool = tool_upload_and_parse
        self.tool_candidate_retrieval_tool = tool_candidate_retrieval
        self.tool_nli_compare_tool = tool_nli_compare
        self.tool_build_report_tool = tool_build_report

    # Graph nodes
    def node_parse_upload(self, state: PolicyState) -> PolicyState:
        path = state["upload_path"]
        parsed = self._tool_upload_and_parse_impl(path)
        clauses = [Clause(**c) for c in parsed["clauses"]]
        state["new_policy_clauses"] = clauses
        return state

    def node_retrieve_candidates(self, state: PolicyState) -> PolicyState:
        mapping: Dict[str, List[Clause]] = {}
        for cl in state["new_policy_clauses"]:
            res = self._tool_candidate_retrieval_impl(cl.text, k=12)
            cands = [Clause(**c) for c in res["candidates"]][:12]
            try:
                self.hooks.on_candidates_retrieved(cl, cands)
            except Exception:
                pass
            mapping[cl.id] = cands
        state["candidate_matches"] = mapping
        return state

    def node_detect_conflicts(self, state: PolicyState) -> PolicyState:
        found: List[Conflict] = []
        for new_cl in state["new_policy_clauses"]:
            for cand in state["candidate_matches"].get(new_cl.id, []):
                verdict = self._tool_nli_compare_impl(new_cl.text, cand.text)
                rel = verdict.get("relation", "unrelated").lower()
                conf = float(verdict.get("confidence", 0))
                if rel in ("contradiction", "duplication") and conf >= 0.55:
                    found.append(
                        Conflict(
                            new_clause_id=new_cl.id,
                            existing_clause_id=cand.id,
                            relation="contradiction" if rel == "contradiction" else "duplication",
                            confidence=conf,
                            rationale=verdict.get("rationale", ""),
                        )
                    )
                elif rel == "unrelated":
                    continue
                else:
                    if conf >= 0.4:
                        found.append(
                            Conflict(
                                new_clause_id=new_cl.id,
                                existing_clause_id=cand.id,
                                relation="possible-conflict",
                                confidence=conf,
                                rationale=verdict.get("rationale", ""),
                            )
                        )
        for cf in found:
            try:
                self.hooks.on_conflict_detected(cf)
            except Exception:
                pass
        state["conflicts"] = found
        return state

    def node_build_report(self, state: PolicyState) -> PolicyState:
        id_to_text: Dict[str, str] = {c.id: c.text for c in state["new_policy_clauses"]}
        existing_map: Dict[str, str] = {}
        for cand_list in state.get("candidate_matches", {}).values():
            for c in cand_list:
                existing_map[c.id] = c.text
        conflicts_json: List[Dict[str, Any]] = []
        for cf in state["conflicts"]:
            conflicts_json.append(
                {
                    "new_id": cf.new_clause_id,
                    "new_text": id_to_text.get(cf.new_clause_id, ""),
                    "existing_id": cf.existing_clause_id,
                    "existing_text": existing_map.get(cf.existing_clause_id, ""),
                    "relation": cf.relation,
                    "confidence": cf.confidence,
                    "rationale": cf.rationale,
                    "source": existing_map.get(cf.existing_clause_id, "")[:80],
                }
            )
        report = self._tool_build_report_impl(conflicts_json, state.get("upload_path") or "")
        state["report_md"] = report
        return state

    # Graph
    def build_graph(self) -> StateGraph:
        graph = StateGraph(PolicyState)
        graph.add_node("parse_upload", self.node_parse_upload)
        graph.add_node("retrieve_candidates", self.node_retrieve_candidates)
        graph.add_node("detect_conflicts", self.node_detect_conflicts)
        graph.add_node("build_report", self.node_build_report)
        graph.add_edge(START, "parse_upload")
        graph.add_edge("parse_upload", "retrieve_candidates")
        graph.add_edge("retrieve_candidates", "detect_conflicts")
        graph.add_edge("detect_conflicts", "build_report")
        graph.add_edge("build_report", END)
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    @property
    def retrieval_strategy(self) -> RetrievalStrategy:
        if self._retrieval_strategy is None:
            self._retrieval_strategy = HybridRetrieval(self.vector_store, lambda: self.bm25_corpus)
        return self._retrieval_strategy

    @property
    def nli_classifier(self) -> NLIClassifier:
        if self._nli_classifier is None:
            self._nli_classifier = LLMNLI(self.llm, self.nli_prompt)
        return self._nli_classifier

    @property
    def reporter(self) -> Reporter:
        if self._reporter is None:
            self._reporter = LLMReporter(self.llm, self.report_prompt)
        return self._reporter

    def get_tools(self) -> List[Any]:
        return [
            self.tool_upload_and_parse_tool,
            self.tool_candidate_retrieval_tool,
            self.tool_nli_compare_tool,
            self.tool_build_report_tool,
        ]

    # Public API
    def run_conflict_detection(self, upload_path: str) -> Tuple[List[Conflict], str]:
        init_state: PolicyState = {
            "upload_path": upload_path,
            "new_policy_clauses": [],
            "candidate_matches": {},
            "conflicts": [],
            "report_md": "",
        }
        final_state = self.build_graph().invoke(init_state)
        return final_state["conflicts"], final_state["report_md"]


