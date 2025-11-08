import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared.schemas import Clause, Conflict, PolicyState
from shared.tools import Tools
from iter1.graph_iter1 import build_graph as build_iter1_graph
from iter2.graph_iter2 import build_graph as build_iter2_graph, AgenticState
from iter3.graph_iter3 import node_gate_enqueue
from iter3.hitl_queue import queue
from shared.metrics import (
    record_planner_strategy,
    record_web_snippets,
    record_hitl_mcp,
    record_hitl_auto,
    record_hitl_cli,
)

logger = logging.getLogger(__name__)

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator


def _ensure_list(container: Dict[str, Any], key: str) -> List[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        value = []
        container[key] = value
    return value


def _ensure_dict(container: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        value = {}
        container[key] = value
    return value


@dataclass
class PlannerAgent:
    """Decides which downstream agents should run for iteration 4."""

    tools: Tools
    default_strategy: str = field(default_factory=lambda: os.getenv("ITER4_STRATEGY", "judge"))

    @traceable(run_name="iter4_planner")
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Happy path: choose detection strategy and record routing metadata."""
        audit = _ensure_list(state, "audit_log")
        strategy = (self.default_strategy or "judge").lower()
        if strategy not in {"baseline", "judge"}:
            strategy = "judge"

        needs_remediator = True
        require_hitl = True
        if os.getenv("ITER4_AUTOPILOT", "").lower() in {"1", "true", "yes"}:
            require_hitl = False

        plan = {
            "strategy": strategy,
            "use_iter2": strategy == "judge",
            "needs_remediation": needs_remediator,
            "require_hitl": require_hitl,
        }
        state["plan"] = plan
        audit.append({"event": "plan", "plan": plan})
        logger.info("[Iter4-Planner] Selected plan=%s", plan)
        record_planner_strategy(strategy)
        return state


@dataclass
class RetrievalAgent:
    """Runs baseline or judge-enhanced retrieval/detection as a delegated agent."""

    tools: Tools

    def __post_init__(self) -> None:
        """Precompile graphs for reuse."""
        self._iter1_graph = build_iter1_graph(self.tools)
        self._iter2_graph = build_iter2_graph(self.tools)

    @traceable(run_name="iter4_retriever")
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Happy path: invoke iteration 1 or 2 graphs based on planner strategy."""
        plan = state.get("plan", {})
        use_iter2 = plan.get("use_iter2", True)
        upload_path = state.get("upload_path", "")
        audit = _ensure_list(state, "audit_log")
        logger.info("[Iter4-Retriever] Starting analysis via %s", "iter2" if use_iter2 else "iter1")

        if use_iter2:
            agentic_state: AgenticState = {
                "upload_path": upload_path,
                "new_policy_clauses": [],
                "candidate_matches": {},
                "conflicts": [],
                "report_md": "",
                "traces": [],
            }
            result = self._iter2_graph.invoke(agentic_state)
        else:
            policy_state: PolicyState = {
                "upload_path": upload_path,
                "new_policy_clauses": [],
                "candidate_matches": {},
                "conflicts": [],
                "report_md": "",
            }
            result = self._iter1_graph.invoke(policy_state)

        state["new_policy_clauses"] = result["new_policy_clauses"]
        state["candidate_matches"] = result["candidate_matches"]
        state["conflicts"] = result["conflicts"]
        state["report_md"] = result.get("report_md", "")
        audit.append(
            {
                "event": "analysis_complete",
                "strategy": "iter2" if use_iter2 else "iter1",
                "conflicts": len(state["conflicts"]),
            }
        )
        logger.info("[Iter4-Retriever] Completed analysis, conflicts=%d", len(state["conflicts"]))
        return state


@dataclass
class RemediationAgent:
    """Generates suggested remediation actions for confirmed conflicts."""

    tools: Tools
    max_tokens: int = 150

    @traceable(run_name="iter4_remediator")
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Happy path: synthesize remediation guidance for each conflict."""
        conflicts: List[Conflict] = state.get("conflicts", [])
        remediations = _ensure_dict(state, "remediations")
        if not conflicts:
            logger.info("[Iter4-Remediator] No conflicts detected; skipping remediation.")
            return state

        for cf in conflicts:
            key = f"{cf.new_id}|{cf.existing_id}"
            if key in remediations:
                continue
            context_text = ""
            snippets: List[str] = []
            if self.tools:
                search_query = f"{cf.relation} policy conflict {cf.new_id.split(':')[0]}"
                for result in self.tools.web_search(search_query, max_results=3):
                    snippet = result.get("snippet")
                    if snippet:
                        title = result.get("title") or "Source"
                        snippets.append(f"{title}: {snippet}")
                if snippets:
                    context_text = "\nHelpful references:\n" + "\n".join(snippets[:3])
                    record_web_snippets(iteration=4, snippet_count=len(snippets))
            prompt = (
                "You are a compliance analyst. "
                "Given the following conflict between a new policy clause and an existing clause, "
                "suggest one short remediation recommendation (<= 80 words) that resolves the issue.\n"
                f"New Clause ({cf.new_id}):\n{cf.rationale}\n"
                f"Conflict Type: {cf.relation}\n"
                "Respond with a concise recommendation only."
            )
            if context_text:
                prompt += f"\n{context_text}"
            try:
                response = self.tools.call_llm(prompt)
                suggestion = (response.content or "").strip()
                if not suggestion:
                    suggestion = "Review conflicting clauses with stakeholders to determine required adjustments."
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[Iter4-Remediator] LLM remediation failed: %s", exc)
                suggestion = "Consult compliance team to draft a remediation plan for this conflict."
            remediations[key] = suggestion
        logger.info("[Iter4-Remediator] Generated remediation guidance for %d conflicts", len(remediations))
        return state


@dataclass
class HitlSupervisorAgent:
    """Supervises human-in-the-loop review via CLI prompts or MCP server calls."""

    tools: Optional[Tools] = None
    use_interrupt: bool = True
    unattended: bool = False
    mcp_endpoint: Optional[str] = None

    @traceable(run_name="iter4_hitl_supervisor")
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Happy path: enqueue conflicts for review and either pause or route to MCP."""
        audit = _ensure_list(state, "audit_log")
        state["metrics_iteration"] = 4
        state = node_gate_enqueue(state)  # reuse Iteration 3 gating for thresholds
        review_ids = state.get("review_ids", [])
        if not review_ids:
            audit.append({"event": "hitl_skipped", "reason": "no_review_items"})
            logger.info("[Iter4-HITL] No conflicts queued for human review.")
            return state

        open_items = [item for item in queue.list_open() if item.get("status", "open") == "open"]
        audit.append({"event": "hitl_pending", "count": len(open_items)})

        if self.unattended and self.mcp_endpoint:
            try:
                import requests  # pragma: no cover - optional dependency

                payload = {
                    "upload_path": state.get("upload_path"),
                    "items": open_items,
                }
                resp = requests.post(
                    f"{self.mcp_endpoint.rstrip('/')}/queue", json=payload, timeout=5
                )
                resp.raise_for_status()
                audit.append({"event": "mcp_dispatch", "endpoint": self.mcp_endpoint, "status": resp.status_code})
                logger.info("[Iter4-HITL] Dispatched %d items to MCP endpoint %s", len(open_items), self.mcp_endpoint)
                if self.tools:
                    self.tools.remember_assistant(
                        f"Dispatched {len(open_items)} conflicts to MCP endpoint {self.mcp_endpoint}"
                    )
                record_hitl_mcp(iteration=4, count=len(open_items))
                return state
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning("[Iter4-HITL] MCP dispatch failed (%s); falling back to CLI prompt.", exc)

        if not self.use_interrupt:
            for cid in review_ids:
                queue.label(cid, label="approve", notes="auto-approve (no interrupt)")
            audit.append({"event": "hitl_auto", "approved": len(review_ids)})
            logger.info("[Iter4-HITL] Auto-approved %d queued items (no interrupt mode).", len(review_ids))
            if self.tools:
                self.tools.remember_assistant(f"Auto-approved {len(review_ids)} conflicts without interrupt.")
            record_hitl_auto(iteration=4, count=len(review_ids))
            return state

        logger.info("[Iter4-HITL] Pausing for manual approval via CLI.")
        approved_count = 0
        rejected_count = 0
        for item in open_items:
            cid = item["conflict_id"]
            rel = item.get("relation", "unknown")
            conf = item.get("confidence", 0.0)
            prompt = f"Do you approve conflict {cid} (relation={rel}, confidence={conf:.2f})? [y/n]: "
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ("y", "yes"):
                    queue.label(cid, label="approve", notes="approved via iter4 CLI")
                    approved_count += 1
                    logger.info("[Iter4-HITL] Approved %s", cid)
                    if self.tools:
                        self.tools.remember_user(f"Approved conflict {cid} (relation={rel}, confidence={conf:.2f})")
                    break
                if user_input in ("n", "no"):
                    queue.label(cid, label="reject", notes="rejected via iter4 CLI")
                    rejected_count += 1
                    logger.info("[Iter4-HITL] Rejected %s", cid)
                    if self.tools:
                        self.tools.remember_user(f"Rejected conflict {cid} (relation={rel}, confidence={conf:.2f})")
                    break
                print("Please enter 'y' or 'n'.")
        audit.append(
            {"event": "hitl_cli_complete", "approved": approved_count, "rejected": rejected_count}
        )
        record_hitl_cli(iteration=4, approved=approved_count, rejected=rejected_count)
        return state


