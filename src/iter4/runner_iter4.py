import logging
import time
from typing import Any, Dict, Tuple

from shared.tools import Tools
from iter4.graph_iter4 import build_graph
from iter3.hitl_queue import queue
from shared.metrics import record_hitl_auto, record_report

logger = logging.getLogger(__name__)


def _initial_state(upload_path: str) -> Dict[str, Any]:
    return {
        "upload_path": upload_path,
        "new_policy_clauses": [],
        "candidate_matches": {},
        "conflicts": [],
        "report_md": "",
        "review_ids": [],
        "remediations": {},
        "audit_log": [],
        "plan": {},
        "metrics_iteration": 4,
    }


def run(
    upload_path: str,
    *,
    approve_all: bool = False,
    unattended: bool = False,
    use_interrupt: bool = True,
    mcp_endpoint: str | None = None,
) -> Tuple[list, str]:
    """
    Happy path: execute the iteration 4 multi-agent workflow and return conflicts plus report.
    """
    total_start = time.time()
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 18 + "ITERATION 4: AGENTIC MCP WORKFLOW" + " " * 20 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")

    if approve_all:
        use_interrupt = False
        unattended = False
        logger.info("[Iter4-Runner] Auto-approval enabled; disabling interrupts and unattended mode.")

    tools = Tools()
    graph = build_graph(
        tools,
        use_interrupt=use_interrupt,
        unattended=unattended,
        mcp_endpoint=mcp_endpoint,
    )
    logger.info("[Iter4-Runner] ✓ Graph compiled successfully")

    config = {"configurable": {"thread_id": f"iter4-{int(time.time())}"}}
    state = _initial_state(upload_path)
    logger.info("[Iter4-Runner] 🎬 Starting workflow execution")
    out = graph.invoke(state, config=config)

    snapshot = graph.get_state(config)
    if snapshot and snapshot.next:
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 17 + "WORKFLOW PAUSED - AWAITING REVIEW" + " " * 21 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        values = snapshot.values or {}
        review_ids = values.get("review_ids", [])
        logger.info("[Iter4-Runner] ⏸️  Pending review items: %s", review_ids)
        logger.info("[Iter4-Runner] 💡 Resume later with iter4.resume_workflow(graph, config)")
        return values.get("conflicts", []), values.get("report_md", "")

    if approve_all and out.get("review_ids"):
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "AUTO-APPROVING QUEUED ITEMS" + " " * 34 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        for idx, cid in enumerate(out["review_ids"]):
            queue.label(cid, label="approve", notes="auto-approve iteration 4")
            logger.info("[Iter4-Runner]   ✓ Auto-approved %s (%d/%d)", cid, idx + 1, len(out["review_ids"]))
            tools.remember_assistant(f"Auto-approved conflict {cid} (iteration 4).")
        record_hitl_auto(iteration=4, count=len(out["review_ids"]))
        out = graph.invoke(None, config=config)

    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 24 + "ITERATION 4 COMPLETED" + " " * 32 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("[Iter4-Runner] ⏱️  Total execution time: %.2f seconds", total_elapsed)
    logger.info("[Iter4-Runner] 📊 Final conflicts: %d", len(out.get("conflicts", [])))
    record_report(
        iteration=4,
        conflicts=len(out.get("conflicts", [])),
        report_length=len(out.get("report_md", "")),
        duration=0.0,
    )
    return out.get("conflicts", []), out.get("report_md", "")


def resume_workflow(graph, config=None) -> Tuple[list, str]:
    """
    Happy path: resume a paused iteration 4 workflow after human approval is complete.
    """
    config = config or {"configurable": {"thread_id": "iter4-resume"}}
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 24 + "RESUMING ITERATION 4" + " " * 30 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    out = graph.invoke(None, config=config)
    logger.info("[Iter4-Runner] ✅ Workflow resumed successfully.")
    logger.info("[Iter4-Runner] 📊 Final conflicts: %d", len(out.get("conflicts", [])))
    return out.get("conflicts", []), out.get("report_md", "")


