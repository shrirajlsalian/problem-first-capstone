import logging
import time
from shared.tools import Tools
from iter2.graph_iter2 import build_graph, AgenticState

logger = logging.getLogger(__name__)

def run(upload_path: str):
    """Happy path: execute Iteration 2 with judge verification and return the final conflicts and report."""
    total_start_time = time.time()
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 18 + "ITERATION 2: AGENTIC JUDGE VERIFICATION" + " " * 20 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    logger.info(f"[Iter2-Runner] 🚀 Initializing Iteration 2 workflow with judge verification")
    logger.info(f"[Iter2-Runner] 📂 Upload path: {upload_path}")
    logger.info(f"[Iter2-Runner] 🎯 Enhanced with: LLM judge verification and confidence adjustment")
    
    tools = Tools()
    logger.info(f"[Iter2-Runner] 🏗️  Building LangGraph workflow graph...")
    g = build_graph(tools)
    logger.info(f"[Iter2-Runner] ✓ Graph built successfully")
    
    logger.info(f"[Iter2-Runner] 🎬 Initializing state and invoking graph...")
    state: AgenticState = {"upload_path": upload_path, "new_policy_clauses": [], "candidate_matches": {}, "conflicts": [], "report_md": "", "traces": []}
    
    out = g.invoke(state)
    
    total_elapsed = time.time() - total_start_time
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 25 + "ITERATION 2 COMPLETED" + " " * 33 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"[Iter2-Runner] ⏱️  Total execution time: {total_elapsed:.2f} seconds")
    logger.info(f"[Iter2-Runner] 📊 Final results: {len(out['conflicts'])} verified conflicts detected")
    logger.info(f"[Iter2-Runner] 📋 Judge traces: {len(out.get('traces', []))} verification records")
    logger.info("")
    
    return out["conflicts"], out["report_md"]
