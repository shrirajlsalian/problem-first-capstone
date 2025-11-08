import logging
import time
from shared.schemas import PolicyState
from shared.tools import Tools
from iter1.graph_iter1 import build_graph

logger = logging.getLogger(__name__)

def run(upload_path: str, indexed_paths=None):
    """Happy path: build the Iteration 1 workflow, run it end-to-end, and return conflicts plus report."""
    total_start_time = time.time()
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "ITERATION 1: BASIC CONFLICT DETECTION" + " " * 20 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    logger.info(f"[Iter1-Runner] 🚀 Initializing Iteration 1 workflow")
    logger.info(f"[Iter1-Runner] 📂 Upload path: {upload_path}")
    
    tools = Tools()
    if indexed_paths:
        logger.info(f"[Iter1-Runner] 📚 Indexing existing policies from: {indexed_paths}")
        tools.index_existing(indexed_paths)
    else:
        logger.info(f"[Iter1-Runner] ℹ️  No indexed paths provided, using existing vector store")
    
    logger.info(f"[Iter1-Runner] 🏗️  Building LangGraph workflow graph...")
    g = build_graph(tools)
    logger.info(f"[Iter1-Runner] ✓ Graph built successfully")
    
    logger.info(f"[Iter1-Runner] 🎬 Initializing state and invoking graph...")
    init: PolicyState = {"upload_path": upload_path, "new_policy_clauses": [], "candidate_matches": {}, "conflicts": [], "report_md": ""}
    
    out = g.invoke(init)
    
    total_elapsed = time.time() - total_start_time
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 25 + "ITERATION 1 COMPLETED" + " " * 33 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"[Iter1-Runner] ⏱️  Total execution time: {total_elapsed:.2f} seconds")
    logger.info(f"[Iter1-Runner] 📊 Final results: {len(out['conflicts'])} conflicts detected")
    logger.info("")
    
    return out["conflicts"], out["report_md"]
