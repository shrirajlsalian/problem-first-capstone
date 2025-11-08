import logging
import time
from shared.tools import Tools
from iter3.graph_iter3 import build_graph, HITLState
from iter3.hitl_queue import queue
from shared.metrics import record_hitl_cli, record_hitl_auto

logger = logging.getLogger(__name__)

def run(upload_path: str, approve_all=False, use_interrupt=True):
    """
    Happy path: execute Iteration 3 end-to-end, optionally pausing for human approval, and return results.
    
    Args:
        upload_path: Path to the policy document to analyze
        approve_all: If True, auto-approve all queued items (bypasses human review)
        use_interrupt: If True, use LangGraph interrupt to pause for human approval
    """
    total_start_time = time.time()
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 19 + "ITERATION 3: HUMAN-IN-THE-LOOP REVIEW" + " " * 21 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    logger.info(f"[Iter3-Runner] 🚀 Initializing Iteration 3 workflow with HITL review")
    logger.info(f"[Iter3-Runner] 📂 Upload path: {upload_path}")
    logger.info(f"[Iter3-Runner] 🎯 Enhanced with: Critic verification + Human review queue")
    if approve_all:
        logger.info(f"[Iter3-Runner] ⚙️  Auto-approval mode: ENABLED (all queued items will be auto-approved)")
        use_interrupt = False  # Disable interrupt if auto-approving
        logger.info(f"[Iter3-Runner] 🔔 Interrupt disabled (auto-approval mode)")
    else:
        logger.info(f"[Iter3-Runner] ⚙️  Auto-approval mode: DISABLED (manual review required)")
        if use_interrupt:
            logger.info(f"[Iter3-Runner] 🔔 Interrupt ENABLED: workflow will pause for human approval")
        else:
            logger.info(f"[Iter3-Runner] ⚡ Interrupt DISABLED: workflow will run continuously")
    
    tools = Tools()
    logger.info(f"[Iter3-Runner] 🏗️  Building LangGraph workflow graph...")
    g = build_graph(tools, use_interrupt=use_interrupt)
    logger.info(f"[Iter3-Runner] ✓ Graph built successfully")
    
    logger.info(f"[Iter3-Runner] 🎬 Initializing state and invoking graph...")
    config = {"configurable": {"thread_id": "iter3-run"}}  # Thread ID for checkpointing
    state: HITLState = {
        "upload_path": upload_path,
        "new_policy_clauses": [],
        "candidate_matches": {},
        "conflicts": [],
        "report_md": "",
        "review_ids": [],
        "metrics_iteration": 3,
    }
    
    # Invoke graph - this will pause at interrupt if enabled
    out = g.invoke(state, config=config)
    
    # Check if workflow was interrupted (LangGraph sets next field when paused)
    current_state = g.get_state(config)
    if current_state and current_state.next and use_interrupt and not approve_all:
        # Workflow is paused at interrupt
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "WORKFLOW INTERRUPTED - AWAITING APPROVAL" + " " * 20 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info(f"[Iter3-Runner] ⏸️  Workflow paused at interrupt point (before report node)")
        
        if current_state.values:
            state = current_state.values
            review_ids = state.get("review_ids", [])
            logger.info(f"[Iter3-Runner] 📥 {len(review_ids)} items enqueued for review")
            
            open_items = [item for item in queue.list_open() if item.get("status", "open") == "open"]
            if open_items:
                logger.info(f"[Iter3-Runner] 🗳️  Starting interactive approval prompts...")
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
                            queue.label(cid, label="approve", notes="approved via CLI prompt")
                            approved_count += 1
                            logger.info(f"[Iter3-Runner]   ✅ Approved {cid}")
                            tools.remember_user(f"Approved conflict {cid} (relation={rel}, confidence={conf:.2f})")
                            break
                        if user_input in ("n", "no"):
                            queue.label(cid, label="reject", notes="rejected via CLI prompt")
                            rejected_count += 1
                            logger.info(f"[Iter3-Runner]   ❌ Rejected {cid}")
                            tools.remember_user(f"Rejected conflict {cid} (relation={rel}, confidence={conf:.2f})")
                            break
                        print("Please enter 'y' or 'n'.")
                logger.info(f"[Iter3-Runner] 📝 Approval summary: {approved_count} approved, {rejected_count} rejected")
                record_hitl_cli(iteration=3, approved=approved_count, rejected=rejected_count)
            else:
                logger.info(f"[Iter3-Runner] ℹ️  No open items remain in the queue")
            
            # After interactive approval, resume workflow automatically
            conflicts, report = resume_workflow(g, config=config)
            return conflicts, report
    
    # Workflow completed first pass
    first_pass_elapsed = time.time() - total_start_time
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 22 + "FIRST PASS COMPLETED" + " " * 38 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"[Iter3-Runner] ⏱️  First pass time: {first_pass_elapsed:.2f} seconds")
    logger.info(f"[Iter3-Runner] 📊 Conflicts found: {len(state.get('conflicts', []))}")
    logger.info(f"[Iter3-Runner] 📥 Items enqueued for review: {len(state.get('review_ids', []))}")
    
    if state.get('review_ids'):
        logger.info(f"[Iter3-Runner] 📋 Review queue items: {state['review_ids']}")
    
    if approve_all:
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 18 + "AUTO-APPROVING QUEUED ITEMS" + " " * 32 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info(f"[Iter3-Runner] 🤖 Auto-approving {len(state['review_ids'])} queued items...")
        for idx, cid in enumerate(state["review_ids"]):
            queue.label(cid, label="approve", notes="auto-approve for demo")
            logger.info(f"[Iter3-Runner]   ✓ Auto-approved {idx+1}/{len(state['review_ids'])}: {cid}")
            tools.remember_assistant(f"Auto-approved conflict {cid} during iteration 3 run.")
        record_hitl_auto(iteration=3, count=len(state["review_ids"]))
        
        logger.info(f"[Iter3-Runner] 🔄 Re-invoking graph after auto-approval...")
        second_pass_start = time.time()
        out = g.invoke(None, config=config)  # Resume from checkpoint
        second_pass_elapsed = time.time() - second_pass_start
        logger.info(f"[Iter3-Runner] ⏱️  Second pass time: {second_pass_elapsed:.2f} seconds")
    else:
        if state.get('review_ids') and not use_interrupt:
            # Manual review mode without interrupt
            logger.info("")
            logger.info("╔" + "=" * 78 + "╗")
            logger.info("║" + " " * 20 + "PENDING HUMAN REVIEW" + " " * 38 + "║")
            logger.info("╚" + "=" * 78 + "╝")
            logger.info(f"[Iter3-Runner] ⏳ {len(state['review_ids'])} items are pending human review")
            logger.info(f"[Iter3-Runner] 💡 Use queue.label() to review items, then re-run to generate final report")
            open_items = queue.list_open()
            for item in open_items:
                logger.info(f"[Iter3-Runner]   📋 {item['conflict_id']}: conf={item.get('confidence', 0):.2f}, rel={item.get('relation', 'unknown')}")
            tools.remember_assistant("Pending human review without interrupt; conflicts require manual labeling.")
        elif use_interrupt:
            # Interrupt mode - workflow already paused, just log status
            logger.info(f"[Iter3-Runner] ℹ️  Workflow paused. Resume with: g.invoke(None, config=config)")
    
    total_elapsed = time.time() - total_start_time
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 25 + "ITERATION 3 COMPLETED" + " " * 33 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"[Iter3-Runner] ⏱️  Total execution time: {total_elapsed:.2f} seconds")
    logger.info(f"[Iter3-Runner] 📊 Final results: {len(out['conflicts'])} conflicts in report")
    logger.info(f"[Iter3-Runner] 📥 Review queue: {len(state.get('review_ids', []))} items processed")
    logger.info("")
    
    return out["conflicts"], out["report_md"]

def resume_workflow(g, config=None):
    """
    Happy path: resume a previously interrupted Iteration 3 workflow after manual labels are applied.
    
    Args:
        g: The compiled LangGraph instance
        config: Configuration dict with thread_id (defaults to "iter3-run")
    
    Returns:
        Tuple of (conflicts, report_md)
    """
    if config is None:
        config = {"configurable": {"thread_id": "iter3-run"}}
    
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "RESUMING WORKFLOW AFTER REVIEW" + " " * 28 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"[Iter3-Resume] 🔄 Resuming workflow from checkpoint...")
    
    # Resume from checkpoint
    out = g.invoke(None, config=config)
    
    logger.info(f"[Iter3-Resume] ✅ Workflow completed")
    logger.info(f"[Iter3-Resume] 📊 Final conflicts: {len(out.get('conflicts', []))}")
    logger.info("")
    
    return out["conflicts"], out["report_md"]
