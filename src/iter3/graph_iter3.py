import logging
import time
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from shared.schemas import PolicyState, Conflict
from shared.tools import Tools
from shared.reporting import render_markdown
from iter2.graph_iter2 import judge_verifier
from iter3.hitl_queue import queue
from shared.metrics import (
    record_parse,
    record_retrieval,
    record_detection,
    record_report,
    record_hitl_queue,
)

logger = logging.getLogger(__name__)

class HITLState(PolicyState):
    review_ids: List[str]

def node_planner(state: HITLState) -> HITLState:
    """Happy path: log the intended iteration-3 workflow before any processing begins."""
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 0/6: PLANNING WORKFLOW")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Planner] 📋 Planning conflict detection workflow with HITL review")
    logger.info(f"[Iter3-Planner] 📂 Target policy: {state['upload_path']}")
    logger.info(f"[Iter3-Planner] 🎯 Workflow: Parse → Retrieve → Critic → Gate → Review → Report")
    logger.info("=" * 80)
    return state

def node_parse(state: HITLState, tools: Tools) -> HITLState:
    """Happy path: load, chunk, and convert the new policy into clauses while resetting review ids."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 1/6: PARSING POLICY DOCUMENT")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Parse] 📄 Loading policy document from: {state['upload_path']}")
    docs = tools.load(state["upload_path"])
    logger.info(f"[Iter3-Parse] ✓ Loaded {len(docs)} document(s) from PDF")
    logger.info(f"[Iter3-Parse] 🔪 Chunking documents into smaller pieces...")
    chunks = tools.chunk(docs)
    logger.info(f"[Iter3-Parse] ✓ Split into {len(chunks)} text chunks")
    logger.info(f"[Iter3-Parse] 📝 Converting chunks to policy clauses...")
    state["new_policy_clauses"] = tools.to_clauses(chunks, src=state["upload_path"].split("/")[-1])
    logger.info(f"[Iter3-Parse] ✓ Created {len(state['new_policy_clauses'])} policy clauses")
    state["review_ids"] = []
    elapsed = time.time() - start_time
    logger.info(f"[Iter3-Parse] ⏱️  Parsing completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    record_parse(iteration=3, clauses=len(state["new_policy_clauses"]), duration=elapsed)
    return state

def node_retrieve(state: HITLState, tools: Tools) -> HITLState:
    """Happy path: reuse Iteration 1 retrieval to gather candidates for each new clause."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 2/6: RETRIEVING CANDIDATE MATCHES")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Retrieve] 🔍 Retrieving candidate matches for {len(state['new_policy_clauses'])} clauses")
    from iter1.graph_iter1 import node_retrieve as base
    result = base(state, tools)
    total_candidates = sum(len(v) for v in result["candidate_matches"].values())
    elapsed = time.time() - start_time
    logger.info(f"[Iter3-Retrieve] ✓ Retrieved {total_candidates} total candidate matches")
    logger.info(f"[Iter3-Retrieve] ⏱️  Retrieval completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    record_retrieval(
        iteration=3,
        total_candidates=total_candidates,
        clauses=len(state["new_policy_clauses"]),
        duration=elapsed,
    )
    return result

def node_critic(state: HITLState, tools: Tools) -> HITLState:
    """Happy path: run base detection then adjust conflict confidence with the critic verifier."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 3/6: CRITIC VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Critic] 🧠 Starting conflict detection and critic verification")
    logger.info(f"[Iter3-Critic] 📊 Phase 1: Base NLI detection...")
    from iter1.graph_iter1 import node_detect as base_detect
    s1 = base_detect(state, tools)
    base_conflicts = len(s1['conflicts'])
    logger.info(f"[Iter3-Critic] ✓ Base detection found {base_conflicts} conflicts")
    logger.info(f"[Iter3-Critic] 📊 Phase 2: Critic adjustments...")
    logger.info(f"[Iter3-Critic] 🎯 Applying critic adjustments to {base_conflicts} conflicts")
    
    adjusted: List[Conflict] = []
    skipped = 0
    adjustments_made = 0
    comparisons = sum(
        len(state["candidate_matches"].get(cl.id, [])) for cl in state["new_policy_clauses"]
    )
    
    for idx, cf in enumerate(s1["conflicts"]):
        a = next((c for c in state["new_policy_clauses"] if c.id == cf.new_id), None)
        b = None
        for v in state["candidate_matches"].values():
            for c in v:
                if c.id == cf.existing_id: b = c; break
        if not a or not b: 
            logger.warning(f"[Iter3-Critic] ⚠️  Skipping conflict {cf.new_id} vs {cf.existing_id}: clause not found")
            skipped += 1
            continue
        
        logger.info(f"[Iter3-Critic] 🔍 Critic verifying conflict {idx+1}/{base_conflicts}: {cf.relation} (conf={cf.confidence:.2f})")
        ver = judge_verifier(tools, a.text, b.text, cf)
        adj = cf.copy(update={"confidence": float(ver.get("adjusted_confidence", cf.confidence))})
        
        if abs(adj.confidence - cf.confidence) > 0.01:
            adjustments_made += 1
            logger.info(f"[Iter3-Critic]   ↻ Adjusted confidence: {cf.confidence:.2f} → {adj.confidence:.2f}")
        else:
            logger.debug(f"[Iter3-Critic]   → Confidence unchanged: {adj.confidence:.2f}")
        
        adjusted.append(adj)
    
    s1["conflicts"] = adjusted
    elapsed = time.time() - start_time
    logger.info(f"[Iter3-Critic] ✓ Critic verification complete")
    logger.info(f"[Iter3-Critic] 📊 Statistics:")
    logger.info(f"[Iter3-Critic]   • Base conflicts: {base_conflicts}")
    logger.info(f"[Iter3-Critic]   • After adjustment: {len(adjusted)}")
    logger.info(f"[Iter3-Critic]   • Confidence adjustments: {adjustments_made}")
    logger.info(f"[Iter3-Critic]   • Skipped: {skipped}")
    logger.info(f"[Iter3-Critic] ⏱️  Critic verification completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    contradictions = sum(1 for c in adjusted if c.relation == "contradiction")
    duplications = sum(1 for c in adjusted if c.relation == "duplication")
    possible_conflicts = sum(1 for c in adjusted if c.relation == "possible-conflict")
    record_detection(
        iteration=3,
        comparisons=comparisons,
        contradictions=contradictions,
        duplications=duplications,
        possible_conflicts=possible_conflicts,
        kept=len(adjusted),
        duration=elapsed,
    )
    return s1

def node_gate_enqueue(state: HITLState) -> HITLState:
    """Happy path: route conflicts into auto decisions or the HITL review queue based on confidence."""
    iteration = int(state.get("metrics_iteration", 3))
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 4/6: GATING AND ENQUEUING FOR HITL REVIEW")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Gate] 🚪 Evaluating {len(state['conflicts'])} conflicts for human-in-the-loop review")
    logger.info(f"[Iter3-Gate] 📋 Review criteria: confidence in range [0.40, 0.70)")
    
    ids = []
    enqueued = 0
    auto_approved = 0
    auto_rejected = 0
    
    for cf in state["conflicts"]:
        cid = f"{cf.new_id}|{cf.existing_id}"
        if 0.40 <= cf.confidence < 0.70:
            queue.enqueue({"conflict_id": cid, "relation": cf.relation, "confidence": cf.confidence, "status": "open"})
            ids.append(cid)
            enqueued += 1
            logger.info(f"[Iter3-Gate]   📥 ENQUEUED: {cid} (conf={cf.confidence:.2f}, rel={cf.relation})")
        elif cf.confidence >= 0.70:
            auto_approved += 1
            logger.info(f"[Iter3-Gate]   ✅ AUTO-APPROVED: {cid} (conf={cf.confidence:.2f} >= 0.70)")
        else:
            auto_rejected += 1
            logger.info(f"[Iter3-Gate]   ❌ AUTO-REJECTED: {cid} (conf={cf.confidence:.2f} < 0.40)")
    
    state["review_ids"] = ids
    elapsed = time.time() - start_time
    logger.info(f"[Iter3-Gate] ✓ Gating complete")
    logger.info(f"[Iter3-Gate] 📊 Statistics:")
    logger.info(f"[Iter3-Gate]   • Enqueued for review: {enqueued}")
    logger.info(f"[Iter3-Gate]   • Auto-approved (high conf): {auto_approved}")
    logger.info(f"[Iter3-Gate]   • Auto-rejected (low conf): {auto_rejected}")
    logger.info(f"[Iter3-Gate]   • Total conflicts processed: {len(state['conflicts'])}")
    logger.info(f"[Iter3-Gate] ⏱️  Gating completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    record_hitl_queue(iteration, enqueued, auto_approved, auto_rejected)
    return state

def node_apply_labels_and_report(state: HITLState) -> HITLState:
    """Happy path: apply any human labels, keep approved conflicts, and build the final report."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 5/6: APPLYING HUMAN REVIEW LABELS")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Apply-Labels] 👤 Applying human review labels to conflicts")
    logger.info(f"[Iter3-Apply-Labels] 📋 Checking {len(state['review_ids'])} items in review queue")
    
    kept = []
    rejected = 0
    pending = 0
    approved = 0
    
    for cf in state["conflicts"]:
        cid = f"{cf.new_id}|{cf.existing_id}"
        item = queue.items.get(cid)
        
        if cid in state["review_ids"]:
            # This was enqueued for review
            if item and item.get("status") == "labeled":
                label = item.get("label")
                if label in ("reject", "needs_context"):
                    rejected += 1
                    logger.info(f"[Iter3-Apply-Labels]   ❌ REJECTED by human: {cid} (label={label})")
                elif label == "approve":
                    approved += 1
                    kept.append(cf)
                    logger.info(f"[Iter3-Apply-Labels]   ✅ APPROVED by human: {cid}")
                else:
                    kept.append(cf)
                    logger.info(f"[Iter3-Apply-Labels]   ✓ Kept: {cid} (label={label})")
            else:
                pending += 1
                kept.append(cf)
                logger.warning(f"[Iter3-Apply-Labels]   ⏳ PENDING review: {cid} (no label yet, keeping)")
        else:
            # Not enqueued (auto-approved or auto-rejected earlier)
            if cf.confidence >= 0.70:
                kept.append(cf)
                logger.debug(f"[Iter3-Apply-Labels]   ✓ Auto-kept (high conf): {cid}")
            else:
                logger.debug(f"[Iter3-Apply-Labels]   ✗ Auto-rejected (low conf): {cid}")
    
    state["conflicts"] = kept
    elapsed_labels = time.time() - start_time
    logger.info(f"[Iter3-Apply-Labels] ✓ Labels applied")
    logger.info(f"[Iter3-Apply-Labels] 📊 Review statistics:")
    logger.info(f"[Iter3-Apply-Labels]   • Human approved: {approved}")
    logger.info(f"[Iter3-Apply-Labels]   • Human rejected: {rejected}")
    logger.info(f"[Iter3-Apply-Labels]   • Pending review: {pending}")
    logger.info(f"[Iter3-Apply-Labels]   • Final conflicts kept: {len(kept)}")
    logger.info(f"[Iter3-Apply-Labels] ⏱️  Label application completed in {elapsed_labels:.2f} seconds")
    logger.info("=" * 80)
    
    # Generate report
    report_start = time.time()
    logger.info("=" * 80)
    logger.info("[ITER3] STEP 6/6: GENERATING REPORT")
    logger.info("=" * 80)
    logger.info(f"[Iter3-Report] 📝 Generating markdown report for {len(kept)} final conflicts")
    state["report_md"] = render_markdown(state)
    report_elapsed = time.time() - report_start
    logger.info(f"[Iter3-Report] ✓ Report generated successfully")
    logger.info(f"[Iter3-Report] 📏 Report size: {len(state['report_md'])} characters")
    logger.info(f"[Iter3-Report] ⏱️  Report generation completed in {report_elapsed:.2f} seconds")
    logger.info("=" * 80)
    logger.info("[ITER3] ✅ ALL STEPS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    record_report(
        iteration=3,
        conflicts=len(kept),
        report_length=len(state["report_md"]),
        duration=report_elapsed,
    )
    return state

def build_graph(tools: Tools, use_interrupt: bool = True):
    """
    Happy path: compile the Iteration 3 graph, optionally injecting an interrupt for human approval.
    
    Args:
        tools: Tools instance for document processing
        use_interrupt: If True, adds interrupt before report node for human approval
    """
    g = StateGraph(HITLState)
    g.add_node("plan", node_planner)
    g.add_node("parse", lambda s: node_parse(s, tools))
    g.add_node("retrieve", lambda s: node_retrieve(s, tools))
    g.add_node("critic", lambda s: node_critic(s, tools))
    g.add_node("gate", node_gate_enqueue)
    g.add_node("report", node_apply_labels_and_report)
    g.add_edge(START,"plan"); g.add_edge("plan","parse"); g.add_edge("parse","retrieve")
    g.add_edge("retrieve","critic"); g.add_edge("critic","gate"); g.add_edge("gate","report")
    g.add_edge("report",END)
    
    # Configure checkpoints for state persistence and interrupts
    checkpointer = MemorySaver()
    
    if use_interrupt:
        # Add interrupt BEFORE report node to pause for human review
        logger.info("[Iter3-Graph] 🔔 Interrupt enabled: workflow will pause before report for human approval")
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=["report"]  # Pause here to wait for human review
        )
    else:
        logger.info("[Iter3-Graph] ⚡ Interrupt disabled: workflow will run continuously")
        return g.compile(checkpointer=checkpointer)
