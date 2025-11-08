import json
import logging
import time
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from shared.schemas import PolicyState, Conflict, Clause
from shared.tools import Tools
from shared.reporting import render_markdown
from shared.metrics import (
    record_parse,
    record_retrieval,
    record_detection,
    record_report,
    record_judge_summary,
)

logger = logging.getLogger(__name__)

class AgenticState(PolicyState):
    traces: List[Dict[str, Any]]  # optional

def judge_verifier(tools: Tools, a: str, b: str, cf: Conflict) -> Dict[str, Any]:
    """Happy path: ask the LLM judge to confirm a conflict verdict and adjust confidence accordingly."""
    logger.debug(f"[Iter2-Judge] 🔍 Verifying conflict: {cf.relation} (conf={cf.confidence:.2f})")
    prompt = (
        "Verifier: decide if the verdict is supported by text spans.\n"
        "Return JSON {\"verified\": true|false, \"adjusted_confidence\": 0..1, \"notes\":\"...\"}\n\n"
        f"A: {a}\nB: {b}\nVerdict: {cf.model_dump_json()}\n\nJSON:"
    )
    judge_start = time.time()
    out = tools.call_llm(prompt).content
    judge_elapsed = time.time() - judge_start
    try: 
        result = json.loads(out)
        verified = result.get('verified', False)
        adj_conf = result.get('adjusted_confidence', cf.confidence)
        logger.info(f"[Iter2-Judge] ✓ Judge verification ({judge_elapsed:.2f}s): verified={verified}, adj_conf={adj_conf:.2f} (was {cf.confidence:.2f})")
        if result.get('notes'):
            logger.debug(f"[Iter2-Judge]   Notes: {result.get('notes')[:100]}")
        return result
    except Exception as e: 
        logger.warning(f"[Iter2-Judge] ⚠️  Failed to parse judge response: {e}, using fallback")
        fallback_conf = max(0.0, cf.confidence-0.15)
        logger.info(f"[Iter2-Judge]   Fallback: verified=False, adj_conf={fallback_conf:.2f}")
        return {"verified": False, "adjusted_confidence": fallback_conf, "notes": "parse error"}

def node_parse(state: AgenticState, tools: Tools) -> AgenticState:
    """Happy path: load the uploaded policy, chunk it, and stash fresh clauses plus a trace list."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER2] STEP 1/4: PARSING POLICY DOCUMENT")
    logger.info("=" * 80)
    logger.info(f"[Iter2-Parse] 📄 Loading policy document from: {state['upload_path']}")
    docs = tools.load(state["upload_path"])
    logger.info(f"[Iter2-Parse] ✓ Loaded {len(docs)} document(s) from PDF")
    logger.info(f"[Iter2-Parse] 🔪 Chunking documents into smaller pieces...")
    chunks = tools.chunk(docs)
    logger.info(f"[Iter2-Parse] ✓ Split into {len(chunks)} text chunks")
    logger.info(f"[Iter2-Parse] 📝 Converting chunks to policy clauses...")
    state["new_policy_clauses"] = tools.to_clauses(chunks, src=state["upload_path"].split("/")[-1])
    logger.info(f"[Iter2-Parse] ✓ Created {len(state['new_policy_clauses'])} policy clauses")
    state.setdefault("traces", [])
    elapsed = time.time() - start_time
    logger.info(f"[Iter2-Parse] ⏱️  Parsing completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    record_parse(iteration=2, clauses=len(state["new_policy_clauses"]), duration=elapsed)
    return state

def node_retrieve_adaptive(state: AgenticState, tools: Tools) -> AgenticState:
    """Happy path: reuse Iteration 1 retrieval to gather candidate matches for judge-enhanced detection."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER2] STEP 2/4: RETRIEVING CANDIDATE MATCHES")
    logger.info("=" * 80)
    logger.info(f"[Iter2-Retrieve] 🔍 Retrieving candidate matches (using base retrieval method)")
    from iter1.graph_iter1 import node_retrieve as base_retrieve
    result = base_retrieve(state, tools)
    total_candidates = sum(len(v) for v in result["candidate_matches"].values())
    elapsed = time.time() - start_time
    logger.info(f"[Iter2-Retrieve] ✓ Retrieved {total_candidates} total candidate matches")
    logger.info(f"[Iter2-Retrieve] ⏱️  Retrieval completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    record_retrieval(
        iteration=2,
        total_candidates=total_candidates,
        clauses=len(state["new_policy_clauses"]),
        duration=elapsed,
    )
    return result

def node_detect_with_judge(state: AgenticState, tools: Tools) -> AgenticState:
    """Happy path: run base detection then verify each conflict with an LLM judge, keeping approved ones."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER2] STEP 3/4: DETECTING CONFLICTS WITH JUDGE VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"[Iter2-Detect-Judge] 🧠 Starting conflict detection with judge verification")
    logger.info(f"[Iter2-Detect-Judge] 📊 Phase 1: Base NLI detection...")
    from iter1.graph_iter1 import node_detect as base_detect
    s1 = base_detect(state, tools)
    base_conflicts = len(s1['conflicts'])
    logger.info(f"[Iter2-Detect-Judge] ✓ Base detection found {base_conflicts} conflicts")
    logger.info(f"[Iter2-Detect-Judge] 📊 Phase 2: Judge verification...")
    logger.info(f"[Iter2-Detect-Judge] 🎯 Verifying {base_conflicts} conflicts with LLM judge")
    
    comparisons = sum(
        len(state["candidate_matches"].get(cl.id, [])) for cl in state["new_policy_clauses"]
    )

    kept: List[Conflict] = []
    rejected = 0
    verified_count = 0
    confidence_adjusted = 0
    
    for idx, cf in enumerate(s1["conflicts"]):
        logger.info(f"[Iter2-Detect-Judge] 🔍 Verifying conflict {idx+1}/{base_conflicts}: {cf.relation} (conf={cf.confidence:.2f})")
        a = next((c for c in state["new_policy_clauses"] if c.id == cf.new_id), None)
        b = None
        for v in state["candidate_matches"].values():
            for c in v:
                if c.id == cf.existing_id: b = c; break
        if not a or not b: 
            logger.warning(f"[Iter2-Detect-Judge] ⚠️  Skipping conflict {cf.new_id} vs {cf.existing_id}: clause not found")
            rejected += 1
            continue
        
        j = judge_verifier(tools, a.text, b.text, cf)
        new_conf = cf.copy(update={"confidence": float(j.get("adjusted_confidence", cf.confidence))})
        
        if abs(new_conf.confidence - cf.confidence) > 0.01:
            confidence_adjusted += 1
        
        if not j.get("verified", False) and new_conf.confidence < 0.55:
            logger.info(f"[Iter2-Detect-Judge] ❌ Rejecting conflict: not verified and conf {new_conf.confidence:.2f} < 0.55")
            rejected += 1
            continue
        
        kept.append(new_conf)
        verified_count += 1
        state["traces"].append({"judge": j, "kept": new_conf.model_dump()})
        logger.info(f"[Iter2-Detect-Judge] ✓ Kept conflict: {cf.relation} (conf={new_conf.confidence:.2f})")
    
    s1["conflicts"] = kept
    elapsed = time.time() - start_time
    logger.info(f"[Iter2-Detect-Judge] ✓ Judge verification complete")
    logger.info(f"[Iter2-Detect-Judge] 📊 Statistics:")
    logger.info(f"[Iter2-Detect-Judge]   • Base conflicts found: {base_conflicts}")
    logger.info(f"[Iter2-Detect-Judge]   • Verified and kept: {verified_count}")
    logger.info(f"[Iter2-Detect-Judge]   • Rejected: {rejected}")
    logger.info(f"[Iter2-Detect-Judge]   • Confidence adjustments: {confidence_adjusted}")
    logger.info(f"[Iter2-Detect-Judge]   • Final conflicts: {len(kept)}")
    logger.info(f"[Iter2-Detect-Judge] ⏱️  Detection and verification completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    contradictions = sum(1 for c in kept if c.relation == "contradiction")
    duplications = sum(1 for c in kept if c.relation == "duplication")
    possible_conflicts_kept = sum(1 for c in kept if c.relation == "possible-conflict")
    record_detection(
        iteration=2,
        comparisons=comparisons,
        contradictions=contradictions,
        duplications=duplications,
        possible_conflicts=possible_conflicts_kept,
        kept=len(kept),
        duration=elapsed,
    )
    record_judge_summary(
        iteration=2,
        evaluated=base_conflicts,
        verified=verified_count,
        rejected=rejected,
        confidence_adjusted=confidence_adjusted,
        duration=elapsed,
    )
    return s1

def node_report(state: AgenticState) -> AgenticState:
    """Happy path: render verified conflicts into markdown while logging review statistics."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("[ITER2] STEP 4/4: GENERATING REPORT")
    logger.info("=" * 80)
    logger.info(f"[Iter2-Report] 📝 Generating markdown report for {len(state['conflicts'])} verified conflicts")
    state["report_md"] = render_markdown(state)
    elapsed = time.time() - start_time
    logger.info(f"[Iter2-Report] ✓ Report generated successfully")
    logger.info(f"[Iter2-Report] 📏 Report size: {len(state['report_md'])} characters")
    logger.info(f"[Iter2-Report] ⏱️  Report generation completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    logger.info("[ITER2] ✅ ALL STEPS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    record_report(
        iteration=2,
        conflicts=len(state["conflicts"]),
        report_length=len(state["report_md"]),
        duration=elapsed,
    )
    return state

def build_graph(tools: Tools):
    """Happy path: compile the Iteration 2 LangGraph including judge verification stage."""
    g = StateGraph(AgenticState)
    g.add_node("parse",      lambda s: node_parse(s, tools))
    g.add_node("retrieve",   lambda s: node_retrieve_adaptive(s, tools))
    g.add_node("detect_jud", lambda s: node_detect_with_judge(s, tools))
    g.add_node("report",      node_report)
    g.add_edge(START,"parse"); g.add_edge("parse","retrieve")
    g.add_edge("retrieve","detect_jud"); g.add_edge("detect_jud","report")
    g.add_edge("report",END)
    return g.compile()
