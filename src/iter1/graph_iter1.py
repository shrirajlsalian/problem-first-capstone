import logging
import time
from langgraph.graph import StateGraph, START, END
from typing import List
from shared.schemas import PolicyState, Conflict, Clause
from shared.tools import Tools
from shared.reporting import render_markdown
from shared.metrics import record_parse, record_retrieval, record_detection, record_report

#logger = logging.getLogger(__name__)

def node_parse(state: PolicyState, tools: Tools) -> PolicyState:
    """Happy path: load the uploaded policy, chunk it, and populate new clauses on the state."""
    start_time = time.time()
    #logger.info("=" * 80)
    #logger.info("[ITER1] STEP 1/4: PARSING POLICY DOCUMENT")
    #logger.info("=" * 80)
    #logger.info(f"[Iter1-Parse] 📄 Loading policy document from: {state['upload_path']}")
    docs = tools.load(state["upload_path"])
    #logger.info(f"[Iter1-Parse] ✓ Loaded {len(docs)} document(s) from PDF")
    #logger.info(f"[Iter1-Parse] 🔪 Chunking documents into smaller pieces...")
    chunks = tools.chunk(docs)
    #logger.info(f"[Iter1-Parse] ✓ Split into {len(chunks)} text chunks")
    #logger.info(f"[Iter1-Parse] 📝 Converting chunks to policy clauses...")
    state["new_policy_clauses"] = tools.to_clauses(chunks, src=state["upload_path"].split("/")[-1])
    #logger.info(f"[Iter1-Parse] ✓ Created {len(state['new_policy_clauses'])} policy clauses")
    elapsed = time.time() - start_time
    #logger.info(f"[Iter1-Parse] ⏱️  Parsing completed in {elapsed:.2f} seconds")
    #logger.info("=" * 80)
    record_parse(iteration=1, clauses=len(state["new_policy_clauses"]), duration=elapsed)
    return state

def node_retrieve(state: PolicyState, tools: Tools) -> PolicyState:
    """Happy path: retrieve a deduplicated set of candidate matches for each newly parsed clause."""
    start_time = time.time()
    #logger.info("=" * 80)
    #logger.info("[ITER1] STEP 2/4: RETRIEVING CANDIDATE MATCHES")
    #logger.info("=" * 80)
    #logger.info(f"[Iter1-Retrieve] 🔍 Searching vector store for matches to {len(state['new_policy_clauses'])} clauses")
    mapping = {}
    total_clauses = len(state["new_policy_clauses"])
    for i, cl in enumerate(state["new_policy_clauses"]):
        #logger.info(f"[Iter1-Retrieve] 📋 Processing clause {i+1}/{total_clauses}: {cl.id[:50]}...")
        cands = tools.retrieve(cl.text, k=12)
        #logger.debug(f"[Iter1-Retrieve]   Found {len(cands)} raw candidates from vector store")
        uniq, seen = [], set()
        for d in cands:
            key = (d.page_content[:60], str(d.metadata))
            if key in seen: continue
            seen.add(key)
            uniq.append(Clause(
                id=str(d.metadata.get("source", d.metadata.get("source_id","unknown"))) + ":" + str(d.metadata.get("page", d.metadata.get("chunk", 0))),
                text=d.page_content, source_id=str(d.metadata.get("source", d.metadata.get("source_id","unknown"))),
                meta=d.metadata or {}
            ))
        mapping[cl.id] = uniq[:12]
        #logger.info(f"[Iter1-Retrieve]   ✓ Stored {len(mapping[cl.id])} unique candidate matches for clause {i+1}")
    state["candidate_matches"] = mapping
    total_candidates = sum(len(v) for v in mapping.values())
    elapsed = time.time() - start_time
    #logger.info(f"[Iter1-Retrieve] ✓ Retrieved {total_candidates} total candidate matches across {total_clauses} clauses")
    #logger.info(f"[Iter1-Retrieve] ⏱️  Retrieval completed in {elapsed:.2f} seconds")
    #logger.info("=" * 80)
    record_retrieval(iteration=1, total_candidates=total_candidates, clauses=total_clauses, duration=elapsed)
    return state

def node_detect(state: PolicyState, tools: Tools) -> PolicyState:
    """Happy path: run NLI comparisons to classify conflicts between new clauses and retrieved candidates."""
    start_time = time.time()
    #logger.info("=" * 80)
    #logger.info("[ITER1] STEP 3/4: DETECTING CONFLICTS USING NLI")
    #logger.info("=" * 80)
    #logger.info(f"[Iter1-Detect] 🧠 Starting conflict detection using Natural Language Inference")
    #logger.info(f"[Iter1-Detect] 📊 Comparing {len(state['new_policy_clauses'])} new clauses with candidate matches")
    out: List[Conflict] = []
    comparisons = 0
    contradictions = 0
    duplications = 0
    possible_conflicts = 0
    total_pairs = sum(len(state["candidate_matches"].get(cl.id, [])) for cl in state["new_policy_clauses"])
    #logger.info(f"[Iter1-Detect] 📈 Total comparison pairs: {total_pairs}")
    
    pair_count = 0
    for cl_idx, cl in enumerate(state["new_policy_clauses"]):
        candidates = state["candidate_matches"].get(cl.id, [])
        if candidates:
            #logger.info(f"[Iter1-Detect] 🔍 Analyzing clause {cl_idx+1}/{len(state['new_policy_clauses'])} ({len(candidates)} candidates)")
        for cand in candidates:
            comparisons += 1
            pair_count += 1
            if pair_count % 5 == 0 or pair_count == total_pairs:
                #logger.info(f"[Iter1-Detect]   Progress: {pair_count}/{total_pairs} comparisons completed...")
            
            v = tools.nli_compare(cl.text, cand.text)
            rel, conf, rat = v["relation"], v["confidence"], v["rationale"]
            
            if rel in ("contradiction","duplication") and conf >= 0.55:
                out.append(Conflict(new_id=cl.id, existing_id=cand.id, relation=rel, confidence=conf, rationale=rat))
                if rel == "contradiction":
                    contradictions += 1
                else:
                    duplications += 1
                #logger.info(f"[Iter1-Detect]   ⚠️  CONFLICT FOUND: {rel.upper()} (conf={conf:.2f})")
                #logger.debug(f"[Iter1-Detect]      New: {cl.id} | Existing: {cand.id}")
            elif conf >= 0.40 and rel not in ("unrelated","entailment"):
                out.append(Conflict(new_id=cl.id, existing_id=cand.id, relation="possible-conflict", confidence=conf, rationale=rat))
                possible_conflicts += 1
                #logger.info(f"[Iter1-Detect]   ⚠️  POSSIBLE CONFLICT (conf={conf:.2f})")
                #logger.debug(f"[Iter1-Detect]      New: {cl.id} | Existing: {cand.id}")
    
    state["conflicts"] = out
    elapsed = time.time() - start_time
    #logger.info(f"[Iter1-Detect] ✓ Conflict detection completed")
    #logger.info(f"[Iter1-Detect] 📊 Statistics:")
    #logger.info(f"[Iter1-Detect]   • Total NLI comparisons: {comparisons}")
    #logger.info(f"[Iter1-Detect]   • Contradictions found: {contradictions}")
    #logger.info(f"[Iter1-Detect]   • Duplications found: {duplications}")
    #logger.info(f"[Iter1-Detect]   • Possible conflicts: {possible_conflicts}")
    #logger.info(f"[Iter1-Detect]   • Total conflicts detected: {len(out)}")
    #logger.info(f"[Iter1-Detect] ⏱️  Detection completed in {elapsed:.2f} seconds")
    #logger.info("=" * 80)
    record_detection(
        iteration=1,
        comparisons=comparisons,
        contradictions=contradictions,
        duplications=duplications,
        possible_conflicts=possible_conflicts,
        kept=len(out),
        duration=elapsed,
    )
    return state

def node_report(state: PolicyState) -> PolicyState:
    """Happy path: render the detected conflicts into a markdown report stored on the state."""
    start_time = time.time()
    #logger.info("=" * 80)
    #logger.info("[ITER1] STEP 4/4: GENERATING REPORT")
    #logger.info("=" * 80)
    #logger.info(f"[Iter1-Report] 📝 Generating markdown report for {len(state['conflicts'])} conflicts")
    state["report_md"] = render_markdown(state)
    elapsed = time.time() - start_time
    #logger.info(f"[Iter1-Report] ✓ Report generated successfully")
    #logger.info(f"[Iter1-Report] 📏 Report size: {len(state['report_md'])} characters")
    #logger.info(f"[Iter1-Report] ⏱️  Report generation completed in {elapsed:.2f} seconds")
    #logger.info("=" * 80)
    #logger.info("[ITER1] ✅ ALL STEPS COMPLETED SUCCESSFULLY")
    #logger.info("=" * 80)
    record_report(
        iteration=1,
        conflicts=len(state["conflicts"]),
        report_length=len(state["report_md"]),
        duration=elapsed,
    )
    return state

def build_graph(tools: Tools):
    """Happy path: assemble and compile the Iteration 1 LangGraph pipeline."""
    g = StateGraph(PolicyState)
    g.add_node("parse",   lambda s: node_parse(s, tools))
    g.add_node("retrieve",lambda s: node_retrieve(s, tools))
    g.add_node("detect",  lambda s: node_detect(s, tools))
    g.add_node("report",   node_report)
    g.add_edge(START,"parse"); g.add_edge("parse","retrieve")
    g.add_edge("retrieve","detect"); g.add_edge("detect","report")
    g.add_edge("report",END)
    return g.compile()
