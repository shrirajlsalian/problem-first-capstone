from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypedDict


@dataclass
class Clause:
    id: str
    text: str
    source_id: str  # doc id or path
    meta: Dict[str, Any]


@dataclass
class Conflict:
    new_clause_id: str
    existing_clause_id: str
    relation: str  # "contradiction" | "duplication" | "policy-gap" | "possible-conflict"
    confidence: float
    rationale: str


class PolicyState(TypedDict):
    upload_path: Optional[str]
    new_policy_clauses: List[Clause]
    candidate_matches: Dict[str, List[Clause]]  # new_clause_id -> candidates
    conflicts: List[Conflict]
    report_md: str


