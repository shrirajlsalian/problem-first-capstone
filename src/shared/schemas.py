from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass
class Clause:
    id: str
    text: str
    source_id: str
    meta: Dict[str, Any]

class Conflict(BaseModel):
    new_id: str
    existing_id: str
    relation: str = Field(pattern="^(contradiction|duplication|possible-conflict)$")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

class NLIVerdict(BaseModel):
    relation: str = Field(pattern="^(contradiction|duplication|entailment|unrelated)$")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

class PolicyState(TypedDict):
    upload_path: Optional[str]
    new_policy_clauses: List[Clause]
    candidate_matches: Dict[str, List[Clause]]
    conflicts: List[Conflict]
    report_md: str
