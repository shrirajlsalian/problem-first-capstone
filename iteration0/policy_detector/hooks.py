from __future__ import annotations

from typing import List
from .types import Clause, Conflict


class Hooks:
    """Lifecycle hooks to enable HITL, logging, or custom side-effects.
    Override any of these in the future without changing pipeline code."""

    def on_upload_parsed(self, clauses: List[Clause]) -> None:
        pass

    def on_candidates_retrieved(self, new_clause: Clause, candidates: List[Clause]) -> None:
        pass

    def on_conflict_detected(self, conflict: Conflict) -> None:
        pass

    def on_report_built(self, report_md: str) -> None:
        pass


