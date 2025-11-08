import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    # Optional dependency, referenced throughout the assignments
    # (see Assignment 2 "[Bonus] Using Comet Opik" notebook).
    from opik import Opik
except ImportError:  # pragma: no cover - environment without Opik installed
    Opik = None  # type: ignore[assignment]


class OpikMetrics:
    """Minimal Opik metrics façade.

    Mirrors the lightweight pattern used in the assignments while gracefully
    falling back to simple logging when the Opik SDK is unavailable.
    """

    def __init__(self, client: Optional["Opik"] = None) -> None:  # type: ignore[name-defined]
        self._client = client or self._maybe_create_client()

    @staticmethod
    def _maybe_create_client() -> Optional["Opik"]:  # type: ignore[name-defined]
        if Opik is None:
            logger.debug("Opik SDK not installed; metrics will be logged locally.")
            return None
        try:
            return Opik()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to initialize Opik client; metrics will be logged locally.", exc_info=exc)
            return None

    def counter(self, name: str, value: float = 1.0, **tags: Any) -> None:
        self._record_metric("counter", name, value, tags)

    def gauge(self, name: str, value: float, **tags: Any) -> None:
        self._record_metric("gauge", name, value, tags)

    def _record_metric(self, metric_type: str, name: str, value: float, tags: Dict[str, Any]) -> None:
        metadata = {"type": metric_type, **tags}
        if self._client is not None:
            try:
                log_metric = getattr(self._client, "log_metric", None)
                if callable(log_metric):
                    log_metric(name=name, value=value, metadata=metadata)
                    return

                log_metrics = getattr(self._client, "log_metrics", None)
                if callable(log_metrics):
                    log_metrics(metrics={name: value}, metadata=metadata)
                    return

                logger.debug("Opik client does not expose log_metric(s); falling back to logging.")
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Unable to send metric '%s' to Opik; falling back to logging.", name, exc_info=exc)

        logger.info("[OPIK METRIC] %s %s value=%s %s", metric_type, name, value, metadata or {})


opik = OpikMetrics()


def record_parse(iteration: int, clauses: int, duration: float) -> None:
    opik.counter("parse.clauses", value=clauses, iteration=iteration)
    opik.gauge("parse.duration", duration, iteration=iteration)


def record_retrieval(iteration: int, total_candidates: int, clauses: int, duration: float) -> None:
    opik.counter("retrieval.candidates", value=total_candidates, iteration=iteration)
    if clauses:
        opik.gauge("retrieval.candidates_per_clause", total_candidates / clauses, iteration=iteration)
    opik.gauge("retrieval.duration", duration, iteration=iteration)


def record_detection(
    iteration: int,
    comparisons: int,
    contradictions: int,
    duplications: int,
    possible_conflicts: int,
    kept: int,
    duration: float,
) -> None:
    opik.counter("detection.comparisons", value=comparisons, iteration=iteration)
    if contradictions:
        opik.counter("detection.contradictions", value=contradictions, iteration=iteration)
    if duplications:
        opik.counter("detection.duplications", value=duplications, iteration=iteration)
    if possible_conflicts:
        opik.counter("detection.possible_conflicts", value=possible_conflicts, iteration=iteration)
    opik.gauge("detection.kept_conflicts", kept, iteration=iteration)
    opik.gauge("detection.duration", duration, iteration=iteration)


def record_report(iteration: int, conflicts: int, report_length: int, duration: float) -> None:
    opik.gauge("report.conflict_count", conflicts, iteration=iteration)
    opik.gauge("report.length_chars", report_length, iteration=iteration)
    opik.gauge("report.duration", duration, iteration=iteration)


def record_judge_summary(
    iteration: int,
    evaluated: int,
    verified: int,
    rejected: int,
    confidence_adjusted: int,
    duration: float,
) -> None:
    opik.counter("judge.evaluated", value=evaluated, iteration=iteration)
    opik.counter("judge.verified", value=verified, iteration=iteration)
    if rejected:
        opik.counter("judge.rejected", value=rejected, iteration=iteration)
    if confidence_adjusted:
        opik.counter("judge.confidence_adjustments", value=confidence_adjusted, iteration=iteration)
    opik.gauge("judge.duration", duration, iteration=iteration)


def record_hitl_queue(iteration: int, enqueued: int, auto_approved: int, auto_rejected: int) -> None:
    opik.counter("hitl.enqueued", value=enqueued, iteration=iteration)
    if auto_approved:
        opik.counter("hitl.auto_approved", value=auto_approved, iteration=iteration)
    if auto_rejected:
        opik.counter("hitl.auto_rejected", value=auto_rejected, iteration=iteration)


def record_hitl_cli(iteration: int, approved: int, rejected: int) -> None:
    opik.counter("hitl.cli_approved", value=approved, iteration=iteration)
    opik.counter("hitl.cli_rejected", value=rejected, iteration=iteration)


def record_hitl_auto(iteration: int, count: int) -> None:
    opik.counter("hitl.auto_approve_batch", value=count, iteration=iteration)


def record_hitl_mcp(iteration: int, count: int) -> None:
    opik.counter("hitl.mcp_dispatched", value=count, iteration=iteration)


def record_planner_strategy(strategy: str) -> None:
    opik.counter("iter4.planner.strategy", value=1, strategy=strategy)


def record_web_snippets(iteration: int, snippet_count: int) -> None:
    if snippet_count:
        opik.counter("remediation.web_snippets", value=snippet_count, iteration=iteration)
