"""
Minimal MCP-style server exposing endpoints for the Iteration 4 review queue.

Run with:
    uvicorn iter4.mcp_server:app --reload
"""

from typing import List

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "FastAPI is required for the MCP server. Install with `pip install fastapi uvicorn`."
    ) from exc

from iter3.hitl_queue import queue


class LabelRequest(BaseModel):
    conflict_id: str
    label: str
    notes: str | None = None


class BulkQueueRequest(BaseModel):
    upload_path: str
    items: List[dict]


app = FastAPI(title="Iteration 4 MCP Bridge")


@app.get("/pending")
def pending_items() -> List[dict]:
    """Happy path: return open queue items for external review tools."""
    return queue.list_open()


@app.post("/label")
def label_conflict(body: LabelRequest) -> dict:
    """Happy path: apply a human label to a conflict via HTTP."""
    if body.label not in {"approve", "reject", "needs_context"}:
        raise HTTPException(status_code=400, detail="Label must be approve/reject/needs_context")
    queue.label(body.conflict_id, body.label, notes=body.notes or "labeled via MCP server")
    return {"status": "ok"}


@app.post("/queue")
def queue_items(body: BulkQueueRequest) -> dict:
    """Happy path: acknowledge that conflicts were dispatched to an external reviewer."""
    # No-op in this demo; real system would persist downstream notifications.
    return {"status": "queued", "count": len(body.items)}


