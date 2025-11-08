from typing import Dict, Any, List
import time
import logging

logger = logging.getLogger(__name__)

class ReviewQueue:
    """Simple in-memory queue to track conflicts awaiting human labels."""

    def __init__(self):
        """Happy path: initialize an empty queue of review items."""
        self.items: Dict[str, Dict[str, Any]] = {}
        logger.debug("[HITL-Queue] Initialized review queue")
    
    def enqueue(self, item: Dict[str, Any]):
        """Happy path: add or replace a conflict item that requires human review."""
        conflict_id = item["conflict_id"]
        self.items[conflict_id] = item
        logger.debug(f"[HITL-Queue] 📥 Enqueued conflict {conflict_id} (conf={item.get('confidence', 0):.2f}, rel={item.get('relation', 'unknown')})")
    
    def list_open(self) -> List[Dict[str, Any]]:
        """Happy path: return all items that are still awaiting a human label."""
        open_items = [v for v in self.items.values() if v.get("status","open")=="open"]
        logger.debug(f"[HITL-Queue] 📋 Listed {len(open_items)} open items for review")
        return open_items
    
    def label(self, conflict_id: str, label: str, notes: str = ""):
        """Happy path: store the manual decision for a specific conflict and timestamp it."""
        if conflict_id in self.items:
            it = self.items[conflict_id]
            old_status = it.get("status", "unknown")
            it["status"] = "labeled"
            it["label"] = label
            it["notes"] = notes
            it["labeled_ts"] = time.time()
            logger.info(f"[HITL-Queue] 🏷️  Labeled conflict {conflict_id}: {label} (notes: {notes[:50] if notes else 'none'})")
            logger.debug(f"[HITL-Queue]   Status changed: {old_status} → labeled")
        else:
            logger.warning(f"[HITL-Queue] ⚠️  Cannot label conflict {conflict_id}: not found in queue")

queue = ReviewQueue()
