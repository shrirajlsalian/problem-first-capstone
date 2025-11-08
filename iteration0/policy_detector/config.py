from __future__ import annotations

import os
from pathlib import Path


class PolicyConfig:
    """Configuration for the Policy Conflict Detector."""

    def __init__(
        self,
        base_dir: str | None = None,
        openai_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        chroma_dir: str | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> None:
        self.base_dir = Path(
            base_dir or os.getenv("POLICY_BASE_DIR", r"D:\problem_first_ai\Capstone\problem-first-capstone")
        ).expanduser()
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", str(self.base_dir / "policy_index"))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


