"""
Agentic Policy Conflict Detector (runner)
-----------------------------------------

Thin CLI entrypoint that wires the refactored package together.
"""
from __future__ import annotations

import os
from pathlib import Path

from policy_detector import PolicyConflictDetector, PolicyConfig


def main() -> None:
    base_dir = Path(os.getenv("POLICY_BASE_DIR", r"D:\\problem_first_ai\\Capstone\\problem-first-capstone")).expanduser()
    default_upload_path = Path(os.getenv("POLICY_UPLOAD_PATH", base_dir / "SOC_2_TSC_Structure.csv")).expanduser()
    default_existing_dir = Path(os.getenv("POLICY_EXISTING_DIR", base_dir / "documents")).expanduser()

    detector = PolicyConflictDetector(PolicyConfig())

    pdf_paths = sorted(p for p in default_existing_dir.glob("*.pdf") if p.is_file())
    if pdf_paths:
        detector.add_existing_policies([str(p) for p in pdf_paths])
    else:
        print(f"No PDF policies found in {default_existing_dir}. Skipping indexing.")

    if default_upload_path.is_file():
        conflicts, report = detector.run_conflict_detection(str(default_upload_path))
        print(report)
        if not conflicts:
            print("No conflicts detected.")
    else:
        print(f"Upload file not found at {default_upload_path}. Configure POLICY_UPLOAD_PATH or update defaults.")


if __name__ == "__main__":
    main()


