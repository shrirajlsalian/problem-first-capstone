from shared.schemas import PolicyState

def render_markdown(state: PolicyState) -> str:
    """Happy path: build a markdown summary of conflicts captured in the policy state."""
    lines = [
        "# Conflict Report",
        "## Summary",
        f"- Upload: {state['upload_path']}",
        f"- Conflicts detected: {len(state['conflicts'])}",
        "## Conflicts",
    ]
    for c in state["conflicts"]:
        lines.append(f"- **{c.relation}** ({c.confidence:.2f}) — {c.new_id} vs {c.existing_id}")
        lines.append(f"  - {c.rationale}")
    remediations = state["remediations"] if isinstance(state, dict) and "remediations" in state else None
    if remediations:
        lines.append("## Recommended Actions")
        for c in state["conflicts"]:
            key = f"{c.new_id}|{c.existing_id}"
            suggestion = remediations.get(key)
            if suggestion:
                lines.append(f"- {key}: {suggestion}")
    return "\n".join(lines)
