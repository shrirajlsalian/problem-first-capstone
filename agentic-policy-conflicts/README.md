# Agentic Policy Conflicts

Iterations 1–3 of a policy conflict detector using LangGraph, LangChain, LangSmith, optional Tavily, and Opik-style metrics stubs.

## Quickstart

```bash
pip install -e .
cp .env.example .env  # fill keys
python -m src.app.main --iteration 1 --upload data/uploads/YourPolicy.pdf
```

See `src/app/main.py` for CLI options. Index your existing corpus in `data/existing/`.
