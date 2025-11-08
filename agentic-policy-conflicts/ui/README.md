# Agentic Policy Conflicts – UI

This directory contains a FastAPI-powered UI for running the policy conflict detector without using the CLI.

## Getting Started

1. Install the project dependencies (from the project root):

   ```bash
   pip install -e .
   ```

   The editable install ensures the UI can import the shared `src` package.

2. Launch the UI server:

   ```bash
   uvicorn ui.main:app --reload
   ```

   By default the server binds to `http://127.0.0.1:8000`. Use the `--host` / `--port` flags as needed.

3. Open the UI in a browser and submit a run:

   - Upload a policy document (PDF, DOCX, TXT, or CSV).
   - Choose the iteration to execute.
   - Optionally override the corpus directory or iteration-specific options.
   - Review the live activity log and rendered Markdown report.

## Notes

- Iteration 4 appears only when `iter4.runner_iter4` can be imported in the current environment.
- Uploaded files are written to a temporary directory during processing and removed when the run completes.
- The UI captures the existing logger output, so you get the same telemetry that the CLI path prints to the console.


