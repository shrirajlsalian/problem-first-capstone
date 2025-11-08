import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure src is importable when running `uvicorn ui.main:app`
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from app.main import index_corpus, run_iter1, run_iter2, run_iter3  # noqa: E402

try:  # noqa: E402 -- imported after sys.path modification
    from iter4.runner_iter4 import run as run_iter4
except Exception as exc:  # pragma: no cover - optional iteration 4 support
    run_iter4 = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "Iteration 4 runner unavailable; UI will disable Iteration 4 option. Error: %s", exc
    )

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Policy Conflict Detector UI", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

DEFAULT_INDEX_DIR = "./data/existing"
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "agentic_policy_conflicts_ui"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _bool_from_form(value: Optional[str]) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return lowered in {"true", "1", "yes", "on"}


def _save_upload(upload: UploadFile) -> Path:
    filename = Path(upload.filename or "upload").name
    upload_path = UPLOAD_ROOT / filename
    with upload_path.open("wb") as f_out:
        shutil.copyfileobj(upload.file, f_out)
    upload.file.close()
    return upload_path


def _ensure_index_dir(index_dir: str) -> str:
    if os.path.isabs(index_dir):
        return index_dir
    return str(ROOT_DIR / index_dir)


def _capture_logs() -> tuple[io.StringIO, logging.Handler]:
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return buffer, handler


def _release_logs(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)


def _run_iteration(payload: Dict[str, Any]) -> Dict[str, Any]:
    iteration = payload["iteration"]
    upload_path = payload["upload_path"]
    index_dir = payload["index_dir"]
    approve_all = payload.get("approve_all", False)
    unattended = payload.get("unattended", False)
    use_interrupt = payload.get("use_interrupt", True)
    mcp_endpoint = payload.get("mcp_endpoint")

    tools = None
    try:
        from shared.tools import Tools

        tools = Tools()
        index_corpus(tools, index_dir)
    finally:
        # Explicit cleanup of any resources Tools might hold (LLM sessions, etc.)
        del tools

    if iteration == "1":
        conflicts, report = run_iter1(str(upload_path))
    elif iteration == "2":
        conflicts, report = run_iter2(str(upload_path))
    elif iteration == "3":
        conflicts, report = run_iter3(str(upload_path), approve_all=approve_all)
    elif iteration == "4":
        if run_iter4 is None:
            raise HTTPException(status_code=400, detail="Iteration 4 not available in this environment.")
        conflicts, report = run_iter4(
            str(upload_path),
            approve_all=approve_all,
            unattended=unattended,
            use_interrupt=use_interrupt,
            mcp_endpoint=mcp_endpoint,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported iteration: {iteration}")

    return {
        "iteration": iteration,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "report": report,
    }


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_index_dir": DEFAULT_INDEX_DIR,
            "iteration4_available": run_iter4 is not None,
        },
    )


@app.post("/run")
async def run_policy(request: Request) -> JSONResponse:
    form = await request.form()
    upload = form.get("upload")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="A policy document upload is required.")

    iteration = (form.get("iteration") or "").strip()
    if iteration not in {"1", "2", "3", "4"}:
        raise HTTPException(status_code=400, detail="Iteration must be one of 1, 2, 3, or 4.")
    if iteration == "4" and run_iter4 is None:
        raise HTTPException(status_code=400, detail="Iteration 4 is not available.")

    index_dir = (form.get("index_dir") or DEFAULT_INDEX_DIR).strip()
    index_dir = _ensure_index_dir(index_dir)

    approve_all = _bool_from_form(form.get("approve_all"))
    unattended = _bool_from_form(form.get("unattended"))
    use_interrupt = not _bool_from_form(form.get("disable_interrupts"))
    mcp_endpoint = (form.get("mcp_endpoint") or os.getenv("MCP_ENDPOINT", "http://localhost:8000")).strip()

    upload_path = _save_upload(upload)

    log_stream, handler = _capture_logs()
    try:
        result = _run_iteration(
            {
                "iteration": iteration,
                "upload_path": upload_path,
                "index_dir": index_dir,
                "approve_all": approve_all,
                "unattended": unattended,
                "use_interrupt": use_interrupt,
                "mcp_endpoint": mcp_endpoint,
            }
        )
        status = "success"
        error = None
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        logger.exception("UI run failed: %s", exc)
        status = "error"
        result = {}
        error = str(exc)
    finally:
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                logger.debug("Could not remove temporary upload at %s", upload_path)
        _release_logs(handler)
        logs = log_stream.getvalue()
        log_stream.close()

    response_payload = {"status": status, "logs": logs, **result}
    if error:
        response_payload["error"] = error
        return JSONResponse(status_code=500, content=jsonable_encoder(response_payload))
    return JSONResponse(content=jsonable_encoder(response_payload))


