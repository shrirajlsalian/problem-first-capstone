import argparse, os, glob, sys
import logging
from pathlib import Path
from rich import print

# Add src directory to path for absolute imports
# Resolve to absolute path to avoid duplication issues
src_path = Path(__file__).resolve().parent.parent
src_path_str = str(src_path)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

from shared.tools import Tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def index_corpus(tools: Tools, existing_dir: str):
    """Happy path: locate supported files in the corpus directory and ingest them into the vector store."""
    # Resolve path relative to project root if it's a relative path
    if not os.path.isabs(existing_dir):
        project_root = Path(__file__).resolve().parent.parent.parent
        existing_dir = str(project_root / existing_dir)
    
    logger.info(f"Indexing corpus from directory: {existing_dir}")
    
    # Check if directory exists
    if not os.path.exists(existing_dir):
        logger.warning(f"Directory does not exist: {existing_dir}")
        print(f"[yellow]Directory does not exist: {existing_dir}[/yellow]")
        return
    
    paths = []
    for ext in ("*.pdf","*.docx","*.doc","*.txt","*.csv"):
        paths += glob.glob(os.path.join(existing_dir, ext))
    if paths:
        logger.info(f"Found {len(paths)} files to index")
        # Store initial count to check if indexing was skipped
        try:
            initial_count = tools.vs._collection.count()
        except:
            initial_count = 0
        tools.index_existing(paths)
        # Check if new documents were added
        try:
            final_count = tools.vs._collection.count()
            if final_count > initial_count:
                logger.info(f"Successfully indexed {len(paths)} files into Chroma")
                print(f"[green]Indexed {len(paths)} files into Chroma[/green]")
            else:
                print(f"[blue]ChromaDB already contains data. Skipped indexing {len(paths)} files.[/blue]")
        except:
            # If we can't check, assume indexing happened
            logger.info(f"Successfully indexed {len(paths)} files into Chroma")
            print(f"[green]Indexed {len(paths)} files into Chroma[/green]")
    else:
        logger.warning(f"No files found in {existing_dir}")
        print(f"[yellow]No files found in {existing_dir}[/yellow]")

def run_iter1(upload_path: str):
    """Happy path: execute Iteration 1 on the given upload and print the resulting report."""
    from iter1.runner_iter1 import run
    conflicts, report = run(upload_path)
    print(report)

def run_iter2(upload_path: str):
    """Happy path: execute Iteration 2 with judge verification and print the resulting report."""
    from iter2.runner_iter2 import run
    conflicts, report = run(upload_path)
    print(report)

def run_iter3(upload_path: str, approve_all: bool = False):
    """Happy path: execute Iteration 3, optionally auto-approving, and print the resulting report."""
    from iter3.runner_iter3 import run
    conflicts, report = run(upload_path, approve_all=approve_all)
    print(report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", choices=["1","2","3","4"], required=True)
    ap.add_argument("--upload", required=True, help="Path to uploaded policy")
    ap.add_argument("--index", default="./data/existing", help="Directory of existing corpus")
    ap.add_argument("--approve-all", action="store_true", help="Iteration 3: auto-approve queued items")
    ap.add_argument("--unattended", action="store_true", help="Iteration 4: dispatch review via MCP without CLI prompts")
    ap.add_argument("--mcp-endpoint", default=os.getenv("MCP_ENDPOINT", "http://localhost:8000"), help="Iteration 4: MCP server base URL")
    args = ap.parse_args()

    logger.info(f"Starting policy conflict detection - Iteration {args.iteration}")
    logger.info(f"Upload path: {args.upload}")
    logger.info(f"Index directory: {args.index}")
    
    tools = Tools()
    index_corpus(tools, args.index)

    if args.iteration == "1":
        run_iter1(args.upload)
    elif args.iteration == "2":
        run_iter2(args.upload)
    elif args.iteration == "3":
        run_iter3(args.upload, approve_all=args.approve_all)
    else:
        from iter4.runner_iter4 import run as run_iter4
        conflicts, report = run_iter4(
            args.upload,
            approve_all=args.approve_all,
            unattended=args.unattended,
            use_interrupt=not args.unattended,
            mcp_endpoint=args.mcp_endpoint,
        )
        print(report)
