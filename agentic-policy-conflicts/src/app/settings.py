from dotenv import load_dotenv; load_dotenv()
import os

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BASE_DIR     = os.getenv("POLICY_BASE_DIR", "./data")
CHROMA_DIR   = os.getenv("CHROMA_DIR", "./storage/chroma")

USE_LANGSMITH = os.getenv("LANGCHAIN_TRACING_V2","false").lower() == "true"
USE_TAVILY    = bool(os.getenv("TAVILY_API_KEY"))

THRESH_STRONG = float(os.getenv("THRESH_STRONG", "0.55"))
THRESH_POSS   = float(os.getenv("THRESH_POSS", "0.40"))
