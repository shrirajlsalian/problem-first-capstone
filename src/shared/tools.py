import os, json, re
import logging
import time
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from shared.schemas import Clause, NLIVerdict
from app import settings

try:
    from tavily import TavilyClient  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    TavilyClient = None

logger = logging.getLogger(__name__)

class Tools:
    def __init__(self, chunk_size=1200, chunk_overlap=150):
        """Initialize loaders, vector store, and LLM for policy analysis."""
        logger.info("[Tools] 🔧 Initializing Tools class")
        logger.info(f"[Tools] 📐 Configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.info(f"[Tools] 🤖 Embedding model: {settings.EMBED_MODEL}")
        logger.info(f"[Tools] 🤖 LLM model: {settings.OPENAI_MODEL}")
        
        logger.debug("[Tools] 🏗️  Creating text splitter...")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n","\n",". ","."," "]
        )
        
        logger.debug("[Tools] 🏗️  Initializing OpenAI embeddings...")
        self.emb = OpenAIEmbeddings(model=settings.EMBED_MODEL)
        
        logger.debug("[Tools] 🏗️  Connecting to ChromaDB vector store...")
        self.vs = Chroma(
            collection_name="policies",
            embedding_function=self.emb,
            persist_directory=settings.CHROMA_DIR
        )
        
        # Check existing document count
        try:
            existing_count = self.vs._collection.count()
            logger.info(f"[Tools] 📊 ChromaDB contains {existing_count} existing documents")
        except Exception as e:
            logger.debug(f"[Tools] Could not check ChromaDB count: {e}")
        
        self.bm25_corpus: List[Document] = []
        logger.debug(f"[Tools] 📊 BM25 corpus initialized (empty)")
        
        logger.debug("[Tools] 🏗️  Initializing ChatOpenAI LLM...")
        self.llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        self.conversation: List[BaseMessage] = []
        logger.info("[Tools] ✅ Tools initialization complete")
        self.web_client = None
        if getattr(settings, "USE_TAVILY", False) and TavilyClient:
            try:
                api_key = os.getenv("TAVILY_API_KEY")
                self.web_client = TavilyClient(api_key=api_key)  # type: ignore[arg-type]
                logger.info("[Tools] 🌐 Tavily web search client enabled")
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("[Tools] Tavily client initialization failed: %s", exc)
        elif getattr(settings, "USE_TAVILY", False):
            logger.warning("[Tools] Tavily requested but tavily package not installed.")

    # memory helpers
    def remember_user(self, content: str) -> None:
        """Happy path: persist a user-side utterance in conversation memory."""
        self.conversation.append(HumanMessage(content=content))

    def remember_assistant(self, content: str) -> None:
        """Happy path: persist an assistant-side utterance in conversation memory."""
        self.conversation.append(AIMessage(content=content))

    def call_llm(self, prompt: str) -> BaseMessage:
        """Happy path: issue a prompt using accumulated conversation memory."""
        messages = self.conversation + [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        self.conversation.append(messages[-1])
        if isinstance(response, BaseMessage):
            self.conversation.append(response)
            return response
        text = getattr(response, "content", str(response))
        ai_msg = AIMessage(content=text)
        self.conversation.append(ai_msg)
        return ai_msg

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Happy path: run a Tavily web search and return structured snippets."""
        if not self.web_client:
            logger.debug("[Tools] Web search requested but Tavily client unavailable")
            return []
        try:
            response = self.web_client.search(  # type: ignore[union-attr]
                query=query,
                search_depth="basic",
                max_results=max_results,
            )
            results = response.get("results", []) if isinstance(response, dict) else response
            cleaned: List[Dict[str, Any]] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                cleaned.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": item.get("content") or item.get("snippet"),
                    }
                )
            logger.debug("[Tools] Web search returned %d results for query='%s'", len(cleaned), query[:60])
            return cleaned
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning("[Tools] Web search failed: %s", exc)
            return []

    # loaders
    def load(self, path: str) -> List[Document]:
        start_time = time.time()
        """Happy path: read a supported document type (pdf/txt/csv/docx) into LangChain Documents."""
        logger.info(f"[Tools-Load] 📄 Loading document: {path}")
        ext = os.path.splitext(path)[1].lower()
        logger.debug(f"[Tools-Load] 📋 File extension: {ext}")
        
        try:
            if ext == ".pdf":
                logger.debug("[Tools-Load] 📄 Using PyPDFLoader for PDF file")
                docs = PyPDFLoader(path).load()
            elif ext in [".txt", ".csv"]:
                logger.debug(f"[Tools-Load] 📝 Using TextLoader for {ext} file")
                docs = TextLoader(path, encoding="utf-8").load()
            elif ext in [".docx", ".doc"]:
                logger.debug(f"[Tools-Load] 📄 Using Docx2txtLoader for {ext} file")
                docs = Docx2txtLoader(path).load()
            else:
                raise ValueError(f"Unsupported file type: {path}")
            
            elapsed = time.time() - start_time
            total_chars = sum(len(doc.page_content) for doc in docs)
            logger.info(f"[Tools-Load] ✅ Loaded {len(docs)} pages/sections ({total_chars:,} characters) in {elapsed:.2f}s")
            return docs
        except Exception as e:
            logger.error(f"[Tools-Load] ❌ Failed to load document {path}: {e}")
            raise

    def chunk(self, docs: List[Document]) -> List[Document]:
        start_time = time.time()
        """Happy path: split documents into overlap-aware chunks ready for clause extraction."""
        logger.debug(f"[Tools-Chunk] 🔪 Chunking {len(docs)} documents...")
        chunks = self.splitter.split_documents(docs)
        elapsed = time.time() - start_time
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        logger.debug(f"[Tools-Chunk] ✅ Created {len(chunks)} chunks (avg size: {avg_chunk_size:.0f} chars) in {elapsed:.2f}s")
        return chunks

    def to_clauses(self, docs: List[Document], src: str) -> List[Clause]:
        start_time = time.time()
        """Happy path: convert each chunk into a Clause with metadata including its source."""
        logger.debug(f"[Tools-ToClauses] 📝 Converting {len(docs)} documents to clauses (source: {src})")
        out = []
        skipped = 0
        for i, d in enumerate(docs):
            t = (d.page_content or "").strip()
            if not t:
                skipped += 1
                continue
            out.append(Clause(id=f"{src}:{i}", text=t, source_id=src, meta=d.metadata or {}))
        elapsed = time.time() - start_time
        logger.debug(f"[Tools-ToClauses] ✅ Created {len(out)} clauses (skipped {skipped} empty) in {elapsed:.3f}s")
        return out

    def index_existing(self, paths: List[str], *, force_rebuild=False):
        start_time = time.time()
        """Happy path: ingest existing policy files into both Chroma vector store and BM25 corpus."""
        logger.info("=" * 80)
        logger.info("[Tools-Index] 📚 INDEXING DOCUMENTS INTO VECTOR STORE")
        logger.info("=" * 80)
        logger.info(f"[Tools-Index] 📂 Processing {len(paths)} file(s)")
        
        # Check if ChromaDB already has data
        try:
            existing_count = self.vs._collection.count()
            if existing_count > 0 and not force_rebuild:
                logger.info(f"[Tools-Index] ⚠️  ChromaDB already contains {existing_count} documents")
                logger.info(f"[Tools-Index] ℹ️  Skipping ingestion (use force_rebuild=True to re-index)")
                
                # Rebuild BM25 corpus from existing Chroma data if it's empty
                if not self.bm25_corpus:
                    logger.info("[Tools-Index] 🔄 Rebuilding BM25 corpus from existing ChromaDB data...")
                    rebuild_start = time.time()
                    all_docs = self.vs.get(include=['documents', 'metadatas'])
                    if all_docs and all_docs.get('documents'):
                        for i, doc_text in enumerate(all_docs['documents']):
                            metadata = all_docs['metadatas'][i] if all_docs.get('metadatas') else {}
                            self.bm25_corpus.append(Document(page_content=doc_text, metadata=metadata))
                        rebuild_elapsed = time.time() - rebuild_start
                        logger.info(f"[Tools-Index] ✅ Rebuilt BM25 corpus: {len(self.bm25_corpus)} documents in {rebuild_elapsed:.2f}s")
                    else:
                        logger.warning("[Tools-Index] ⚠️  No documents found in ChromaDB for BM25 corpus")
                else:
                    logger.info(f"[Tools-Index] ℹ️  BM25 corpus already contains {len(self.bm25_corpus)} documents")
                logger.info("=" * 80)
                return
        except Exception as e:
            logger.debug(f"[Tools-Index] Could not check existing collection count: {e}. Proceeding with indexing.")
        
        logger.info(f"[Tools-Index] 🚀 Starting indexing process...")
        if force_rebuild:
            logger.warning("[Tools-Index] 🔄 Force rebuild enabled: clearing existing collection and corpus")
            self.vs.delete_collection()
            self.bm25_corpus.clear()
            logger.info("[Tools-Index] ✅ Cleared existing data")
        
        total_chunks = 0
        for i, p in enumerate(paths):
            file_start = time.time()
            logger.info(f"[Tools-Index] 📄 [{i+1}/{len(paths)}] Processing: {os.path.basename(p)}")
            try:
                chunks = self.chunk(self.load(p))
                logger.info(f"[Tools-Index]   📊 Generated {len(chunks)} chunks from this file")
                
                logger.debug(f"[Tools-Index]   💾 Adding chunks to vector store...")
                self.vs.add_documents(chunks)
                logger.debug(f"[Tools-Index]   ✅ Added to ChromaDB vector store")
                
                self.bm25_corpus.extend(chunks)
                logger.debug(f"[Tools-Index]   ✅ Added to BM25 corpus")
                
                total_chunks += len(chunks)
                file_elapsed = time.time() - file_start
                logger.info(f"[Tools-Index]   ✅ Completed in {file_elapsed:.2f}s")
            except Exception as e:
                logger.error(f"[Tools-Index]   ❌ Failed to index {p}: {e}")
                continue
        
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"[Tools-Index] ✅ INDEXING COMPLETE")
        logger.info(f"[Tools-Index] 📊 Statistics:")
        logger.info(f"[Tools-Index]   • Files processed: {len(paths)}")
        logger.info(f"[Tools-Index]   • Total chunks indexed: {total_chunks}")
        logger.info(f"[Tools-Index]   • BM25 corpus size: {len(self.bm25_corpus)}")
        try:
            final_count = self.vs._collection.count()
            logger.info(f"[Tools-Index]   • ChromaDB document count: {final_count}")
        except:
            pass
        logger.info(f"[Tools-Index] ⏱️  Total indexing time: {elapsed:.2f} seconds")
        logger.info("=" * 80)

    def retrieve(self, query: str, k=12) -> List[Document]:
        start_time = time.time()
        """Happy path: return up to k dense + sparse retrieval candidates combined for a query clause."""
        logger.debug(f"[Tools-Retrieve] 🔍 Retrieving top {k} candidates for query ({len(query)} chars)")
        
        # Dense retrieval (vector similarity)
        dense_start = time.time()
        dense = self.vs.similarity_search(query, k=k)
        dense_elapsed = time.time() - dense_start
        logger.debug(f"[Tools-Retrieve]   📊 Dense retrieval: {len(dense)} results in {dense_elapsed:.3f}s")
        
        # Sparse retrieval (BM25)
        sparse = []
        if self.bm25_corpus:
            sparse_start = time.time()
            bm25_retriever = BM25Retriever.from_documents(self.bm25_corpus)
            sparse = bm25_retriever.get_relevant_documents(query)[:k]
            sparse_elapsed = time.time() - sparse_start
            logger.debug(f"[Tools-Retrieve]   📊 Sparse retrieval (BM25): {len(sparse)} results in {sparse_elapsed:.3f}s")
        else:
            logger.debug(f"[Tools-Retrieve]   ⚠️  BM25 corpus empty, skipping sparse retrieval")
        
        # Combine and deduplicate (simple approach - keep order: dense first)
        combined = dense + sparse
        total_elapsed = time.time() - start_time
        logger.debug(f"[Tools-Retrieve] ✅ Retrieved {len(dense)} dense + {len(sparse)} sparse = {len(combined)} total candidates in {total_elapsed:.3f}s")
        return combined

    def nli_compare(self, a: str, b: str) -> Dict[str, Any]:
        start_time = time.time()
        """Happy path: call the LLM to classify relation/score between two clause texts as JSON."""
        logger.debug(f"[Tools-NLI] 🧠 Performing NLI comparison")
        logger.debug(f"[Tools-NLI]   Clause A: {len(a)} chars")
        logger.debug(f"[Tools-NLI]   Clause B: {len(b)} chars")
        
        prompt = (
            "You are a policy contradiction checker.\n"
            "Return STRICT JSON: {\"relation\": [contradiction, duplication, entailment, unrelated],"
            " \"confidence\": 0..1, \"rationale\": \"...\"}\n\n"
            f"Clause A:\n{a}\n\nClause B:\n{b}\n\nJSON:"
        )
        
        # Call LLM
        llm_start = time.time()
        try:
            msg = self.call_llm(prompt)
            out = msg.content or ""
            llm_elapsed = time.time() - llm_start
            logger.debug(f"[Tools-NLI]   🤖 LLM response received in {llm_elapsed:.2f}s")
            logger.debug(f"[Tools-NLI]   📄 Raw LLM response ({len(out)} chars): {out[:200]}..." if len(out) > 200 else f"[Tools-NLI]   📄 Raw LLM response: {out}")
        except Exception as e:
            logger.error(f"[Tools-NLI]   ❌ LLM invocation failed: {e}")
            return {"relation":"unrelated","confidence":0.1,"rationale":"LLM error: " + str(e)}
        
        clean_out = (out or "").strip()
        if clean_out.startswith("```"):
            # remove code fences e.g. ```json ... ```
            clean_out = clean_out.strip("`")
            if clean_out.lower().startswith("json"):
                clean_out = clean_out[4:].lstrip("\n:").strip()
        elif clean_out.lower().startswith("json"):
            clean_out = clean_out[4:].lstrip("\n:").strip()

        if not clean_out:
            logger.warning(f"[Tools-NLI]   ⚠️  LLM returned empty response, using fallback")
            return {"relation":"unrelated","confidence":0.3,"rationale":"empty llm response"}

        # helper to normalize various malformed outputs
        def _normalize_nli_payload(payload: Any) -> Dict[str, Any]:
            if isinstance(payload, list):
                # Prefer first dict item
                for item in payload:
                    if isinstance(item, dict):
                        payload = item
                        break
                else:
                    # List of strings – treat first entry as relation
                    first = payload[0] if payload else "unrelated"
                    payload = {"relation": first}
            if not isinstance(payload, dict):
                return {"relation": "unrelated", "confidence": 0.3, "rationale": "non-dict json"}
            rel = payload.get("relation")
            if isinstance(rel, list):
                payload["relation"] = rel[0] if rel else "unrelated"
            elif isinstance(rel, str):
                payload["relation"] = rel.strip()
            else:
                payload["relation"] = "unrelated"

            conf = payload.get("confidence", 0.3)
            if isinstance(conf, list):
                conf = conf[0] if conf else 0.3
            if isinstance(conf, str):
                try:
                    conf = float(conf.strip())
                except ValueError:
                    conf = 0.3
            if not isinstance(conf, (int, float)):
                conf = 0.3
            payload["confidence"] = float(conf)

            rationale = payload.get("rationale", "")
            if isinstance(rationale, list):
                rationale = " ".join(str(x) for x in rationale)
            payload["rationale"] = str(rationale)
            return payload

        # Parse JSON
        try:
            data = json.loads(clean_out)
            logger.debug(f"[Tools-NLI]   ✅ JSON parsed successfully (direct parse)")
            data = _normalize_nli_payload(data)
        except json.JSONDecodeError as e:
            logger.warning(f"[Tools-NLI]   ⚠️  Failed to parse NLI JSON directly: {e}")
            logger.warning(f"[Tools-NLI]   🔍 Reason: LLM response may contain extra text before/after JSON")
            logger.debug(f"[Tools-NLI]   🔍 Attempting regex extraction to find JSON object...")
            m = re.search(r"\{[\s\S]*\}", clean_out)
            if m:
                json_candidate = m.group(0)
                logger.debug(f"[Tools-NLI]   📋 Found JSON candidate ({len(json_candidate)} chars): {json_candidate[:150]}...")
                try:
                    data = json.loads(json_candidate)
                    logger.debug(f"[Tools-NLI]   ✅ Extracted JSON using regex successfully")
                    data = _normalize_nli_payload(data)
                except json.JSONDecodeError as e2:
                    logger.warning(f"[Tools-NLI]   ❌ Regex extraction also failed: {e2}")
                    logger.warning(f"[Tools-NLI]   🔄 Using fallback result")
                    data = {"relation":"unrelated","confidence":0.3,"rationale":"parse error: " + str(e2)}
            else:
                logger.warning(f"[Tools-NLI]   ❌ No JSON pattern found in response")
                logger.warning(f"[Tools-NLI]   🔄 Using fallback result")
                data = {"relation":"unrelated","confidence":0.3,"rationale":"parse error: no JSON found"}
        
        # Validate with schema
        try:
            result = NLIVerdict(**data).model_dump()
            total_elapsed = time.time() - start_time
            logger.debug(f"[Tools-NLI]   ✅ Validation passed: {result['relation']} (conf={result['confidence']:.2f})")
            logger.debug(f"[Tools-NLI]   ⏱️  Total NLI time: {total_elapsed:.2f}s")
            return result
        except ValidationError as e:
            logger.warning(f"[Tools-NLI]   ⚠️  NLI result validation failed: {e}")
            logger.warning(f"[Tools-NLI]   🔄 Using fallback result")
            return {"relation":"unrelated","confidence":0.3,"rationale":"schema error: " + str(e)}
