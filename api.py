"""
api.py — FastAPI bridge between the HTML frontend and the LlamaIndex/Groq backend.

Endpoints
---------
GET  /status          → health check (ping models + chromadb)
POST /upload          → ingest a PDF into ChromaDB
POST /query           → run the agentic query engine, return structured JSON
DELETE /documents/{name} → remove a document's vectors from ChromaDB

Run with:
    python api.py
or:
    uvicorn api:app --reload --port 8000
"""

import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path

import db
import chromadb
# nest_asyncio is applied lazily inside get_agent() — NOT at module level
# because uvicorn uses uvloop which conflicts with nest_asyncio at import time.
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

from agent_logic import get_agentic_engine
from app_config import get_dual_models
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────
load_dotenv()

CHROMA_PATH    = "./chroma_db"
COLLECTION     = "research_papers"
DATA_DIR       = "./data"

Path(DATA_DIR).mkdir(exist_ok=True)

# ─────────────────────────────────────────────
#  App + CORS
# ─────────────────────────────────────────────
app = FastAPI(title="Agentic Research Lab API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # fine for local dev; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Lazy singletons — load once, reuse forever
# ─────────────────────────────────────────────
_agent   = None
_db      = None

def get_db():
    global _db
    if _db is None:
        _db = chromadb.PersistentClient(path=CHROMA_PATH)
    return _db

def get_agent():
    global _agent
    if _agent is None:
        # Apply nest_asyncio here (not at module level) to avoid uvloop conflict
        try:
            import nest_asyncio
            import asyncio
            loop = asyncio.get_event_loop()
            nest_asyncio.apply(loop)
        except Exception:
            pass
        _agent = get_agentic_engine()
    return _agent

# ─────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    username: str = None

class QueryResponse(BaseModel):
    answer: str
    research_gaps: list[str]
    citations: list[dict]        # [{file, page}]
    steps: list[str]             # agent reasoning log

# ─────────────────────────────────────────────
#  Helper: parse response into structured parts
# ─────────────────────────────────────────────
def parse_response(response_obj) -> tuple[str, list[str], list[dict]]:
    """
    Splits the raw LlamaIndex response into:
      - main answer text  (everything before the Research Gaps section)
      - research_gaps     (bullet list items)
      - citations         (file + page from source_nodes)
    Uses multiple fallback patterns so gaps are never silently dropped.
    """
    raw_text = str(response_obj)

    # --- Pattern 1: explicit ### header ---
    gaps_pattern = re.compile(
        r"(?:^|\n)#{1,3}\s*(?:🔍\s*)?Identified Research Gaps.*",
        re.IGNORECASE,
    )
    gaps_match = gaps_pattern.search(raw_text)

    if gaps_match:
        answer_text = raw_text[: gaps_match.start()].strip()
        gaps_block  = raw_text[gaps_match.start():]
    else:
        # --- Pattern 2: look for any "Research Gaps" heading variant ---
        loose = re.search(
            r"(?:^|\n)(?:#{1,3}\s*)?(?:🔍\s*)?Research Gaps?:?",
            raw_text, re.IGNORECASE
        )
        if loose:
            answer_text = raw_text[: loose.start()].strip()
            gaps_block  = raw_text[loose.start():]
        else:
            # No dedicated section — treat full text as answer, gaps empty
            answer_text = raw_text.strip()
            gaps_block  = ""

    # Extract bullet items: handles -, *, 1., **Bold:** text
    gap_items = re.findall(
        r"^[ \t]*(?:-|\*|\d+\.)[ \t]+(.+?)(?=\n[ \t]*(?:-|\*|\d+\.)|\Z)",
        gaps_block,
        re.MULTILINE | re.DOTALL,
    )
    research_gaps = [g.strip().replace("\n", " ") for g in gap_items if g.strip()]

    # --- Citations from source nodes — enriched with snippet + section ---
    citations: list[dict] = []
    seen = set()
    if hasattr(response_obj, "source_nodes"):
        for node in response_obj.source_nodes:
            meta  = node.metadata or {}
            fname = meta.get("file_name", meta.get("filename", ""))
            page  = meta.get("source", meta.get("page_label", meta.get("page", "?")))

            if not fname or fname in ("Unknown", ""):
                continue

            key = f"{fname}::{page}"
            if key in seen:
                continue
            seen.add(key)

            # Extract a useful snippet: first 240 chars of the chunk text, cleaned up
            raw_text  = getattr(node, "text", "") or getattr(node, "node", {}).text if hasattr(node, "node") else ""
            snippet   = " ".join(raw_text.split())[:240]  # collapse whitespace, trim
            if len(raw_text.split()) * 5 > 240:
                snippet += "…"

            # Try to find a section heading in the first line of the chunk
            first_line = raw_text.strip().splitlines()[0].strip() if raw_text.strip() else ""
            # A section heading is usually short (<= 80 chars) and doesn't end with a period
            is_heading = len(first_line) <= 80 and not first_line.endswith(".")
            section    = first_line if is_heading else ""

            # Relevance score (if available from reranker)
            score = round(node.score, 3) if hasattr(node, "score") and node.score is not None else None

            citations.append({
                "file":    fname,
                "page":    str(page),
                "section": section,
                "snippet": snippet,
                "score":   score,
            })

    return answer_text, research_gaps, citations


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/status")
def status():
    """Quick health check used by the frontend status bar."""
    try:
        db  = get_db()
        col = db.get_or_create_collection(COLLECTION)
        doc_count = col.count()
        return {
            "llm":      "llama-3.3-70b-versatile (Groq)",
            "database": "ChromaDB connected",
            "documents": doc_count,
            "ready":    True,
        }
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


@app.get("/debug/chunks")
def debug_chunks():
    """
    Shows exactly how many chunks are indexed per PDF in ChromaDB.
    Use this to verify documents are properly ingested before querying.
    Visit: http://localhost:8000/debug/chunks
    """
    try:
        db  = get_db()
        col = db.get_or_create_collection(COLLECTION)
        total = col.count()

        if total == 0:
            return {
                "total_chunks": 0,
                "files": {},
                "warning": "ChromaDB is EMPTY — upload your PDFs first!"
            }

        # Count chunks per file
        results = col.get(include=["metadatas"])
        file_counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            fname = meta.get("file_name", meta.get("filename", "unknown"))
            file_counts[fname] = file_counts.get(fname, 0) + 1

        return {
            "total_chunks": total,
            "files": file_counts,
            "status": "✅ Documents indexed and ready"
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/documents/{filename}/serve")
def serve_pdf(filename: str):
    """
    Serve a PDF file directly so the frontend iframe can load it
    via a stable HTTP URL instead of a short-lived blob URL.
    Works even after page refresh.
    """
    safe_name = Path(filename).name           # strip any path traversal
    pdf_path  = Path(DATA_DIR) / safe_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found in data/")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@app.get("/documents")
def list_documents(username: str = None):
    """Return PDF filenames that belong to a specific user only."""
    if username:
        user_files = db.get_user_documents(username)
        # Only include files that still exist on disk
        visible = [f for f in user_files if (Path(DATA_DIR) / f).exists()]
    else:
        visible = [p.name for p in Path(DATA_DIR).glob("*.pdf")]
    return {"files": sorted(visible)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), username: str = None):
    """
    Receive a PDF from the browser, save it to ./data/, then ingest it
    into ChromaDB (extract → chunk → embed → store).
    Optionally links the file to a username for per-user document isolation.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest_path = Path(DATA_DIR) / file.filename
    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        # Ingest into ChromaDB
        _ingest_single_pdf(dest_path)

        # Register ownership in SQLite
        if username:
            db.register_document(username, file.filename)

        # Reload agent so it picks up the new document
        global _agent
        _agent = None

        return {
            "status":   "indexed",
            "filename": file.filename,
            "size_kb":  round(len(content) / 1024, 1),
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents/{filename}")
def delete_document(filename: str, username: str = None):
    """Remove all ChromaDB vectors that came from the given file."""
    try:
        chroma  = get_db()
        col = chroma.get_or_create_collection(COLLECTION)

        results = col.get(where={"file_name": filename})
        ids     = results.get("ids", [])
        if ids:
            col.delete(ids=ids)

        # Remove from data/ if no other user owns this file
        if username:
            db.remove_user_document(username, filename)
            # Only delete disk file if no other user references it
            remaining_owners = db.get_user_documents.__doc__  # placeholder
            conn = __import__('sqlite3').connect(db.DB_FILE)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM user_documents WHERE filename = ?", (filename,))
            count = cur.fetchone()[0]
            conn.close()
            if count == 0:
                dest = Path(DATA_DIR) / filename
                if dest.exists(): dest.unlink()
        else:
            dest = Path(DATA_DIR) / filename
            if dest.exists(): dest.unlink()

        global _agent
        _agent = None

        return {"status": "removed", "filename": filename, "vectors_deleted": len(ids)}

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Main endpoint: runs the agentic RAG pipeline and returns
    a structured JSON with answer, gaps, citations, and step log.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # --- Intent Router ---
    # Intercept casual greetings and rubbish to avoid running the heavy RAG pipeline
    q = req.question.strip().lower()
    greetings = {"hi", "hello", "hey", "yo", "sup", "how are you", "what's up", "good morning", "good afternoon", "good evening", "help", "who are you"}
    
    if q in greetings or (len(q) < 6 and "?" not in q):
        # Still save greeting exchange if user is logged in
        if req.username:
            db.save_chat_message(req.username, "user", req.question)
            db.save_chat_message(req.username, "assistant", "👋 **Hi there!** I am your Agentic AI Research Assistant.\n\nI am designed to analyze complex academic literature. Please upload some PDF papers to the workspace on the left, and we can start finding differences, comparing methodologies, and synthesizing information!")
        return QueryResponse(
            answer="👋 **Hi there!** I am your Agentic AI Research Assistant.\n\nI am designed to analyze complex academic literature. Please upload some PDF papers to the workspace on the left, and we can start finding differences, comparing methodologies, and synthesizing information!",
            research_gaps=[],
            citations=[],
            steps=["Intent Router → detected conversational query", "Bypassing RAG pipeline → responding directly"]
        )

    steps = [
        "Routing query → decomposing into sub-questions",
        "Embedding query → BAAI/bge-small-en-v1.5",
        "Searching ChromaDB → Dense (top-10) + BM25 keyword (top-10)",
        "Fusing results → Reciprocal Rank Fusion (RRF)",
        "Reranking → Cross-encoder (BAAI/bge-reranker-base, local)",
        "Synthesizing response → llama-3.3-70b-versatile",
    ]

    try:
        agent   = get_agent()
        response = agent.query(req.question)
        answer, gaps, citations = parse_response(response)

        # Persist the exchange to SQLite
        if req.username:
            db.save_chat_message(req.username, "user", req.question)
            db.save_chat_message(req.username, "assistant", answer, citations=citations, research_gaps=gaps)

        return QueryResponse(
            answer=answer,
            research_gaps=gaps,
            citations=citations,
            steps=steps,
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _ingest_single_pdf(pdf_path: Path):
    """Chunk + embed a single PDF and upsert into ChromaDB."""
    _, _, embed_model = get_dual_models()

    reader    = PyMuPDFReader()
    documents = reader.load_data(file_path=pdf_path)

    # Attach filename metadata so citations work
    for doc in documents:
        doc.metadata["file_name"] = pdf_path.name

    db              = get_db()
    chroma_col      = db.get_or_create_collection(COLLECTION)
    vector_store    = ChromaVectorStore(chroma_collection=chroma_col)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=False,
    )
    print(f"[api] Ingested: {pdf_path.name} ({len(documents)} pages)")


# ─────────────────────────────────────────────
#  Auth Routes
# ─────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/register")
def register(req: AuthRequest):
    success, msg = db.register_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

@app.post("/auth/login")
def login(req: AuthRequest):
    if db.verify_user(req.username, req.password):
        return {"status": "success", "message": "Login successful."}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

@app.get("/chat/history/{username}")
def get_history(username: str):
    history = db.get_chat_history(username)
    return {"status": "success", "history": history}

# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀  Agentic Research Lab API starting...")
    print("📡  Frontend: open frontend_preview/index.html in your browser")
    print("📖  API docs: http://localhost:8000/docs\n")
    # loop="asyncio" prevents uvloop from loading — uvloop conflicts with
    # nest_asyncio which LlamaIndex requires for Mac M-series chips.
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # reload=True spawns a uvloop subprocess — avoid
        loop="asyncio",     # force standard asyncio, not uvloop
    )
