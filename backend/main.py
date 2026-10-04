from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AeroMind API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    thread_id: str

from backend.rag.graph import get_rag_graph
from backend.rag.chunker import chunk_text
from backend.rag.document_loader import load_pdf
from backend.rag.vector_store import add_to_vector_store

rag_app = get_rag_graph()
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", Path(__file__).resolve().parents[1] / "data" / "documents"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

@app.post("/query")
async def process_query(req: QueryRequest):
    initial_state = {
        "query": req.query,
        "context": "",
        "sources": [],
        "response": ""
    }
    
    try:
        result = rag_app.invoke(initial_state)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="The local embedding model could not be loaded. Check the backend logs and Python security policy.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Question processing failed. Check the PostgreSQL connection and backend logs.",
        ) from exc
    
    return {
        "response": result["response"],
        "sources": result["sources"]
    }

@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF must be 25 MB or smaller.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{Path(original_name).stem}-{uuid4().hex[:8]}.pdf"
    file_path = DOCUMENTS_DIR / stored_name
    file_path.write_bytes(content)

    try:
        text = load_pdf(str(file_path))
        if not text.strip():
            raise HTTPException(status_code=422, detail="No readable text was found in the PDF.")

        chunks = chunk_text(text)
        metadatas = [{"source": stored_name, "chunk_index": index} for index in range(len(chunks))]
        add_to_vector_store(chunks, metadatas=metadatas)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not process and index this PDF.") from exc

    return {"filename": stored_name, "chunks_added": len(chunks)}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
