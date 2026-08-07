from fastapi import FastAPI
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

rag_app = get_rag_graph()

@app.post("/query")
async def process_query(req: QueryRequest):
    initial_state = {
        "query": req.query,
        "context": "",
        "sources": [],
        "response": ""
    }
    
    result = rag_app.invoke(initial_state)
    
    return {
        "response": result["response"],
        "sources": result["sources"]
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
