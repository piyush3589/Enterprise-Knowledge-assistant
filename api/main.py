"""
FastAPI — REST layer for the knowledge assistant.

Endpoints:
  POST /query         → run full multi-agent pipeline
  POST /query/batch   → multiple questions at once
  GET  /documents     → list all documents in knowledge base
  GET  /health        → health check
"""

import sys
import time
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.models import QueryRequest, QueryResponse
from agents.pipeline import run_pipeline
from agents.retriever import retriever

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Multi-agent RAG system with hybrid search and reranking for industrial IoT documentation.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Enterprise Knowledge Assistant"}


@app.get("/documents")
def list_documents():
    """List all documents in the knowledge base."""
    try:
        retriever._ensure_loaded()
        docs = []
        for doc_id, meta in zip(retriever._all_ids, retriever._all_metadata):
            docs.append({"id": doc_id, **meta})
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {e}")


@app.post("/query")
def query(req: QueryRequest):
    """
    Run the full multi-agent pipeline:
    Router → Hybrid Retrieval → Reranking → Answer Generation
    """
    start = time.time()
    try:
        result = run_pipeline(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    duration_ms = int((time.time() - start) * 1000)
    return {**result, "duration_ms": duration_ms}


@app.post("/query/batch")
def query_batch(queries: list[str]):
    """Run multiple queries sequentially."""
    results = []
    for q in queries:
        try:
            result = run_pipeline(q)
            results.append(result)
        except Exception as e:
            results.append({"query": q, "error": str(e)})
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
