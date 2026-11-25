"""Cloud Run FastAPI service for Vertex AI Hybrid RAG."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.hybrid_retriever import HybridRRFEngine

app = FastAPI(title="GCP Vertex Hybrid RAG", version="1.0.0")
retriever = HybridRRFEngine()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=2, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "region": os.getenv("GCP_REGION", "europe-west3")}


@app.post("/v1/query")
def query_rag(req: QueryRequest) -> dict[str, object]:
    hits = retriever.search(req.query, top_k=req.top_k)
    answer = " ".join(hit.text for hit in hits)
    return {
        "query": req.query,
        "answer": answer,
        "citations": [{"doc_id": h.doc_id, "source": h.source, "rrf_score": h.rrf_score} for h in hits],
    }
