"""Cloud Run FastAPI service for Vertex AI Hybrid RAG."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.vertex_client import VertexHybridRAGClient

app = FastAPI(title="GCP Vertex Hybrid RAG", version="1.0.0")
rag_client = VertexHybridRAGClient()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=2, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "region": os.getenv("GCP_REGION", "europe-west3")}


@app.post("/v1/query")
def query_rag(req: QueryRequest) -> dict[str, object]:
    hits = rag_client.retrieve(req.query, top_k=req.top_k)
    answer, backend = rag_client.generate_grounded_answer(req.query, hits)
    return {
        "query": req.query,
        "backend": backend,
        "answer": answer,
        "citations": [{"doc_id": h.doc_id, "source": h.source, "rrf_score": h.rrf_score} for h in hits],
    }
