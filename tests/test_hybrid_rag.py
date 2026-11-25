"""Tests for GCP Vertex Hybrid RAG RRF retriever and FastAPI endpoint."""

from fastapi.testclient import TestClient

from src.app.server import app
from src.rag.hybrid_retriever import HybridRRFEngine

client = TestClient(app)


def test_rrf_retriever_ranks_relevant_clause_first():
    engine = HybridRRFEngine()
    hits = engine.search("termination notice days", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "cuad-01"


def test_query_endpoint_returns_grounded_citations():
    resp = client.post("/v1/query", json={"query": "GDPR Schrems II fines", "top_k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"][0]["doc_id"] == "sec-10k-02"
    assert "GDPR" in body["answer"]
