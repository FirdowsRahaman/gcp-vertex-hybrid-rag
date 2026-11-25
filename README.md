# GCP Vertex AI Hybrid RAG Blueprint

Serverless **GCP Hybrid RAG** reference architecture combining **Terraform** (Cloud Run v2 + Vertex AI Vector Search + GCS in `europe-west3`), **Reciprocal Rank Fusion (RRF)** over sparse lexical + dense vector retrieval, and a **FastAPI** serving layer with Gemini grounding hooks and deterministic offline execution for CI.

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests -v
```
