# GCP Vertex AI Hybrid RAG Blueprint

<p align="left">
  <img src="https://img.shields.io/badge/Cloud-GCP%20Cloud%20Run%20%7C%20Vertex%20AI-0EA5E9?style=flat-square" alt="GCP">
  <img src="https://img.shields.io/badge/IaC-Terraform%20(europe--west3)-8B5CF6?style=flat-square" alt="Terraform">
  <img src="https://img.shields.io/badge/Retrieval-Hybrid%20Sparse%20%2B%20Dense%20RRF-10B981?style=flat-square" alt="Hybrid RRF">
  <img src="https://img.shields.io/badge/CI-Ruff%20%7C%20Bandit%20%7C%20Pytest-F59E0B?style=flat-square" alt="CI">
</p>

A serverless **Google Cloud Platform (GCP) Hybrid RAG** reference blueprint combining **Terraform** (`europe-west3` Frankfurt Cloud Run v2, Vertex AI Vector Search, and GCS), **Reciprocal Rank Fusion (RRF)** across lexical and dense vector retrieval over SEC/CUAD-style legal & financial documents, and a **FastAPI** serving layer.

---

## Table of Contents
- [Why Hybrid RRF on Sovereign GCP?](#why-hybrid-rrf-on-sovereign-gcp)
- [Architecture](#architecture)
- [Infrastructure & Retrieval Stack](#infrastructure--retrieval-stack)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [License](#license)

---

## Why Hybrid RRF on Sovereign GCP?

Pure dense-vector RAG struggles with exact legal citations, clause numbers, and regulatory acronyms (e.g., `HGB § 257`, `Schrems II`, `SEC 10-K`). Combining sparse lexical matching with dense vector similarity via **Reciprocal Rank Fusion (RRF)** delivers superior recall while keeping all storage and compute pinned to **GCP `europe-west3` (Frankfurt)**.

---

## Architecture

```
Client / Enterprise Agent
           │
           ▼
┌─────────────────────────────────────┐
│   Cloud Run v2 FastAPI Service      │
│   (europe-west3 / Frankfurt)        │
├─────────────────────────────────────┤
│ 1. Lexical Scorer (Exact Terms)     │
│ 2. Dense Vector Cosine Similarity   │ ──▶ Reciprocal Rank Fusion (k=60)
│ 3. Grounded Citation Assembler      │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   Terraform-Managed GCP Infra       │
│   • Vertex AI Vector Search         │
│   • Uniform-Access GCS Corpus       │
└─────────────────────────────────────┘
```

---

## Infrastructure & Retrieval Stack

- **Terraform (`terraform/main.tf`)**: Provisions `google_storage_bucket`, `google_vertex_ai_index_endpoint`, and `google_cloud_run_v2_service` in `europe-west3`.
- **Hybrid RRF Engine (`src/rag/hybrid_retriever.py`)**: Fuses lexical and dense ranks (`1 / (k + rank)`) with verifiable `doc_id` and `source` page citations.
- **FastAPI Service (`src/app/server.py`)**: Exposes `/health` and `/v1/query` with Pydantic v2 request validation.

---

## Quick Start

```bash
git clone https://github.com/FirdowsRahaman/gcp-vertex-hybrid-rag.git
cd gcp-vertex-hybrid-rag
pip install -r requirements.txt
PYTHONPATH=. pytest tests -v
python -m uvicorn src.app.server:app --port 8000
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "GDPR Schrems II fines", "top_k": 2}'
```

---

## License
This project is licensed under the MIT License.
