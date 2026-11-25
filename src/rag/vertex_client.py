"""Vertex AI Vector Search & Gemini Grounding Client with Local RRF Fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.rag.hybrid_retriever import HybridRRFEngine, RetrievedChunk


@dataclass(frozen=True)
class VertexConfig:
    """Configuration for GCP Vertex AI Vector Search & Gemini."""

    project_id: str | None = os.getenv("GCP_PROJECT_ID")
    location: str = os.getenv("GCP_REGION", "europe-west3")
    index_endpoint_id: str | None = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
    deployed_index_id: str | None = os.getenv("VERTEX_DEPLOYED_INDEX_ID")
    gemini_model: str = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def is_configured(self) -> bool:
        return bool(self.project_id)


class VertexHybridRAGClient:
    """Queries Vertex AI Vector Search + Gemini when configured, or falls back to local RRF."""

    def __init__(self, config: VertexConfig | None = None, local_engine: HybridRRFEngine | None = None) -> None:
        self.config = config or VertexConfig()
        self.local_engine = local_engine or HybridRRFEngine()

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        """Retrieve ranked chunks from Vertex AI Vector Search or local RRF fallback."""
        if self.config.is_configured and self.config.index_endpoint_id and self.config.deployed_index_id:
            try:
                from google.cloud import aiplatform
                from vertexai.language_models import TextEmbeddingModel

                aiplatform.init(project=self.config.project_id, location=self.config.location)
                embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
                embedding = embed_model.get_embeddings([query])[0].values

                endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=self.config.index_endpoint_id
                )
                matches = endpoint.find_neighbors(
                    deployed_index_id=self.config.deployed_index_id,
                    queries=[embedding],
                    num_neighbors=top_k,
                )
                return [
                    RetrievedChunk(
                        doc_id=neighbor.id,
                        source=f"vertex://{self.config.location}/{neighbor.id}",
                        text=f"Vertex neighbor {neighbor.id} (distance={neighbor.distance:.4f})",
                        rrf_score=round(1.0 / (1.0 + float(neighbor.distance)), 6),
                    )
                    for neighbor in (matches[0] if matches else [])
                ]
            except Exception as exc:  # noqa: BLE001
                print(f"[VertexAI] Vector Search fallback to local RRF: {exc}")

        return self.local_engine.search(query, top_k=top_k)

    def generate_grounded_answer(self, query: str, hits: list[RetrievedChunk]) -> tuple[str, str]:
        """Synthesize grounded answer via Vertex AI Gemini or deterministic citation join."""
        context_block = "\n".join(f"[{h.source}] {h.text}" for h in hits)
        if self.config.is_configured:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                vertexai.init(project=self.config.project_id, location=self.config.location)
                model = GenerativeModel(self.config.gemini_model)
                prompt = (
                    "Answer the question strictly using the provided sources and cite [source].\n\n"
                    f"Sources:\n{context_block}\n\nQuestion: {query}"
                )
                response = model.generate_content(prompt)
                return response.text.strip(), f"vertex:{self.config.gemini_model}"
            except Exception as exc:  # noqa: BLE001
                print(f"[VertexAI] Gemini generation fallback to local synthesis: {exc}")

        return " ".join(hit.text for hit in hits), "local-rrf-fallback"
