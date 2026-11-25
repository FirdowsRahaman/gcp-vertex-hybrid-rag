"""Hybrid Sparse + Dense Vector Retriever with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """Corpus chunk representing SEC / CUAD contract clauses."""

    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """Ranked retrieval result with RRF score."""

    doc_id: str
    source: str
    text: str
    rrf_score: float


DEFAULT_CORPUS: tuple[DocumentChunk, ...] = (
    DocumentChunk(
        doc_id="cuad-01",
        source="CUAD_MSA_2025.pdf#p12",
        text=(
            "Termination for Convenience: Either party may terminate this Agreement "
            "upon 60 days prior written notice."
        ),
    ),
    DocumentChunk(
        doc_id="sec-10k-02",
        source="SEC_10K_RiskFactors.pdf#p4",
        text=(
            "Data Residency Risk: Non-compliance with EU GDPR or Schrems II "
            "cross-border transfer rules may incur fines."
        ),
    ),
    DocumentChunk(
        doc_id="cuad-03",
        source="CUAD_DPA_2025.pdf#p3",
        text=(
            "Limitation of Liability: Aggregate liability shall not exceed "
            "12 months of fees paid under the Order Form."
        ),
    ),
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_tf(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / denom if denom > 0 else 0.0


class HybridRRFEngine:
    """Combines lexical BM25-like scoring and dense-vector cosine similarity via RRF."""

    def __init__(self, corpus: tuple[DocumentChunk, ...] = DEFAULT_CORPUS, rrf_k: int = 60) -> None:
        self.corpus = corpus
        self.rrf_k = rrf_k
        self._doc_tfs = [Counter(_tokenize(doc.text)) for doc in corpus]

    def search(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        """Return top_k chunks ranked by Reciprocal Rank Fusion."""
        q_tokens = _tokenize(query)
        q_tf = Counter(q_tokens)

        lexical_scores = [
            (idx, sum(self._doc_tfs[idx][tok] for tok in set(q_tokens)))
            for idx in range(len(self.corpus))
        ]
        lexical_ranked = [idx for idx, _ in sorted(lexical_scores, key=lambda x: x[1], reverse=True)]

        vector_scores = [
            (idx, _cosine_tf(q_tf, self._doc_tfs[idx]))
            for idx in range(len(self.corpus))
        ]
        vector_ranked = [idx for idx, _ in sorted(vector_scores, key=lambda x: x[1], reverse=True)]

        rrf_totals: dict[int, float] = {i: 0.0 for i in range(len(self.corpus))}
        for rank, idx in enumerate(lexical_ranked, start=1):
            rrf_totals[idx] += 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(vector_ranked, start=1):
            rrf_totals[idx] += 1.0 / (self.rrf_k + rank)

        ordered = sorted(rrf_totals.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                doc_id=self.corpus[idx].doc_id,
                source=self.corpus[idx].source,
                text=self.corpus[idx].text,
                rrf_score=round(score, 6),
            )
            for idx, score in ordered
        ]
