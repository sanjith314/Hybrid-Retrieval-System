"""Dense stub retriever — deterministic hash-based pseudo-scorer."""

from __future__ import annotations

import hashlib

from src.domain.entities import ScoredDocument
from src.infrastructure.data.corpus_store import InMemoryCorpusStore


def _pseudo_score(query: str, doc_id: str) -> float:
    """Produce a deterministic float in [0, 1] from query + doc_id."""
    h = hashlib.sha256(f"{query}||{doc_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class DenseStubRetriever:
    """Scores documents with a deterministic hash-based pseudo-score.

    Acts as a repeatable stand-in for SBERT cosine similarity.
    """

    def __init__(self, corpus_store: InMemoryCorpusStore) -> None:
        self._store = corpus_store

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        scored: list[ScoredDocument] = []
        for doc_id in self._store.ids():
            score = _pseudo_score(query, doc_id)
            scored.append(ScoredDocument(doc_id=doc_id, score=score, source="dense"))

        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:top_k]
