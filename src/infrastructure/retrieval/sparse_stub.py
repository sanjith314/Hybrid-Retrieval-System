"""Sparse stub retriever — deterministic token-overlap (Jaccard) scorer."""

from __future__ import annotations

import re

from src.domain.entities import ScoredDocument
from src.infrastructure.data.corpus_store import InMemoryCorpusStore


def _tokenize(text: str) -> set[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SparseStubRetriever:
    """Scores documents by Jaccard similarity between query and document tokens.

    This is a deterministic, model-free stand-in for BM25.
    """

    def __init__(self, corpus_store: InMemoryCorpusStore) -> None:
        self._store = corpus_store

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[ScoredDocument] = []
        for doc in self._store.list_all():
            doc_tokens = _tokenize(doc.text)
            if not doc_tokens:
                continue
            intersection = query_tokens & doc_tokens
            union = query_tokens | doc_tokens
            score = len(intersection) / len(union)
            scored.append(ScoredDocument(doc_id=doc.doc_id, score=score, source="sparse"))

        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:top_k]
