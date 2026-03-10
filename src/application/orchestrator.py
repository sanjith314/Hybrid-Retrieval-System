"""Orchestrator — coordinates retrieval, normalisation, and fusion."""

from __future__ import annotations

from src.domain.entities import ScoredDocument
from src.infrastructure.fusion.normalizer import min_max_normalize


class Orchestrator:
    """Central pipeline: sparse → dense → normalise → fuse → return."""

    def __init__(self, sparse_retriever, dense_retriever, fusion_strategy) -> None:
        self._sparse = sparse_retriever
        self._dense = dense_retriever
        self._fusion = fusion_strategy

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid") -> list[ScoredDocument]:
        """Run the retrieval pipeline.

        Parameters
        ----------
        query : plain-text query string
        top_k : number of results to return
        mode  : "sparse", "dense", or "hybrid"
        """
        if mode == "sparse":
            return self._sparse.retrieve(query, top_k)
        elif mode == "dense":
            return self._dense.retrieve(query, top_k)
        elif mode == "hybrid":
            sparse_results = self._sparse.retrieve(query, top_k)
            dense_results = self._dense.retrieve(query, top_k)

            sparse_norm = min_max_normalize(sparse_results)
            dense_norm = min_max_normalize(dense_results)

            return self._fusion.fuse(sparse_norm, dense_norm, top_k)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose sparse, dense, or hybrid.")
