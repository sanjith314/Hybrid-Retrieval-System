"""Domain protocols — abstract interfaces that infrastructure adapters must satisfy."""

from __future__ import annotations

from typing import Any, Protocol

from src.domain.entities import ScoredDocument


class Retriever(Protocol):
    """Retrieves and scores documents for a query."""

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        """Return the *top_k* scored documents for *query*."""
        ...


class FusionStrategy(Protocol):
    """Fuses two ranked lists into a single ranking."""

    def fuse(
        self,
        sparse_results: list[ScoredDocument],
        dense_results: list[ScoredDocument],
        top_k: int = 10,
    ) -> list[ScoredDocument]:
        """Merge sparse and dense results, return *top_k*."""
        ...


class EvaluationMetric(Protocol):
    """Computes retrieval-quality metrics."""

    def compute(
        self,
        run: dict[str, list[ScoredDocument]],
        qrels: dict[str, list[str]],
        k_values: list[int],
    ) -> dict[str, Any]:
        """Return a mapping of metric names to values."""
        ...
