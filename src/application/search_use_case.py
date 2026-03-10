"""Search use case — thin wrapper around the orchestrator for single-query search."""

from __future__ import annotations

from src.application.orchestrator import Orchestrator
from src.domain.entities import ScoredDocument


class SearchUseCase:
    """Execute a single-query search through the retrieval pipeline."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator

    def execute(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> list[ScoredDocument]:
        """Return ranked documents for *query*."""
        return self._orchestrator.search(query, top_k=top_k, mode=mode)
