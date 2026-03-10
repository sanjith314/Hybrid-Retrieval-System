"""Weighted fusion — combines sparse and dense scores with an alpha weight."""

from __future__ import annotations

from src.domain.entities import ScoredDocument


class WeightedFusion:
    """HybridScore = alpha * sparse_norm + (1 - alpha) * dense_norm.

    Documents appearing in only one branch get 0.0 for the missing branch.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def fuse(
        self,
        sparse_results: list[ScoredDocument],
        dense_results: list[ScoredDocument],
        top_k: int = 10,
    ) -> list[ScoredDocument]:
        sparse_map: dict[str, float] = {r.doc_id: r.score for r in sparse_results}
        dense_map: dict[str, float] = {r.doc_id: r.score for r in dense_results}

        all_doc_ids = set(sparse_map) | set(dense_map)

        fused: list[ScoredDocument] = []
        for doc_id in all_doc_ids:
            s_score = sparse_map.get(doc_id, 0.0)
            d_score = dense_map.get(doc_id, 0.0)
            hybrid = self.alpha * s_score + (1 - self.alpha) * d_score
            fused.append(ScoredDocument(doc_id=doc_id, score=hybrid, source="hybrid"))

        fused.sort(key=lambda sd: sd.score, reverse=True)
        return fused[:top_k]
