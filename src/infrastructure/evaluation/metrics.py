"""Metrics engine — computes Precision@k, Recall@k, NDCG@k from scratch."""

from __future__ import annotations

import math
from typing import Any

from src.domain.entities import ScoredDocument


class MetricsEngine:
    """Computes standard IR evaluation metrics using only Python stdlib."""

    # -- public interface (satisfies EvaluationMetric protocol) ---------------

    def compute(
        self,
        run: dict[str, list[ScoredDocument]],
        qrels: dict[str, list[str]],
        k_values: list[int],
    ) -> dict[str, Any]:
        """Aggregate metrics across all queries for each k.

        Parameters
        ----------
        run : dict[query_id, list[ScoredDocument]]  (already ranked)
        qrels : dict[query_id, list[relevant_doc_id]]
        k_values : list of k values

        Returns
        -------
        dict with keys like "Precision@5", "Recall@10", "NDCG@20",
        each mapping to the *macro-average* across queries.
        """
        results: dict[str, float] = {}
        for k in k_values:
            precisions, recalls, ndcgs = [], [], []
            for qid, relevant in qrels.items():
                retrieved = [sd.doc_id for sd in run.get(qid, [])][:k]
                precisions.append(self._precision_at_k(retrieved, relevant, k))
                recalls.append(self._recall_at_k(retrieved, relevant))
                ndcgs.append(self._ndcg_at_k(retrieved, relevant, k))

            n = len(qrels) or 1
            results[f"Precision@{k}"] = sum(precisions) / n
            results[f"Recall@{k}"] = sum(recalls) / n
            results[f"NDCG@{k}"] = sum(ndcgs) / n

        return results

    # -- private helpers ------------------------------------------------------

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
        """Fraction of retrieved docs (up to k) that are relevant."""
        if k == 0:
            return 0.0
        rel_set = set(relevant)
        hits = sum(1 for d in retrieved[:k] if d in rel_set)
        return hits / k

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
        """Fraction of relevant docs found in the retrieved set."""
        if not relevant:
            return 0.0
        rel_set = set(relevant)
        hits = sum(1 for d in retrieved if d in rel_set)
        return hits / len(rel_set)

    @staticmethod
    def _ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
        """Normalised Discounted Cumulative Gain at k (binary relevance)."""
        rel_set = set(relevant)

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in rel_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1

        # Ideal DCG (all relevant docs at top positions)
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0
