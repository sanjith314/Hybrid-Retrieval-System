"""Evaluate use case — runs all queries, computes metrics, and persists results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.application.orchestrator import Orchestrator
from src.domain.entities import Qrel, Query, ScoredDocument
from src.infrastructure.evaluation.metrics import MetricsEngine


class EvaluateUseCase:
    """Evaluate the full pipeline over a set of queries and ground-truth."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        metrics_engine: MetricsEngine,
        runs_dir: Path,
    ) -> None:
        self._orchestrator = orchestrator
        self._metrics = metrics_engine
        self._runs_dir = runs_dir

    def execute(
        self,
        queries: list[Query],
        qrels: list[Qrel],
        k_values: list[int],
        mode: str = "hybrid",
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Run evaluation and return / persist metric results.

        Parameters
        ----------
        queries : queries to evaluate
        qrels : ground-truth relevance judgments
        k_values : list of k values for P/R/NDCG
        mode : "sparse", "dense", or "hybrid"
        top_k : how many docs each retriever returns (should be >= max(k_values))
        """
        # Build run dict: {query_id: [ScoredDocument, ...]}
        run: dict[str, list[ScoredDocument]] = {}
        for q in queries:
            run[q.query_id] = self._orchestrator.search(q.claim, top_k=top_k, mode=mode)

        # Build qrels dict: {query_id: [doc_id, ...]}
        qrels_dict: dict[str, list[str]] = {qr.query_id: qr.relevant_doc_ids for qr in qrels}

        # Compute metrics
        results = self._metrics.compute(run, qrels_dict, k_values)
        results["mode"] = mode
        results["num_queries"] = len(queries)

        # Pretty-print to terminal
        self._print_table(results, k_values)

        # Persist JSON
        self._save(results)

        return results

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _print_table(results: dict[str, Any], k_values: list[int]) -> None:
        header = f"{'Metric':<18}" + "".join(f"{'@'+str(k):>10}" for k in k_values)
        print("\n" + "=" * len(header))
        print(f"  Evaluation Results  (mode={results['mode']})")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for metric_name in ("Precision", "Recall", "NDCG"):
            row = f"{metric_name:<18}"
            for k in k_values:
                key = f"{metric_name}@{k}"
                val = results.get(key, 0.0)
                row += f"{val:>10.4f}"
            print(row)
        print("=" * len(header) + "\n")

    def _save(self, results: dict[str, Any]) -> None:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._runs_dir / f"metrics_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved → {path}")
