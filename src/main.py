"""CLI entrypoint for the hybrid retrieval system.

Usage:
    python -m src.main build-stub-index
    python -m src.main search --query "..." --top-k 10 --mode hybrid
    python -m src.main evaluate --mode hybrid --k 5 10 20
"""

from __future__ import annotations

import argparse
import sys

from src.application.evaluate_use_case import EvaluateUseCase
from src.application.orchestrator import Orchestrator
from src.application.search_use_case import SearchUseCase
from src.config.settings import Settings
from src.infrastructure.data.corpus_store import InMemoryCorpusStore
from src.infrastructure.data.dataset_loader import JsonFileDatasetLoader
from src.infrastructure.evaluation.metrics import MetricsEngine
from src.infrastructure.fusion.weighted_fusion import WeightedFusion
from src.infrastructure.retrieval.dense_stub import DenseStubRetriever
from src.infrastructure.retrieval.sparse_stub import SparseStubRetriever


def _build_components(settings: Settings):
    """Wire up all components and return (loader, store, orchestrator)."""
    loader = JsonFileDatasetLoader(settings.data_dir)
    store = InMemoryCorpusStore()

    # Load corpus into store
    for doc in loader.load_documents():
        store.add(doc)

    sparse = SparseStubRetriever(store)
    dense = DenseStubRetriever(store)
    fusion = WeightedFusion(alpha=settings.alpha)
    orchestrator = Orchestrator(sparse, dense, fusion)

    return loader, store, orchestrator


# ── sub-commands ──────────────────────────────────────────────────────────


def cmd_build_stub_index(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.ensure_dirs()

    loader = JsonFileDatasetLoader(settings.data_dir)
    loader.generate_sample_data()

    # Verify by loading back
    docs = loader.load_documents()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    print(f"✓ Sample data generated under {settings.data_dir / 'raw'}")
    print(f"  {len(docs)} documents, {len(queries)} queries, {len(qrels)} qrel entries")


def cmd_search(args: argparse.Namespace) -> None:
    settings = Settings(alpha=args.alpha)
    _, _, orchestrator = _build_components(settings)

    results = SearchUseCase(orchestrator).execute(
        query=args.query, top_k=args.top_k, mode=args.mode,
    )

    if not results:
        print("No results found.")
        return

    print(f"\nSearch results (mode={args.mode}, top_k={args.top_k}, alpha={settings.alpha}):\n")
    print(f"{'Rank':<6}{'Doc ID':<12}{'Score':<12}{'Source':<10}")
    print("-" * 40)
    for i, sd in enumerate(results, 1):
        print(f"{i:<6}{sd.doc_id:<12}{sd.score:<12.4f}{sd.source:<10}")
    print()


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = Settings(alpha=args.alpha, k_values=args.k)
    loader, _, orchestrator = _build_components(settings)

    queries = loader.load_queries()
    qrels = loader.load_qrels()

    if not queries:
        print("No queries found. Run 'build-stub-index' first.")
        sys.exit(1)

    metrics_engine = MetricsEngine()
    evaluator = EvaluateUseCase(orchestrator, metrics_engine, settings.runs_dir)
    evaluator.execute(
        queries=queries,
        qrels=qrels,
        k_values=settings.k_values,
        mode=args.mode,
        top_k=max(settings.k_values) * 2,
    )


# ── argument parser ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-retrieval",
        description="Hybrid Retrieval System (BM25 + Semantic) — Phase 1 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-stub-index
    sub.add_parser("build-stub-index", help="Generate sample data files")

    # search
    sp_search = sub.add_parser("search", help="Run a retrieval query")
    sp_search.add_argument("--query", required=True, help="Plain-text query")
    sp_search.add_argument("--top-k", type=int, default=10, help="Number of results")
    sp_search.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    sp_search.add_argument("--alpha", type=float, default=0.5, help="Fusion weight (0=dense, 1=sparse)")

    # evaluate
    sp_eval = sub.add_parser("evaluate", help="Evaluate retrieval quality")
    sp_eval.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    sp_eval.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="k-values for metrics")
    sp_eval.add_argument("--alpha", type=float, default=0.5, help="Fusion weight (0=dense, 1=sparse)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "build-stub-index": cmd_build_stub_index,
        "search": cmd_search,
        "evaluate": cmd_evaluate,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
