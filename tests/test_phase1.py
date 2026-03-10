"""Phase 1 unit tests — entities, stubs, normalizer, fusion, metrics, orchestrator."""

from __future__ import annotations

import math

from src.domain.entities import Document, Qrel, Query, ScoredDocument
from src.infrastructure.data.corpus_store import InMemoryCorpusStore
from src.infrastructure.evaluation.metrics import MetricsEngine
from src.infrastructure.fusion.normalizer import min_max_normalize
from src.infrastructure.fusion.weighted_fusion import WeightedFusion
from src.infrastructure.retrieval.dense_stub import DenseStubRetriever
from src.infrastructure.retrieval.sparse_stub import SparseStubRetriever


# ── helpers ───────────────────────────────────────────────────────────────


def _make_store() -> InMemoryCorpusStore:
    store = InMemoryCorpusStore()
    store.add_many([
        Document("d1", "COVID vaccine", "mRNA vaccine efficacy trial results",
                 "COVID vaccine mRNA vaccine efficacy trial results"),
        Document("d2", "NLP advances", "Transformer models improve benchmarks",
                 "NLP advances Transformer models improve benchmarks"),
        Document("d3", "Climate change", "Global warming threatens biodiversity",
                 "Climate change Global warming threatens biodiversity"),
    ])
    return store


# ── entity tests ──────────────────────────────────────────────────────────


class TestEntities:
    def test_document_fields(self):
        d = Document("id1", "Title", "Abstract", "Title Abstract")
        assert d.doc_id == "id1"
        assert d.title == "Title"
        assert d.abstract == "Abstract"
        assert d.text == "Title Abstract"

    def test_query_fields(self):
        q = Query("q1", "Some claim")
        assert q.query_id == "q1"
        assert q.claim == "Some claim"

    def test_scored_document(self):
        sd = ScoredDocument("id1", 0.75, "hybrid")
        assert sd.doc_id == "id1"
        assert sd.score == 0.75
        assert sd.source == "hybrid"

    def test_qrel_fields(self):
        qr = Qrel("q1", ["d1", "d2"])
        assert qr.query_id == "q1"
        assert qr.relevant_doc_ids == ["d1", "d2"]


# ── corpus store tests ───────────────────────────────────────────────────


class TestCorpusStore:
    def test_add_and_get(self):
        store = _make_store()
        assert len(store) == 3
        assert store.get("d1") is not None
        assert store.get("nonexistent") is None
        assert "d2" in store

    def test_ids(self):
        store = _make_store()
        assert set(store.ids()) == {"d1", "d2", "d3"}


# ── sparse stub tests ────────────────────────────────────────────────────


class TestSparseStub:
    def test_returns_deterministic_results(self):
        store = _make_store()
        retriever = SparseStubRetriever(store)
        r1 = retriever.retrieve("vaccine efficacy", top_k=3)
        r2 = retriever.retrieve("vaccine efficacy", top_k=3)
        assert [sd.doc_id for sd in r1] == [sd.doc_id for sd in r2]

    def test_scores_are_positive(self):
        store = _make_store()
        retriever = SparseStubRetriever(store)
        results = retriever.retrieve("vaccine", top_k=3)
        assert all(sd.score >= 0 for sd in results)

    def test_ranked_descending(self):
        store = _make_store()
        retriever = SparseStubRetriever(store)
        results = retriever.retrieve("vaccine covid", top_k=3)
        scores = [sd.score for sd in results]
        assert scores == sorted(scores, reverse=True)

    def test_source_tag(self):
        store = _make_store()
        retriever = SparseStubRetriever(store)
        results = retriever.retrieve("vaccine", top_k=1)
        assert results[0].source == "sparse"


# ── dense stub tests ─────────────────────────────────────────────────────


class TestDenseStub:
    def test_returns_deterministic_results(self):
        store = _make_store()
        retriever = DenseStubRetriever(store)
        r1 = retriever.retrieve("some query", top_k=3)
        r2 = retriever.retrieve("some query", top_k=3)
        assert [sd.doc_id for sd in r1] == [sd.doc_id for sd in r2]
        assert [sd.score for sd in r1] == [sd.score for sd in r2]

    def test_scores_in_range(self):
        store = _make_store()
        retriever = DenseStubRetriever(store)
        results = retriever.retrieve("any query", top_k=3)
        assert all(0.0 <= sd.score <= 1.0 for sd in results)

    def test_source_tag(self):
        store = _make_store()
        retriever = DenseStubRetriever(store)
        results = retriever.retrieve("query", top_k=1)
        assert results[0].source == "dense"


# ── normalizer tests ─────────────────────────────────────────────────────


class TestNormalizer:
    def test_min_max(self):
        docs = [
            ScoredDocument("a", 10.0, "sparse"),
            ScoredDocument("b", 5.0, "sparse"),
            ScoredDocument("c", 0.0, "sparse"),
        ]
        normed = min_max_normalize(docs)
        assert normed[0].score == 1.0  # max → 1
        assert normed[1].score == 0.5  # mid → 0.5
        assert normed[2].score == 0.0  # min → 0

    def test_equal_scores_return_zero(self):
        docs = [
            ScoredDocument("a", 5.0, "sparse"),
            ScoredDocument("b", 5.0, "sparse"),
        ]
        normed = min_max_normalize(docs)
        assert all(sd.score == 0.0 for sd in normed)

    def test_empty_list(self):
        assert min_max_normalize([]) == []


# ── weighted fusion tests ────────────────────────────────────────────────


class TestWeightedFusion:
    def test_equal_weight(self):
        fusion = WeightedFusion(alpha=0.5)
        sparse = [ScoredDocument("d1", 1.0, "sparse")]
        dense = [ScoredDocument("d1", 0.6, "dense")]
        result = fusion.fuse(sparse, dense, top_k=1)
        assert len(result) == 1
        assert abs(result[0].score - 0.8) < 1e-9

    def test_sparse_only_doc(self):
        fusion = WeightedFusion(alpha=0.5)
        sparse = [ScoredDocument("d1", 1.0, "sparse")]
        dense = []  # d1 not in dense → defaults to 0.0
        result = fusion.fuse(sparse, dense, top_k=1)
        assert abs(result[0].score - 0.5) < 1e-9

    def test_alpha_bounds(self):
        import pytest
        with pytest.raises(ValueError):
            WeightedFusion(alpha=1.5)

    def test_source_tag(self):
        fusion = WeightedFusion(alpha=0.5)
        result = fusion.fuse(
            [ScoredDocument("d1", 1.0, "sparse")],
            [ScoredDocument("d1", 1.0, "dense")],
            top_k=1,
        )
        assert result[0].source == "hybrid"


# ── metrics engine tests ─────────────────────────────────────────────────


class TestMetricsEngine:
    def _build_run_and_qrels(self):
        # Query q1: relevant = {d1, d3}
        # Retrieved ranking: d1, d2, d3, d4, d5
        run = {
            "q1": [
                ScoredDocument("d1", 0.9, "hybrid"),
                ScoredDocument("d2", 0.8, "hybrid"),
                ScoredDocument("d3", 0.7, "hybrid"),
                ScoredDocument("d4", 0.6, "hybrid"),
                ScoredDocument("d5", 0.5, "hybrid"),
            ]
        }
        qrels = {"q1": ["d1", "d3"]}
        return run, qrels

    def test_precision_at_5(self):
        run, qrels = self._build_run_and_qrels()
        engine = MetricsEngine()
        results = engine.compute(run, qrels, [5])
        # 2 relevant in top-5 → P@5 = 0.4
        assert abs(results["Precision@5"] - 0.4) < 1e-9

    def test_recall_at_5(self):
        run, qrels = self._build_run_and_qrels()
        engine = MetricsEngine()
        results = engine.compute(run, qrels, [5])
        # Both relevant docs found → R@5 = 1.0
        assert abs(results["Recall@5"] - 1.0) < 1e-9

    def test_ndcg_at_5(self):
        run, qrels = self._build_run_and_qrels()
        engine = MetricsEngine()
        results = engine.compute(run, qrels, [5])
        # DCG: 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG: 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.6309 ≈ 1.6309
        expected_ndcg = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
        assert abs(results["NDCG@5"] - expected_ndcg) < 1e-4

    def test_precision_at_2(self):
        run, qrels = self._build_run_and_qrels()
        engine = MetricsEngine()
        results = engine.compute(run, qrels, [2])
        # top-2: d1 (rel), d2 (not rel) → P@2 = 0.5
        assert abs(results["Precision@2"] - 0.5) < 1e-9


# ── orchestrator integration test ────────────────────────────────────────


class TestOrchestrator:
    def test_hybrid_search_returns_results(self):
        from src.application.orchestrator import Orchestrator

        store = _make_store()
        sparse = SparseStubRetriever(store)
        dense = DenseStubRetriever(store)
        fusion = WeightedFusion(alpha=0.5)
        orch = Orchestrator(sparse, dense, fusion)

        results = orch.search("vaccine", top_k=3, mode="hybrid")
        assert len(results) > 0
        assert all(sd.source == "hybrid" for sd in results)

    def test_sparse_mode(self):
        from src.application.orchestrator import Orchestrator

        store = _make_store()
        sparse = SparseStubRetriever(store)
        dense = DenseStubRetriever(store)
        fusion = WeightedFusion()
        orch = Orchestrator(sparse, dense, fusion)

        results = orch.search("vaccine", top_k=3, mode="sparse")
        assert all(sd.source == "sparse" for sd in results)

    def test_dense_mode(self):
        from src.application.orchestrator import Orchestrator

        store = _make_store()
        sparse = SparseStubRetriever(store)
        dense = DenseStubRetriever(store)
        fusion = WeightedFusion()
        orch = Orchestrator(sparse, dense, fusion)

        results = orch.search("vaccine", top_k=3, mode="dense")
        assert all(sd.source == "dense" for sd in results)
