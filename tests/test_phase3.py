"""Phase 3 unit tests — SBERT retriever and index persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.domain.entities import Document
from src.infrastructure.retrieval.sbert_retriever import SBERTRetriever


_SAMPLE_DOCS = [
    Document("d1", "COVID vaccine", "mRNA vaccine efficacy trial results",
             "COVID vaccine mRNA vaccine efficacy trial results"),
    Document("d2", "NLP advances", "Transformer models improve benchmarks",
             "NLP advances Transformer models improve benchmarks"),
    Document("d3", "Climate change", "Global warming threatens biodiversity",
             "Climate change Global warming threatens biodiversity"),
]


class TestSBERTRetriever:
    """Tests require the model to be downloaded (happens on first run)."""

    def _build_retriever(self) -> SBERTRetriever:
        r = SBERTRetriever()
        r.build_index(_SAMPLE_DOCS)
        return r

    def test_is_ready_after_build(self):
        r = self._build_retriever()
        assert r.is_ready

    def test_not_ready_before_build(self):
        r = SBERTRetriever()
        assert not r.is_ready

    def test_retrieve_returns_results(self):
        r = self._build_retriever()
        results = r.retrieve("vaccine efficacy", top_k=3)
        assert len(results) == 3

    def test_source_tag(self):
        r = self._build_retriever()
        results = r.retrieve("vaccine", top_k=1)
        assert results[0].source == "dense"

    def test_relevant_doc_ranked_first(self):
        r = self._build_retriever()
        results = r.retrieve("mRNA vaccine efficacy COVID", top_k=3)
        # d1 is about vaccines — should rank first semantically
        assert results[0].doc_id == "d1"

    def test_scores_in_valid_range(self):
        r = self._build_retriever()
        results = r.retrieve("climate warming", top_k=3)
        for sd in results:
            assert -1.0 <= sd.score <= 1.0, f"Score {sd.score} out of cosine range"

    def test_results_sorted_descending(self):
        r = self._build_retriever()
        results = r.retrieve("vaccine", top_k=3)
        scores = [sd.score for sd in results]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic(self):
        r = self._build_retriever()
        r1 = r.retrieve("vaccine", top_k=3)
        r2 = r.retrieve("vaccine", top_k=3)
        assert [sd.doc_id for sd in r1] == [sd.doc_id for sd in r2]


class TestSBERTIndexPersistence:
    def test_save_and_load_roundtrip(self):
        r = SBERTRetriever()
        r.build_index(_SAMPLE_DOCS)
        results_before = r.retrieve("vaccine", top_k=3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sbert_index.pkl"
            r.save_index(path)
            assert path.exists()

            r2 = SBERTRetriever()
            r2.load_index(path)
            results_after = r2.retrieve("vaccine", top_k=3)

        assert [sd.doc_id for sd in results_before] == [sd.doc_id for sd in results_after]
        # Scores should be very close (float precision)
        for a, b in zip(results_before, results_after):
            assert abs(a.score - b.score) < 1e-6
