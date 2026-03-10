"""Phase 2 unit tests — preprocessor, BM25 retriever, index persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.domain.entities import Document
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.retrieval.preprocessor import STOP_WORDS, tokenize


# ── preprocessor tests ────────────────────────────────────────────────────


class TestPreprocessor:
    def test_lowercases(self):
        assert tokenize("Hello WORLD", remove_stopwords=False) == ["hello", "world"]

    def test_removes_punctuation(self):
        tokens = tokenize("COVID-19, mRNA! vaccines.", remove_stopwords=False)
        assert tokens == ["covid", "19", "mrna", "vaccines"]

    def test_removes_stopwords(self):
        tokens = tokenize("this is a test of the system")
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "the" not in tokens
        assert "test" in tokens
        assert "system" in tokens

    def test_keeps_all_without_stopword_removal(self):
        tokens = tokenize("this is a test", remove_stopwords=False)
        assert tokens == ["this", "is", "a", "test"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_stop_words_is_frozen(self):
        assert isinstance(STOP_WORDS, frozenset)


# ── BM25 retriever tests ─────────────────────────────────────────────────

_SAMPLE_DOCS = [
    Document("d1", "COVID vaccine", "mRNA vaccine efficacy trial results",
             "COVID vaccine mRNA vaccine efficacy trial results"),
    Document("d2", "NLP advances", "Transformer models improve benchmarks",
             "NLP advances Transformer models improve benchmarks"),
    Document("d3", "Climate change", "Global warming threatens biodiversity",
             "Climate change Global warming threatens biodiversity"),
]


class TestBM25Retriever:
    def _build_retriever(self) -> BM25Retriever:
        r = BM25Retriever()
        r.build_index(_SAMPLE_DOCS)
        return r

    def test_is_ready_after_build(self):
        r = self._build_retriever()
        assert r.is_ready

    def test_not_ready_before_build(self):
        r = BM25Retriever()
        assert not r.is_ready

    def test_retrieve_returns_results(self):
        r = self._build_retriever()
        results = r.retrieve("vaccine efficacy", top_k=3)
        assert len(results) > 0

    def test_source_tag(self):
        r = self._build_retriever()
        results = r.retrieve("vaccine", top_k=1)
        assert results[0].source == "sparse"

    def test_relevant_doc_ranked_first(self):
        r = self._build_retriever()
        results = r.retrieve("vaccine efficacy mRNA", top_k=3)
        # d1 contains vaccine, efficacy, mRNA — should rank first
        assert results[0].doc_id == "d1"

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
        assert [sd.score for sd in r1] == [sd.score for sd in r2]


class TestBM25IndexPersistence:
    def test_save_and_load_roundtrip(self):
        r = BM25Retriever()
        r.build_index(_SAMPLE_DOCS)
        results_before = r.retrieve("vaccine", top_k=3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25.pkl"
            r.save_index(path)
            assert path.exists()

            r2 = BM25Retriever()
            r2.load_index(path)
            results_after = r2.retrieve("vaccine", top_k=3)

        assert [sd.doc_id for sd in results_before] == [sd.doc_id for sd in results_after]
        assert [sd.score for sd in results_before] == [sd.score for sd in results_after]
