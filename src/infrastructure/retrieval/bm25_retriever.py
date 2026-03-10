"""BM25 retriever — real lexical retrieval using rank-bm25."""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.domain.entities import Document, ScoredDocument
from src.infrastructure.retrieval.preprocessor import tokenize


class BM25Retriever:
    """Okapi-BM25 retriever backed by ``rank_bm25.BM25Okapi``.

    Satisfies the ``Retriever`` protocol defined in ``src.domain.protocols``.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    # -- index lifecycle ----------------------------------------------------

    def build_index(self, documents: list[Document]) -> None:
        """Tokenise documents and build the BM25 index in memory."""
        self._doc_ids = [doc.doc_id for doc in documents]
        corpus_tokens = [tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(corpus_tokens)

    def save_index(self, path: Path) -> None:
        """Persist the BM25 index + doc-id mapping to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "doc_ids": self._doc_ids}, f)

    def load_index(self, path: Path) -> None:
        """Load a previously saved index from *path*."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._doc_ids = data["doc_ids"]

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None

    # -- retrieval (Retriever protocol) -------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        """Return the *top_k* documents ranked by BM25 score."""
        if not self.is_ready:
            raise RuntimeError(
                "BM25 index not built. Run 'build-index' first."
            )

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Pair (doc_id, score), sort descending
        paired = list(zip(self._doc_ids, scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        return [
            ScoredDocument(doc_id=did, score=float(s), source="sparse")
            for did, s in paired[:top_k]
        ]
