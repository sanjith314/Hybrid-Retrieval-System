"""SBERT retriever — real semantic retrieval using sentence-transformers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.domain.entities import Document, ScoredDocument

# Default model as specified in the report (384-dim)
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SBERTRetriever:
    """Dense retriever using Sentence-BERT cosine similarity.

    Satisfies the ``Retriever`` protocol defined in ``src.domain.protocols``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._doc_ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (n_docs, dim)

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # -- index lifecycle ----------------------------------------------------

    def build_index(self, documents: list[Document], batch_size: int = 32) -> None:
        """Encode all documents and store embeddings in memory."""
        model = self._ensure_model()
        self._doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.text for doc in documents]
        self._embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,  # pre-normalise so dot == cosine
        )

    def save_index(self, path: Path) -> None:
        """Persist doc_ids + embeddings array to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "doc_ids": self._doc_ids,
                "embeddings": self._embeddings,
                "model_name": self._model_name,
            }, f)

    def load_index(self, path: Path) -> None:
        """Load a previously saved index from *path*."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._doc_ids = data["doc_ids"]
        self._embeddings = data["embeddings"]
        self._model_name = data.get("model_name", _DEFAULT_MODEL)

    @property
    def is_ready(self) -> bool:
        return self._embeddings is not None and len(self._doc_ids) > 0

    # -- retrieval (Retriever protocol) -------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        """Return the *top_k* documents ranked by cosine similarity."""
        if not self.is_ready:
            raise RuntimeError(
                "SBERT index not built. Run 'embed-index' first."
            )

        model = self._ensure_model()

        # Encode query (normalised so dot product == cosine)
        q_emb = model.encode(
            [query], normalize_embeddings=True,
        )  # shape (1, dim)

        # Brute-force cosine similarity (dot product since vectors are normalised)
        scores = np.dot(self._embeddings, q_emb.T).flatten()  # shape (n_docs,)

        # Sort descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            ScoredDocument(
                doc_id=self._doc_ids[i],
                score=float(scores[i]),
                source="dense",
            )
            for i in top_indices
        ]
