"""In-memory corpus store — holds documents keyed by doc_id."""

from __future__ import annotations

from src.domain.entities import Document


class InMemoryCorpusStore:
    """Simple dict-backed store for corpus documents."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    # -- mutators ----------------------------------------------------------

    def add(self, doc: Document) -> None:
        """Add or replace a document."""
        self._docs[doc.doc_id] = doc

    def add_many(self, docs: list[Document]) -> None:
        for doc in docs:
            self.add(doc)

    # -- queries -----------------------------------------------------------

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def list_all(self) -> list[Document]:
        return list(self._docs.values())

    def ids(self) -> list[str]:
        return list(self._docs.keys())

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs
