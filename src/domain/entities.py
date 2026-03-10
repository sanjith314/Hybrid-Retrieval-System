"""Domain entities — pure data contracts for the retrieval system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A corpus document."""
    doc_id: str
    title: str
    abstract: str
    text: str  # typically title + " " + abstract


@dataclass(frozen=True)
class Query:
    """A retrieval query / scientific claim."""
    query_id: str
    claim: str


@dataclass
class ScoredDocument:
    """A document with a retrieval score and provenance tag."""
    doc_id: str
    score: float
    source: str  # "sparse", "dense", or "hybrid"


@dataclass(frozen=True)
class Qrel:
    """Ground-truth relevance judgments for a single query."""
    query_id: str
    relevant_doc_ids: list[str] = field(default_factory=list)
