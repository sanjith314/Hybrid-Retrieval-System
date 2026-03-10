"""Score normalizer — min-max normalizes ScoredDocument lists to [0, 1]."""

from __future__ import annotations

from src.domain.entities import ScoredDocument


def min_max_normalize(results: list[ScoredDocument]) -> list[ScoredDocument]:
    """Return a new list with scores scaled to [0, 1] via min-max.

    If all scores are equal the normalised score is set to 0.0
    (as specified in the architecture doc).
    """
    if not results:
        return []

    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    normalised: list[ScoredDocument] = []
    for r in results:
        norm_score = (r.score - lo) / span if span > 0 else 0.0
        normalised.append(ScoredDocument(doc_id=r.doc_id, score=norm_score, source=r.source))

    return normalised
