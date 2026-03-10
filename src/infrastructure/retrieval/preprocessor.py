"""Text preprocessing and tokenization for BM25 retrieval."""

from __future__ import annotations

import re

# Common English stop words (no NLTK dependency needed)
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "for", "from", "had", "has", "have", "he", "her", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "no",
    "nor", "not", "now", "of", "on", "or", "our", "out", "own", "say",
    "she", "so", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "too", "up", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
})

_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, strip punctuation, optionally remove stop words.

    Returns a *list* (not set) because BM25 needs term frequencies.
    """
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens
