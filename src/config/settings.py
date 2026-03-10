"""Application-wide settings with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/Hybrid-Retrieval-System


@dataclass
class Settings:
    """Centralised configuration for the hybrid-retrieval pipeline."""

    # Fusion weight: 1.0 = sparse-only, 0.0 = dense-only
    alpha: float = 0.5

    # Default number of results to return
    top_k: int = 10

    # k-values for evaluation metrics
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Paths (resolved relative to project root)
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    runs_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "runs")
    indexes_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "indexes")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.runs_dir,
            self.indexes_dir / "lexical",
            self.indexes_dir / "dense",
        ):
            d.mkdir(parents=True, exist_ok=True)
