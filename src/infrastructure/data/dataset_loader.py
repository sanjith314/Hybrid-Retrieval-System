"""Dataset loader — reads/writes JSON files and generates sample data."""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities import Document, Qrel, Query


class JsonFileDatasetLoader:
    """Load and save corpus, queries, and qrels from/to JSON files."""

    def __init__(self, data_dir: Path) -> None:
        self._raw = data_dir / "raw"
        self._raw.mkdir(parents=True, exist_ok=True)

    # -- readers -----------------------------------------------------------

    def load_documents(self) -> list[Document]:
        path = self._raw / "corpus.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return [
            Document(
                doc_id=r["doc_id"],
                title=r["title"],
                abstract=r["abstract"],
                text=r.get("text", f"{r['title']} {r['abstract']}"),
            )
            for r in records
        ]

    def load_queries(self) -> list[Query]:
        path = self._raw / "queries.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return [Query(query_id=r["query_id"], claim=r["claim"]) for r in records]

    def load_qrels(self) -> list[Qrel]:
        path = self._raw / "qrels.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return [
            Qrel(query_id=r["query_id"], relevant_doc_ids=r["relevant_doc_ids"])
            for r in records
        ]

    # -- sample data generator ---------------------------------------------

    def generate_sample_data(self) -> None:
        """Write a small, deterministic sample dataset for testing."""
        docs = [
            {
                "doc_id": "doc_001",
                "title": "COVID-19 Vaccine Efficacy",
                "abstract": "This study evaluates the efficacy of mRNA vaccines against COVID-19 and reports a 95% effectiveness rate in clinical trials.",
                "text": "COVID-19 Vaccine Efficacy This study evaluates the efficacy of mRNA vaccines against COVID-19 and reports a 95% effectiveness rate in clinical trials.",
            },
            {
                "doc_id": "doc_002",
                "title": "Natural Language Processing Advances",
                "abstract": "Recent advances in transformer-based models have significantly improved performance on natural language understanding benchmarks.",
                "text": "Natural Language Processing Advances Recent advances in transformer-based models have significantly improved performance on natural language understanding benchmarks.",
            },
            {
                "doc_id": "doc_003",
                "title": "Climate Change and Biodiversity",
                "abstract": "Global warming is causing shifts in species distributions and threatening biodiversity in tropical ecosystems.",
                "text": "Climate Change and Biodiversity Global warming is causing shifts in species distributions and threatening biodiversity in tropical ecosystems.",
            },
            {
                "doc_id": "doc_004",
                "title": "SARS-CoV-2 Variants",
                "abstract": "Emerging variants of SARS-CoV-2 show increased transmissibility and partial immune escape from existing vaccines.",
                "text": "SARS-CoV-2 Variants Emerging variants of SARS-CoV-2 show increased transmissibility and partial immune escape from existing vaccines.",
            },
            {
                "doc_id": "doc_005",
                "title": "Deep Learning for Protein Folding",
                "abstract": "AlphaFold demonstrates that deep learning can predict protein structures with atomic-level accuracy.",
                "text": "Deep Learning for Protein Folding AlphaFold demonstrates that deep learning can predict protein structures with atomic-level accuracy.",
            },
            {
                "doc_id": "doc_006",
                "title": "Renewable Energy Storage",
                "abstract": "Lithium-ion battery improvements and solid-state batteries are enabling more efficient renewable energy storage solutions.",
                "text": "Renewable Energy Storage Lithium-ion battery improvements and solid-state batteries are enabling more efficient renewable energy storage solutions.",
            },
            {
                "doc_id": "doc_007",
                "title": "mRNA Vaccine Technology",
                "abstract": "mRNA vaccine platforms can be rapidly adapted to new pathogens and have shown strong safety profiles in large-scale trials.",
                "text": "mRNA Vaccine Technology mRNA vaccine platforms can be rapidly adapted to new pathogens and have shown strong safety profiles in large-scale trials.",
            },
            {
                "doc_id": "doc_008",
                "title": "Quantum Computing Applications",
                "abstract": "Quantum computers promise breakthroughs in cryptography, drug discovery, and optimization problems.",
                "text": "Quantum Computing Applications Quantum computers promise breakthroughs in cryptography, drug discovery, and optimization problems.",
            },
        ]

        queries = [
            {"query_id": "q_001", "claim": "mRNA vaccines are effective against COVID-19"},
            {"query_id": "q_002", "claim": "Transformer models improve NLP benchmarks"},
            {"query_id": "q_003", "claim": "Climate change threatens tropical biodiversity"},
        ]

        qrels = [
            {"query_id": "q_001", "relevant_doc_ids": ["doc_001", "doc_004", "doc_007"]},
            {"query_id": "q_002", "relevant_doc_ids": ["doc_002"]},
            {"query_id": "q_003", "relevant_doc_ids": ["doc_003"]},
        ]

        for name, data in [("corpus.json", docs), ("queries.json", queries), ("qrels.json", qrels)]:
            with open(self._raw / name, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
