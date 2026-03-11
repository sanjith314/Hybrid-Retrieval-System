# 🔍 Hybrid Retrieval System

A hybrid information retrieval system that combines **BM25 lexical search** with **SBERT semantic search** using late fusion, evaluated on the [SciFact](https://github.com/allenai/scifact) dataset.

## Architecture

```
Query
  │
  ├──► BM25 (sparse)     ──► normalize ──┐
  │                                       ├──► weighted fusion ──► ranked results
  └──► SBERT (dense)     ──► normalize ──┘
```

- **Sparse retriever**: BM25Okapi via `rank-bm25`
- **Dense retriever**: `all-MiniLM-L6-v2` (384-dim) via `sentence-transformers`
- **Fusion**: α-weighted combination (α=1 → sparse only, α=0 → dense only)
- **Evaluation**: Precision@k, Recall@k, NDCG@k

## Results on SciFact

| Metric | Sparse (BM25) | Dense (SBERT) | **Hybrid (α=0.5)** |
|--------|:---:|:---:|:---:|
| NDCG@10 | 0.6643 | 0.6561 | **0.7146** |
| Recall@20 | 0.8390 | 0.8410 | **0.8873** |

> Hybrid fusion outperforms both individual retrievers across all metrics.

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Load the SciFact dataset (~3 MB download)
python -m src.main load-scifact

# Build indexes (SBERT takes ~30s on CPU)
python -m src.main build-index
python -m src.main embed-index

# Search
python -m src.main search --query "mRNA vaccines are effective" --mode hybrid --top-k 10

# Evaluate
python -m src.main evaluate --mode hybrid --k 5 10 20
```

## Streamlit UI

```bash
streamlit run src/infrastructure/ui/streamlit_app.py
```

Features:
- Interactive search with mode/alpha/top-k controls
- **Explainability panel**: per-document sparse, dense, and hybrid score breakdown
- **Evaluation tab**: run metrics and export JSON

## CLI Commands

| Command | Description |
|---------|-------------|
| `load-scifact` | Download SciFact dataset from S3 |
| `build-stub-index` | Generate small sample dataset for testing |
| `build-index` | Build BM25 index from corpus |
| `embed-index` | Build SBERT embeddings (downloads model on first run) |
| `search --query "..." --mode [sparse\|dense\|hybrid]` | Run a retrieval query |
| `evaluate --mode [sparse\|dense\|hybrid] --k 5 10 20` | Compute P@k, R@k, NDCG@k |

## Project Structure

```
src/
├── domain/           # Entities (Document, Query, ScoredDocument) & protocols
├── application/      # Orchestrator, search & evaluate use cases
├── config/           # Settings
└── infrastructure/
    ├── data/         # Corpus store, JSON loader, SciFact downloader
    ├── retrieval/    # BM25, SBERT, preprocessor, stubs
    ├── fusion/       # Normalizer, weighted fusion
    ├── evaluation/   # Precision, Recall, NDCG
    └── ui/           # Streamlit app
tests/                # Unit tests (50 tests across 3 phases)
```

## Requirements

- Python 3.10+
- ~2 GB disk for `sentence-transformers` + `torch`
- SBERT model weights (~80 MB, auto-downloaded)
