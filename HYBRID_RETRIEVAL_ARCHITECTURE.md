# Hybrid Retrieval System Architecture (BM25 + Semantic)

## 1) Goal and Scope

This architecture is based on your uploaded report and presentation:
- BM25 lexical retrieval
- SBERT semantic retrieval (`all-MiniLM-L6-v2`, 384-dim in the report)
- Hybrid fusion using normalized score combination (`alpha`-weighted, default `0.5`)
- Evaluation with `Precision@k`, `Recall@k`, and `NDCG@k` for `k in {5,10,20}`
- Dataset target: SciFact (report mentions 5,183 docs and 300 queries)

Current constraint from your request:
- Do **not** connect to any database yet.
- Do **not** connect to any model yet.

So this plan defines a production-ready structure with mock/stub components first, then a clear path to turn on real BM25/SBERT later.

---

## 2) High-Level Architecture

```text
User Query
   |
   v
Query Service / API Layer
   |
   v
Retrieval Orchestrator
   |------------------------|
   v                        v
Sparse Retriever         Dense Retriever
(stub now, BM25 later)   (stub now, SBERT later)
   |                        |
   | top-k + scores         | top-k + scores
   |------------------------|
             v
      Score Normalizer
             v
        Fusion Module
    (weighted sum or RRF)
             v
      Ranked Result List
             |
             v
      Evaluation / UI Layer
```

Design principle from the report:
- Keep sparse and dense modules independent and fuse late.
- This makes debugging and upgrading easier (swap BM25, SBERT, FAISS, rerankers later without redesigning everything).

---

## 3) Implementation Layers

## 3.1 Domain Layer (Pure interfaces + data contracts)
- `Query`
- `Document`
- `ScoredDocument`
- `Retriever` interface:
  - `retrieve(query: str, top_k: int) -> list[ScoredDocument]`
- `FusionStrategy` interface:
  - `fuse(sparse_results, dense_results, top_k) -> list[ScoredDocument]`
- `EvaluationMetric` interface:
  - `compute(run, qrels, k_values)`

No external models or DB required here.

## 3.2 Application Layer (Use-cases / orchestration)
- `BuildIndexUseCase` (stub now)
- `SearchUseCase`
- `EvaluateUseCase`
- `Orchestrator`:
  - Calls sparse retriever
  - Calls dense retriever
  - Normalizes scores
  - Applies fusion
  - Returns final ranking

## 3.3 Infrastructure Layer (Adapters)
- `InMemoryCorpusStore` (for now)
- `JsonFileDatasetLoader` (for now)
- `BM25Retriever` (later)
- `SBERTRetriever` (later)
- `WeightedFusion` (now)
- `RRFFusion` (optional, based on slides)
- `MetricsEngine` (now)
- `StreamlitUI` or API adapter (later)

---

## 4) Project Structure (Recommended)

```text
hybrid-retrieval-system/
  src/
    domain/
      entities.py
      protocols.py
    application/
      orchestrator.py
      search_use_case.py
      evaluate_use_case.py
    infrastructure/
      data/
        dataset_loader.py
        corpus_store.py
      retrieval/
        sparse_stub.py
        dense_stub.py
        bm25_retriever.py          # later
        sbert_retriever.py         # later
      fusion/
        normalizer.py
        weighted_fusion.py
        rrf_fusion.py              # optional
      evaluation/
        metrics.py
      ui/
        streamlit_app.py           # later
      api/
        app.py                     # optional
    config/
      settings.py
  data/
    raw/
    processed/
  indexes/
    lexical/                       # later
    dense/                         # later
  runs/
  tests/
  requirements.txt
  README.md
```

---

## 5) Data Contracts

Use stable IDs everywhere to avoid alignment bugs (highlighted in the report).

## 5.1 Document
```json
{
  "doc_id": "12345",
  "title": "Document title",
  "abstract": "Document abstract",
  "text": "title + abstract"
}
```

## 5.2 Query
```json
{
  "query_id": "q_001",
  "claim": "Natural language scientific claim"
}
```

## 5.3 Qrels
```json
{
  "query_id": "q_001",
  "relevant_doc_ids": ["12345", "56789"]
}
```

## 5.4 Retrieval Output
```json
{
  "query_id": "q_001",
  "results": [
    {"doc_id": "12345", "score": 0.8123, "source": "hybrid"},
    {"doc_id": "56789", "score": 0.7441, "source": "hybrid"}
  ]
}
```

---

## 6) Retrieval and Fusion Logic

## 6.1 Sparse branch (now vs later)
- **Now (no model/db):** implement a deterministic token-overlap stub retriever.
- **Later:** replace with BM25 (`rank-bm25` or Pyserini).

## 6.2 Dense branch (now vs later)
- **Now (no model/db):** implement mock dense retriever returning deterministic pseudo-scores (hash-based or seeded random on token sets).
- **Later:** SBERT embeddings + cosine similarity (`all-MiniLM-L6-v2` as in report).

## 6.3 Score normalization
- Min-max normalize each branch separately to `[0,1]`.
- Handle edge-case: if all branch scores equal, output `0.0` for branch-normalized scores.

## 6.4 Weighted fusion (primary)
- Formula from report:
  - `HybridScore = alpha * bm25_norm + (1 - alpha) * sbert_norm`
- Default `alpha = 0.5`.
- Keep `alpha` configurable in settings.

## 6.5 Optional fusion (secondary)
- RRF from slides:
  - `score(doc) = sum(1 / (k + rank_i(doc)))`
- Useful if score scales are unstable across retrievers.

---

## 7) Evaluation Architecture

Implement evaluator independently so all retrievers are compared with the same protocol.

- Inputs: run file + qrels
- Metrics:
  - `Precision@5,10,20`
  - `Recall@5,10,20`
  - `NDCG@5,10,20`
- Output:
  - terminal table
  - machine-readable JSON (`runs/metrics_YYYYMMDD.json`)

Recommended experiment set:
1. Sparse-only
2. Dense-only
3. Hybrid (`alpha=0.5`)
4. Hybrid sensitivity (`alpha=0.3, 0.7`)

---

## 8) Interfaces (No external connections yet)

## 8.1 CLI commands
- `python -m src.main build-stub-index`
- `python -m src.main search --query "..." --top-k 10 --mode hybrid`
- `python -m src.main evaluate --mode hybrid --k 5 10 20`

## 8.2 Optional API endpoints
- `POST /search`
- `POST /evaluate`
- `GET /health`

All can run with local files + in-memory structures until real integrations are enabled.

---

## 9) Phased Implementation Plan

## Phase 1 (Now): Skeleton and stubs
- Build folder structure.
- Implement entities/interfaces.
- Add stub sparse + stub dense retrievers.
- Implement normalization + weighted fusion.
- Implement metric engine and test harness.
- Add CLI entrypoints.

## Phase 2: Real lexical retrieval
- Plug in BM25.
- Add preprocessing/tokenization module.
- Persist lexical index locally under `indexes/lexical/`.

## Phase 3: Real semantic retrieval
- Plug in SBERT encoder.
- Precompute document embeddings offline.
- Save embeddings to local files in `indexes/dense/`.
- Brute-force cosine for SciFact-sized corpus; FAISS optional later.

## Phase 4: UI and polish
- Streamlit interface for query exploration.
- Result explainability panel (sparse score, dense score, fused score).
- Export runs and metric summaries.

---

## 10) What You Need to Download/Connect to Finish the Full Project

This is the exact checklist you asked for.

## Required downloads/connections (later, not now)
1. **Dataset files**
   - SciFact corpus, queries, qrels/test split.
2. **Lexical retrieval dependency**
   - `rank-bm25` (or Pyserini + Java if you choose Pyserini route).
3. **Semantic model dependency**
   - `sentence-transformers`
   - Hugging Face model weights for `sentence-transformers/all-MiniLM-L6-v2`
4. **Numeric/ML runtime**
   - `numpy`, `scipy`, `torch`
5. **Evaluation utilities**
   - `scikit-learn` (optional but useful), or custom metric code only.
6. **UI / serving**
   - `streamlit` (if using web demo)
   - `fastapi` + `uvicorn` (if exposing API)
7. **Optional scalability**
   - `faiss-cpu` for ANN search on larger corpora.

## Not required right now
- Any SQL/NoSQL database.
- Any vector database.
- Remote model serving endpoint.

You can complete baseline project behavior entirely with local files + in-memory objects first.

---

## 11) Suggested `requirements.txt` Evolution

Current minimal setup (already created):
- `pypdf` (used for local PDF extraction/analysis)

When you move to full implementation, expand gradually:
- `rank-bm25`
- `sentence-transformers`
- `numpy`
- `scipy`
- `torch`
- `streamlit` (optional UI)
- `fastapi`, `uvicorn` (optional API)
- `faiss-cpu` (optional scale)

Pin versions after each successful integration and re-run `pip freeze > requirements.txt`.

---

## 12) Acceptance Criteria (Definition of Done)

1. With stubs only (no real model/db), `search` command returns deterministic ranked outputs and fusion behaves correctly.
2. Metrics module computes P@k, R@k, NDCG@k for all three modes (sparse, dense, hybrid).
3. Switching from stubs to real BM25/SBERT requires only adapter replacement, not orchestration rewrite.
4. Reproducible run artifacts saved under `runs/` with config + metrics.

This architecture gives you a clean path from coursework prototype to production-grade retrieval pipeline while respecting your current constraints.
