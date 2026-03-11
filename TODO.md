# TODO: Datasets for Hybrid Retrieval

## Core Dataset (Primary)

- [ ] Download **SciFact** corpus, queries, and qrels.
- [ ] Place raw files under `data/raw/scifact/`.
- [ ] Convert/normalize into project format (doc_id, title, abstract, text, query_id, qrels).
- [ ] Save processed outputs under `data/processed/scifact/`.
- [ ] Add a small sample split for quick local testing.

## Validation

- [ ] Verify query/doc ID alignment for each dataset.
- [ ] Run baseline retrieval (sparse-only) on each dataset.
- [ ] Run hybrid retrieval and record `P@k`, `R@k`, `NDCG@k` in `runs/`.
