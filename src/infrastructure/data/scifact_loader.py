"""SciFact dataset loader — downloads from official S3 and converts to project format.

Uses the official SciFact release tarball which contains:
  - corpus.jsonl   (one doc per line: doc_id, title, abstract, structured)
  - claims_train.jsonl / claims_dev.jsonl / claims_test.jsonl
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

_SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def download_and_convert(data_dir: Path) -> dict[str, int]:
    """Download SciFact from S3 and write corpus/queries/qrels JSON files.

    Returns a summary dict with counts.
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── download and extract ──────────────────────────────────────────────
    print(f"Downloading SciFact from {_SCIFACT_URL} ...")
    resp = urllib.request.urlopen(_SCIFACT_URL)
    data = resp.read()
    print(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")

    tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")

    # Read files from tar archive
    corpus_lines = []
    claims_lines = []

    for member in tar.getmembers():
        name = member.name
        if name.endswith("corpus.jsonl"):
            f = tar.extractfile(member)
            if f:
                corpus_lines = f.read().decode("utf-8").strip().split("\n")
        elif "claims_train" in name or "claims_dev" in name:
            # We use train + dev (test has no labels)
            f = tar.extractfile(member)
            if f:
                claims_lines.extend(f.read().decode("utf-8").strip().split("\n"))

    tar.close()

    # ── corpus → corpus.json ─────────────────────────────────────────────
    print("Converting corpus...")
    docs = []
    for line in corpus_lines:
        row = json.loads(line)
        doc_id = str(row["doc_id"])
        title = row.get("title", "")
        # abstract is a list of sentences
        abstract_parts = row.get("abstract", [])
        abstract = " ".join(abstract_parts)
        text = f"{title} {abstract}".strip()
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "abstract": abstract,
            "text": text,
        })

    with open(raw_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {len(docs)} documents written")

    # ── claims → queries.json + qrels.json ────────────────────────────────
    print("Converting claims...")
    seen_ids: set[int] = set()
    queries = []
    qrels_map: dict[str, set[str]] = defaultdict(set)

    for line in claims_lines:
        row = json.loads(line)
        claim_id = row["id"]
        query_id = f"q_{claim_id}"

        # Add query only once per claim id
        if claim_id not in seen_ids:
            seen_ids.add(claim_id)
            queries.append({
                "query_id": query_id,
                "claim": row["claim"],
            })

        # Build qrels from evidence
        evidence = row.get("evidence", {})
        cited_doc_ids = row.get("cited_doc_ids", [])

        # cited_doc_ids are docs the claim references
        for cid in cited_doc_ids:
            qrels_map[query_id].add(str(cid))

        # evidence dict: {doc_id: [{label, sentences}, ...]}
        for ev_doc_id, ev_list in evidence.items():
            for ev in ev_list:
                if ev.get("label") == "SUPPORT":
                    qrels_map[query_id].add(str(ev_doc_id))

    with open(raw_dir / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {len(queries)} queries written")

    # Only keep qrels for queries that have relevant docs
    qrels = [
        {"query_id": qid, "relevant_doc_ids": sorted(doc_ids)}
        for qid, doc_ids in qrels_map.items()
        if doc_ids
    ]

    with open(raw_dir / "qrels.json", "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {len(qrels)} qrel entries written")

    return {
        "documents": len(docs),
        "queries": len(queries),
        "qrels": len(qrels),
    }
