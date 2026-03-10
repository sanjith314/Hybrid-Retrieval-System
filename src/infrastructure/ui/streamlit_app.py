"""Streamlit UI for the Hybrid Retrieval System.

Launch with:
    cd /path/to/Hybrid-Retrieval-System
    .venv/bin/streamlit run src/infrastructure/ui/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Ensure project root is on path so `src.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.application.orchestrator import Orchestrator
from src.config.settings import Settings
from src.infrastructure.data.corpus_store import InMemoryCorpusStore
from src.infrastructure.data.dataset_loader import JsonFileDatasetLoader
from src.infrastructure.evaluation.metrics import MetricsEngine
from src.infrastructure.fusion.weighted_fusion import WeightedFusion
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.retrieval.dense_stub import DenseStubRetriever
from src.infrastructure.retrieval.sbert_retriever import SBERTRetriever
from src.infrastructure.retrieval.sparse_stub import SparseStubRetriever

# ── page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid Retrieval System",
    page_icon="🔍",
    layout="wide",
)

# ── custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }

    /* Cards for results */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }

    .result-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .score-hybrid  { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .score-sparse  { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }
    .score-dense   { background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; }

    /* Metric card */
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85em;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-active { background: #10b981; }
    .status-stub   { background: #f59e0b; }

    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        margin-bottom: 0;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 1.05em;
    }
</style>
""", unsafe_allow_html=True)

# ── component loading (cached) ───────────────────────────────────────────


@st.cache_resource
def load_components():
    """Load indexes and wire up the orchestrator (cached across reruns)."""
    settings = Settings()
    loader = JsonFileDatasetLoader(settings.data_dir)
    store = InMemoryCorpusStore()

    for doc in loader.load_documents():
        store.add(doc)

    # Sparse retriever
    bm25_path = settings.indexes_dir / "lexical" / "bm25.pkl"
    sparse_status = "stub"
    if bm25_path.exists():
        sparse = BM25Retriever()
        sparse.load_index(bm25_path)
        sparse_status = "bm25"
    else:
        sparse = SparseStubRetriever(store)

    # Dense retriever
    sbert_path = settings.indexes_dir / "dense" / "sbert_index.pkl"
    dense_status = "stub"
    if sbert_path.exists():
        dense = SBERTRetriever()
        dense.load_index(sbert_path)
        dense_status = "sbert"
    else:
        dense = DenseStubRetriever(store)

    return {
        "settings": settings,
        "loader": loader,
        "store": store,
        "sparse": sparse,
        "dense": dense,
        "sparse_status": sparse_status,
        "dense_status": dense_status,
    }


def get_orchestrator(components: dict, alpha: float) -> Orchestrator:
    fusion = WeightedFusion(alpha=alpha)
    return Orchestrator(components["sparse"], components["dense"], fusion)


# ── load ──────────────────────────────────────────────────────────────────

components = load_components()
settings = components["settings"]

# ── header ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔍 Hybrid Retrieval System</h1>
    <p>BM25 + SBERT · Score Fusion · Interactive Exploration</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mode = st.selectbox("Retrieval Mode", ["hybrid", "sparse", "dense"], index=0)
    alpha = st.slider("Fusion α (1=sparse, 0=dense)", 0.0, 1.0, 0.5, 0.05)
    top_k = st.slider("Top-K Results", 1, 20, 10)

    st.markdown("---")
    st.markdown("### 📊 System Status")

    sparse_active = components["sparse_status"] == "bm25"
    dense_active = components["dense_status"] == "sbert"

    s_dot = "status-active" if sparse_active else "status-stub"
    d_dot = "status-active" if dense_active else "status-stub"
    s_label = "BM25 (real)" if sparse_active else "Jaccard (stub)"
    d_label = "SBERT (real)" if dense_active else "Hash (stub)"

    st.markdown(f'<span class="status-dot {s_dot}"></span> Sparse: **{s_label}**', unsafe_allow_html=True)
    st.markdown(f'<span class="status-dot {d_dot}"></span> Dense: **{d_label}**', unsafe_allow_html=True)
    st.markdown(f"📄 Corpus: **{len(components['store'])} docs**")

    st.markdown("---")
    st.markdown("### 🛠️ Quick Guide")
    st.markdown("""
    1. Enter a query in the **Search** tab
    2. Adjust α and mode in the sidebar
    3. Expand results for score breakdown
    4. Switch to **Evaluate** to run metrics
    """)

# ── tabs ──────────────────────────────────────────────────────────────────

tab_search, tab_evaluate = st.tabs(["🔎 Search", "📈 Evaluate"])

# ═══════════════════════════════════════════════════════════════════════════
# SEARCH TAB
# ═══════════════════════════════════════════════════════════════════════════

with tab_search:
    query = st.text_input(
        "Enter your query",
        placeholder="e.g. mRNA vaccines are effective against COVID-19",
        key="search_query",
    )

    if query:
        orchestrator = get_orchestrator(components, alpha)

        if mode == "hybrid":
            details = orchestrator.search_detailed(query, top_k=top_k)
            results = details["results"]
        else:
            results = orchestrator.search(query, top_k=top_k, mode=mode)
            details = None

        if not results:
            st.info("No results found.")
        else:
            st.markdown(f"**{len(results)} results** · mode=`{mode}` · α=`{alpha}`")

            for rank, sd in enumerate(results, 1):
                # Look up doc title if available
                doc = components["store"].get(sd.doc_id)
                title = doc.title if doc else sd.doc_id

                source_class = f"score-{sd.source}"
                st.markdown(f"""
                <div class="result-card">
                    <strong>#{rank}</strong> &nbsp;
                    <code>{sd.doc_id}</code> &nbsp;
                    <span class="score-badge {source_class}">{sd.score:.4f}</span>
                    &nbsp; <span style="color:#d1d5db">{title}</span>
                </div>
                """, unsafe_allow_html=True)

                # Explainability expander (only in hybrid mode)
                if details:
                    with st.expander(f"📐 Score breakdown for {sd.doc_id}"):
                        c1, c2, c3 = st.columns(3)

                        s_raw = details["sparse_raw"].get(sd.doc_id, 0.0)
                        d_raw = details["dense_raw"].get(sd.doc_id, 0.0)
                        s_norm = details["sparse_norm"].get(sd.doc_id, 0.0)
                        d_norm = details["dense_norm"].get(sd.doc_id, 0.0)

                        with c1:
                            st.metric("Sparse Raw", f"{s_raw:.4f}")
                            st.metric("Sparse Norm", f"{s_norm:.4f}")
                        with c2:
                            st.metric("Dense Raw", f"{d_raw:.4f}")
                            st.metric("Dense Norm", f"{d_norm:.4f}")
                        with c3:
                            st.metric("Hybrid Score", f"{sd.score:.4f}")
                            st.caption(f"α·sparse + (1−α)·dense")
                            st.caption(f"{alpha:.2f}×{s_norm:.4f} + {1-alpha:.2f}×{d_norm:.4f}")

                        if doc:
                            st.markdown("**Document text:**")
                            st.caption(doc.text[:300] + ("…" if len(doc.text) > 300 else ""))

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE TAB
# ═══════════════════════════════════════════════════════════════════════════

with tab_evaluate:
    st.markdown("### Run Evaluation")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_mode = st.selectbox("Evaluation Mode", ["hybrid", "sparse", "dense"], key="eval_mode")
    with eval_col2:
        k_input = st.text_input("k-values (comma-separated)", "5, 10, 20", key="k_vals")

    if st.button("🚀 Run Evaluation", type="primary"):
        try:
            k_values = [int(k.strip()) for k in k_input.split(",")]
        except ValueError:
            st.error("Please enter valid comma-separated integers for k-values.")
            st.stop()

        orchestrator = get_orchestrator(components, alpha)
        loader = components["loader"]
        queries = loader.load_queries()
        qrels = loader.load_qrels()

        if not queries:
            st.warning("No queries found. Run `build-stub-index` first.")
            st.stop()

        # Build run
        from src.domain.entities import ScoredDocument
        run = {}
        for q in queries:
            run[q.query_id] = orchestrator.search(q.claim, top_k=max(k_values) * 2, mode=eval_mode)

        qrels_dict = {qr.query_id: qr.relevant_doc_ids for qr in qrels}

        engine = MetricsEngine()
        metrics = engine.compute(run, qrels_dict, k_values)
        metrics["mode"] = eval_mode
        metrics["alpha"] = alpha
        metrics["num_queries"] = len(queries)

        # Display metric cards
        st.markdown("### 📊 Results")
        for metric_name in ("Precision", "Recall", "NDCG"):
            cols = st.columns(len(k_values))
            for i, k in enumerate(k_values):
                key = f"{metric_name}@{k}"
                val = metrics.get(key, 0.0)
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val:.4f}</div>
                        <div class="metric-label">{metric_name}@{k}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("")  # spacing

        # Full results table
        st.markdown("### Detailed Metrics")
        table_data = {}
        for k in k_values:
            table_data[f"@{k}"] = {
                "Precision": metrics.get(f"Precision@{k}", 0.0),
                "Recall": metrics.get(f"Recall@{k}", 0.0),
                "NDCG": metrics.get(f"NDCG@{k}", 0.0),
            }
        st.dataframe(table_data, use_container_width=True)

        # Export
        st.markdown("### 📥 Export")
        json_str = json.dumps(metrics, indent=2)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download metrics JSON",
            data=json_str,
            file_name=f"metrics_{eval_mode}_{ts}.json",
            mime="application/json",
        )

        # Also save to runs/
        runs_path = settings.runs_dir / f"metrics_{ts}.json"
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(runs_path, "w") as f:
            f.write(json_str)
        st.caption(f"Also saved to `{runs_path}`")
