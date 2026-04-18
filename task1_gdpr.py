"""
Task 1 — GDPR Pipeline
-----------------------
Downloads the GDPR PDF, runs the full RIG pipeline on it,
then performs 3-way cross-document alignment with Basel III and Reg BI.
"""

import io, sys, os

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import shutil
import requests
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from config.config import (
    RAW_DIR, PROCESSED_DIR, GRAPHS_DIR, GRAPH_FORMAT,
    SIMILARITY_THRESHOLD, EMBEDDING_MODEL, MIN_PDF_SIZE_KB,
)
from src.pdf_extractor import process_pdf
from src.obligation_parser import parse_document
from src.graph_builder import build_graph, _sanitize_graph
from src.metrics import compute_cas

# ── Constants ───────────────────────────────────────────────────────────────
GDPR_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
GDPR_FILENAME = "gdpr.pdf"
GPT4O_GRAPH_DIR = GRAPHS_DIR / "gpt_4o"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download GDPR PDF
# ═════════════════════════════════════════════════════════════════════════════
def download_gdpr() -> bool:
    """Download the GDPR PDF. Returns True on success."""
    print("\n=== Step 1: Download GDPR PDF ===", flush=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / GDPR_FILENAME

    if dest.exists() and dest.stat().st_size > MIN_PDF_SIZE_KB * 1024:
        size_kb = dest.stat().st_size / 1024
        print(f"  [SKIP] {GDPR_FILENAME} already exists ({size_kb:.0f} KB)", flush=True)
        return True

    print(f"  [DOWNLOAD] GDPR: {GDPR_URL}", flush=True)
    try:
        resp = requests.get(GDPR_URL, headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = dest.stat().st_size / 1024
        if size_kb < MIN_PDF_SIZE_KB:
            print(f"  [WARN] {GDPR_FILENAME} is only {size_kb:.1f} KB — may be incomplete", flush=True)
            dest.unlink()
            _print_curl_fallback()
            return False

        print(f"  [OK] {GDPR_FILENAME} — {size_kb:.0f} KB", flush=True)
        return True

    except Exception as e:
        print(f"  [FAIL] GDPR download: {e}", flush=True)
        _print_curl_fallback()
        return False


def _print_curl_fallback():
    dest = RAW_DIR / GDPR_FILENAME
    print(f"\n  Manual fallback — run this command:", flush=True)
    print(f'  curl -L -o "{dest}" -H "User-Agent: {HEADERS["User-Agent"]}" "{GDPR_URL}"', flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run full pipeline on GDPR
# ═════════════════════════════════════════════════════════════════════════════
def run_gdpr_pipeline() -> dict:
    """Extract, parse, and build graph for GDPR. Returns summary dict."""
    print("\n=== Step 2: GDPR Pipeline (Extract -> Parse -> Graph) ===", flush=True)

    # 2a. Extract text and segment
    print("\n--- 2a. PDF Extraction & Segmentation ---", flush=True)
    pdf_path = RAW_DIR / GDPR_FILENAME
    extraction = process_pdf("gdpr", pdf_path)
    clause_count = len(extraction.get("clauses", []))
    print(f"  GDPR obligation clauses extracted: {clause_count}", flush=True)

    # 2b. Parse obligation tuples with GPT-4o
    print("\n--- 2b. Obligation Parsing (GPT-4o) ---", flush=True)
    parsed = parse_document(extraction)
    tuple_count = len(parsed.get("parsed_tuples", []))
    print(f"  GDPR parsed tuples: {tuple_count}", flush=True)

    # 2c. Build knowledge graph
    print("\n--- 2c. Knowledge Graph Construction ---", flush=True)
    G = build_graph(parsed)
    _sanitize_graph(G)

    # Save to gpt_4o subdirectory
    GPT4O_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    gdpr_graph_path = GPT4O_GRAPH_DIR / f"gdpr_graph.{GRAPH_FORMAT}"
    nx.write_graphml(G, str(gdpr_graph_path))
    print(f"  [GRAPH] gdpr", flush=True)
    print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", flush=True)
    print(f"    Saved to {gdpr_graph_path.name}", flush=True)

    return {
        "clause_count": clause_count,
        "tuple_count": tuple_count,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — 3-Way Cross-Document Alignment
# ═════════════════════════════════════════════════════════════════════════════
def get_obligation_texts(G: nx.DiGraph) -> dict:
    """Extract obligation node IDs and their text from a graph."""
    return {
        node: data.get("text", data.get("obligation_desc", ""))
        for node, data in G.nodes(data=True)
        if data.get("type") == "obligation" and data.get("text")
    }


def run_3way_alignment() -> dict:
    """Load all 3 GPT-4o graphs and compute pairwise CAS."""
    print("\n=== Step 3: 3-Way Cross-Document Alignment ===", flush=True)

    # Load graphs
    doc_names = ["basel3", "regbi", "gdpr"]
    graphs = {}
    for name in doc_names:
        gpath = GPT4O_GRAPH_DIR / f"{name}_graph.{GRAPH_FORMAT}"
        if not gpath.exists():
            print(f"  [ERROR] Graph not found: {gpath}", flush=True)
            return {}
        G = nx.read_graphml(str(gpath))
        graphs[name] = G
        n_obls = len(get_obligation_texts(G))
        print(f"  Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {n_obls} obligations", flush=True)

    # Load sentence transformer
    print(f"\n  Loading sentence transformer ({EMBEDDING_MODEL})...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Get obligation texts and embeddings per document
    obls_by_doc = {}
    embs_by_doc = {}
    for name, G in graphs.items():
        texts_dict = get_obligation_texts(G)
        node_ids = list(texts_dict.keys())
        texts = list(texts_dict.values())
        obls_by_doc[name] = {"node_ids": node_ids, "texts": texts}
        if texts:
            embs_by_doc[name] = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            embs_by_doc[name] = np.array([])
        print(f"  Encoded {name}: {len(texts)} obligation embeddings", flush=True)

    # Pairwise alignment
    pairs = [("basel3", "regbi"), ("basel3", "gdpr"), ("regbi", "gdpr")]
    pair_labels = ["Basel3-RegBI", "Basel3-GDPR", "RegBI-GDPR"]

    all_alignments = []
    pair_results = {}

    for (da, db), label in zip(pairs, pair_labels):
        emb_a = embs_by_doc[da]
        emb_b = embs_by_doc[db]

        if emb_a.size == 0 or emb_b.size == 0:
            print(f"\n  {label}: No obligations in one or both documents", flush=True)
            pair_results[label] = {"alignments": [], "cas": 0.0}
            continue

        # Vectorized similarity
        sim_matrix = emb_a @ emb_b.T
        rows, cols = np.where(sim_matrix >= SIMILARITY_THRESHOLD)

        alignments = []
        for r, c in zip(rows, cols):
            alignments.append({
                "doc_a": da,
                "node_a": obls_by_doc[da]["node_ids"][r],
                "text_a": obls_by_doc[da]["texts"][r][:200],
                "doc_b": db,
                "node_b": obls_by_doc[db]["node_ids"][c],
                "text_b": obls_by_doc[db]["texts"][c][:200],
                "similarity": round(float(sim_matrix[r, c]), 4),
            })

        alignments.sort(key=lambda x: x["similarity"], reverse=True)

        # CAS computation
        total_pairs = len(obls_by_doc[da]["texts"]) * len(obls_by_doc[db]["texts"])
        cas_result = compute_cas(alignments, total_pairs)

        pair_results[label] = {
            "alignments": len(alignments),
            "cas": cas_result["value"],
            "precision": cas_result.get("precision", 0.0),
            "recall": cas_result.get("recall", 0.0),
        }
        all_alignments.extend(alignments)

        print(f"\n  {label}:", flush=True)
        print(f"    Aligned pairs: {len(alignments)}", flush=True)
        print(f"    Total possible pairs: {total_pairs}", flush=True)
        print(f"    CAS = {cas_result['value']:.4f} (precision={cas_result.get('precision', 0):.4f}, recall={cas_result.get('recall', 0):.4f})", flush=True)

    return {
        "pair_results": pair_results,
        "total_alignments": len(all_alignments),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Summary
# ═════════════════════════════════════════════════════════════════════════════
def print_summary(pipeline_stats: dict, alignment_stats: dict):
    """Print final summary."""
    print("\n" + "=" * 60, flush=True)
    print("  TASK 1 SUMMARY — GDPR Integration", flush=True)
    print("=" * 60, flush=True)

    print(f"\n  GDPR Clause Count:     {pipeline_stats.get('clause_count', 'N/A')}", flush=True)
    print(f"  GDPR Graph Nodes:      {pipeline_stats.get('nodes', 'N/A')}", flush=True)
    print(f"  GDPR Graph Edges:      {pipeline_stats.get('edges', 'N/A')}", flush=True)

    print(f"\n  Pairwise CAS Scores:", flush=True)
    pair_results = alignment_stats.get("pair_results", {})
    for label, data in pair_results.items():
        print(f"    {label:20s}  CAS = {data['cas']:.4f}  (alignments: {data['alignments']})", flush=True)

    print(f"\n  Total 3-Way Alignment Count: {alignment_stats.get('total_alignments', 0)}", flush=True)
    print("=" * 60, flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1: Download
    ok = download_gdpr()
    if not ok:
        print("\n[ABORT] GDPR download failed. Use the curl command above.", flush=True)
        sys.exit(1)

    # Step 2: Pipeline
    pipeline_stats = run_gdpr_pipeline()

    # Step 3: 3-way alignment
    alignment_stats = run_3way_alignment()

    # Step 4: Summary
    print_summary(pipeline_stats, alignment_stats)
