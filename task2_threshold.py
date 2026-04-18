"""
Task 2 – Threshold Sensitivity Analysis
========================================
Sweeps similarity thresholds and computes shared / conflicting / gap / CAS
for the GPT-4o obligation graphs (Basel III vs RBI).
"""

import io, sys, os

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from src.metrics import compute_cas

# ── paths ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "graphs" / "gpt_4o"
OUT_DIR = Path(__file__).parent / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. load graphs ─────────────────────────────────────────────────────
print("Loading GPT-4o obligation graphs ...", flush=True)
g_basel = nx.read_graphml(str(DATA_DIR / "basel3_graph.graphml"))
g_regbi = nx.read_graphml(str(DATA_DIR / "regbi_graph.graphml"))

def obligation_texts(G):
    """Return list of non-empty obligation texts."""
    texts = []
    for _, d in G.nodes(data=True):
        if d.get("type") == "obligation" and d.get("text", "").strip():
            texts.append(d["text"].strip())
    return texts

texts_a = obligation_texts(g_basel)
texts_b = obligation_texts(g_regbi)
print(f"  Basel III obligations : {len(texts_a)}", flush=True)
print(f"  RBI obligations      : {len(texts_b)}", flush=True)

# ── 2. encode ──────────────────────────────────────────────────────────
print("Encoding with all-MiniLM-L6-v2 ...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
emb_a = model.encode(texts_a, normalize_embeddings=True, show_progress_bar=True)
emb_b = model.encode(texts_b, normalize_embeddings=True, show_progress_bar=True)

# ── 3. single matrix multiply ─────────────────────────────────────────
print("Computing similarity matrix ...", flush=True)
sim_matrix = emb_a @ emb_b.T  # shape (len_a, len_b)
total_pairs = sim_matrix.size  # len_a * len_b

# ── 4. threshold sweep ────────────────────────────────────────────────
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
rows = []

for t in thresholds:
    # shared: pairs >= threshold
    shared_mask = sim_matrix >= t
    shared_count = int(shared_mask.sum())

    # conflicting: 0.3 <= sim < threshold
    conflict_mask = (sim_matrix >= 0.3) & (sim_matrix < t)
    conflict_count = int(conflict_mask.sum())

    # gap nodes: obligation nodes in EITHER doc with NO cross-doc pair >= threshold
    has_match_a = shared_mask.any(axis=1)  # shape (len_a,)
    has_match_b = shared_mask.any(axis=0)  # shape (len_b,)
    gap_count = int((~has_match_a).sum() + (~has_match_b).sum())

    # CAS via compute_cas
    idxs = np.where(shared_mask)
    alignments = [{"similarity": float(sim_matrix[i, j])} for i, j in zip(idxs[0], idxs[1])]
    cas_result = compute_cas(alignments, total_pairs)
    cas_val = cas_result["value"]

    rows.append({
        "threshold": t,
        "shared_pairs": shared_count,
        "conflicting_pairs": conflict_count,
        "gap_nodes": gap_count,
        "CAS": cas_val,
    })

# ── 5. save CSV ────────────────────────────────────────────────────────
csv_path = OUT_DIR / "threshold_sensitivity.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("threshold,shared_pairs,conflicting_pairs,gap_nodes,CAS\n")
    for r in rows:
        f.write(f"{r['threshold']:.2f},{r['shared_pairs']},{r['conflicting_pairs']},{r['gap_nodes']},{r['CAS']:.4f}\n")

print(f"\nResults saved to {csv_path}", flush=True)

# ── 6. pretty-print table ─────────────────────────────────────────────
hdr = f"{'Threshold':>10} {'Shared':>10} {'Conflict':>10} {'Gaps':>8} {'CAS':>8}"
print("\n" + "=" * len(hdr), flush=True)
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for r in rows:
    print(f"{r['threshold']:>10.2f} {r['shared_pairs']:>10,} {r['conflicting_pairs']:>10,} {r['gap_nodes']:>8,} {r['CAS']:>8.4f}", flush=True)
print("=" * len(hdr), flush=True)
print("Done.", flush=True)
