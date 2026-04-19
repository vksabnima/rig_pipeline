#!/usr/bin/env python3
"""
After extract_qwen.py finishes, compute the metrics needed to slot Qwen
into Table 2 and Table 4 of the paper:
  - RCC, OAL (Branch B / Eq 3), RIF (0.4/0.3/0.3 with gold)
  - CAS for basel3<->regbi using Qwen's per-model graphs
  - Bootstrap 95% CI on RIF using the same protocol as bootstrap_reviewed.py
No additional API calls (sentence-transformer is local; bootstrap is offline).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.config import (PROCESSED_DIR, GRAPHS_DIR, ANNOTATIONS_DIR,
                            OUTPUTS_DIR, EMBEDDING_MODEL, SIMILARITY_THRESHOLD,
                            RANDOM_SEED)
from src.metrics import compute_rcc, compute_oal, compute_cas
from human_review import compute_rif_with_gold
from bootstrap_reviewed import compute_rif_single

np.random.seed(RANDOM_SEED)

DOCS = ["basel3", "regbi", "gdpr"]
MODEL = "qwen"

def load(doc):
    with open(PROCESSED_DIR / f"{doc}_clauses.json", encoding="utf-8") as f:
        clauses = json.load(f)["clauses"]
    with open(PROCESSED_DIR / f"{doc}_tuples_{MODEL}.json", encoding="utf-8") as f:
        tuples = json.load(f)["parsed_tuples"]
    return clauses, tuples

def main():
    # ── Load gold ─────────────────────────────────────────────────────────
    with open(ANNOTATIONS_DIR / "gold_standard_reviewed.csv", encoding="utf-8") as f:
        basel_gold = list(csv.DictReader(f))
    gdpr_gold_path = ANNOTATIONS_DIR / "gdpr_gold_standard.csv"
    gdpr_gold = []
    if gdpr_gold_path.exists():
        with open(gdpr_gold_path, encoding="utf-8") as f:
            gdpr_gold = list(csv.DictReader(f))

    results = {}
    for doc in DOCS:
        clauses, tuples = load(doc)
        rcc = compute_rcc(tuples, len(clauses))["value"]
        oal = compute_oal(tuples, None)["value"]   # Branch B (Eq 3)
        gold = gdpr_gold if doc == "gdpr" else basel_gold
        rif = compute_rif_with_gold(tuples, clauses, gold)
        # Bootstrap CI (only for basel3, regbi where we have a gold protocol)
        ci = None
        if doc != "gdpr" and basel_gold:
            gold_lookup = {g["clause_id"]: g for g in basel_gold}
            per_clause = []
            for t, c in zip(tuples, clauses):
                cid = t.get("clause_id", "")
                per_clause.append(compute_rif_single(t, c.get("text", ""),
                                                     gold_lookup.get(cid, {})))
            arr = np.array(per_clause)
            sample_n = int(0.8 * len(arr))
            boot = []
            for _ in range(1000):
                idx = np.random.choice(len(arr), sample_n, replace=True)
                boot.append(float(np.mean(arr[idx])))
            ci = (round(float(np.percentile(boot, 2.5)), 4),
                  round(float(np.percentile(boot, 97.5)), 4))
        results[doc] = {"RCC": round(rcc, 4), "OAL": round(oal, 4),
                        "RIF": round(rif, 4), "RIF_CI": ci, "n": len(tuples)}
        ci_str = f"[{ci[0]}, {ci[1]}]" if ci else "—"
        print(f"  Qwen / {doc}: RCC={rcc:.4f}  OAL={oal:.4f}  RIF={rif:.4f}  CI={ci_str}")

    # ── CAS: basel3<->regbi using Qwen graphs ──────────────────────────────
    print("\n  Computing CAS (basel3 <-> regbi)...")
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    def obls_with_text(graph_path):
        G = nx.read_graphml(str(graph_path))
        out = []
        for n, d in G.nodes(data=True):
            if d.get("type") == "obligation":
                txt = d.get("obligation_desc", "") or d.get("text", "")
                if txt:
                    out.append(txt)
        return out

    obls_b = obls_with_text(GRAPHS_DIR / MODEL / "basel3_graph.graphml")
    obls_r = obls_with_text(GRAPHS_DIR / MODEL / "regbi_graph.graphml")
    print(f"  Basel obls: {len(obls_b)}, Reg BI obls: {len(obls_r)}")

    emb_a = model.encode(obls_b, normalize_embeddings=True, show_progress_bar=False)
    emb_b = model.encode(obls_r, normalize_embeddings=True, show_progress_bar=False)
    sim = emb_a @ emb_b.T
    aligned = [(r, c, float(sim[r, c]))
               for r, c in zip(*np.where(sim >= SIMILARITY_THRESHOLD))]
    print(f"  Aligned pairs at tau={SIMILARITY_THRESHOLD}: {len(aligned)}")

    if aligned:
        precision = float(np.mean([s for _, _, s in aligned]))
        total_pairs = len(obls_b) * len(obls_r)
        recall = len(aligned) / max(total_pairs, 1)
        cas = round(0.6 * precision + 0.4 * min(recall, 1.0), 4)
    else:
        cas = 0.0
        precision = recall = 0.0
    results["CAS_basel3_regbi"] = {"cas": cas, "precision": round(precision, 4),
                                    "recall": round(recall, 6),
                                    "aligned": len(aligned)}
    for d in ("basel3", "regbi"):
        results[d]["CAS"] = cas

    # GDPR pair CAS using Qwen graphs (analogous to gdpr_sonnet's path)
    print("  Computing CAS (basel3+regbi <-> gdpr)...")
    obls_g = obls_with_text(GRAPHS_DIR / MODEL / "gdpr_graph.graphml")
    other = obls_b + obls_r
    print(f"  GDPR obls: {len(obls_g)}, other (basel+reg): {len(other)}")
    emb_g = model.encode(obls_g, normalize_embeddings=True, show_progress_bar=False)
    emb_o = model.encode(other, normalize_embeddings=True, show_progress_bar=False)
    simg = emb_g @ emb_o.T
    aligned_g = int(np.sum(simg >= SIMILARITY_THRESHOLD))
    if aligned_g > 0:
        precision_g = float(np.mean(simg[simg >= SIMILARITY_THRESHOLD]))
        recall_g = aligned_g / max(len(obls_g) * len(other), 1)
        cas_g = round(0.6 * precision_g + 0.4 * min(recall_g, 1.0), 4)
    else:
        cas_g = 0.0
    results["CAS_gdpr_union"] = {"cas": cas_g, "aligned": aligned_g,
                                  "n_other": len(other), "n_gdpr": len(obls_g)}
    results["gdpr"]["CAS"] = cas_g

    print(f"\n  Qwen CAS basel<->reg = {results['CAS_basel3_regbi']['cas']}")
    print(f"  Qwen CAS gdpr-union  = {results['CAS_gdpr_union']['cas']}")

    out_path = OUTPUTS_DIR / "reports" / "qwen_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
