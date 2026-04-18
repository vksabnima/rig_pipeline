#!/usr/bin/env python3
"""
Recompute all 5 metrics per model with proper per-model CAS and HRR.

CAS: Cross-doc alignment computed from each model's own graph obligation texts.
HRR: Adversarial samples re-parsed through each model (sampled for speed).
"""

import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    PROCESSED_DIR, GRAPHS_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR,
    EMBEDDING_MODEL, SIMILARITY_THRESHOLD, RANDOM_SEED,
)
from src.adversarial_tester import (
    generate_adversarial_samples, PERTURBATION_FUNCTIONS,
    ADVERSARIAL_PERTURBATION_TYPES,
)
from src.metrics import compute_rcc, compute_oal, compute_rif, compute_cas, compute_hrr

GRN = "\033[92m"; CYN = "\033[96m"; BLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"

DOCS = ["basel3", "regbi"]
MODELS = ["gpt-4o", "claude-sonnet", "llama3.2"]
TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]

PARSE_PROMPT = """Extract a JSON obligation tuple from this regulatory clause.
Fields: subject, obligation, condition, threshold, deadline, exception (null if absent).
Return ONLY JSON.

Clause: {text}"""

ADV_REPARSE_SAMPLE = 20  # adversarial samples to re-parse per model for HRR


# =============================================================================
# Per-model CAS: alignment from each model's own graphs
# =============================================================================

def compute_per_model_cas(model_name: str) -> dict:
    """Compute CAS using this model's graph obligation texts."""
    import networkx as nx
    from sentence_transformers import SentenceTransformer

    model_dir = GRAPHS_DIR / model_name.replace("-", "_")
    graphs = {}
    for doc_id in DOCS:
        gpath = model_dir / f"{doc_id}_graph.graphml"
        if gpath.exists():
            graphs[doc_id] = nx.read_graphml(str(gpath))

    if len(graphs) < 2:
        return {"alignments": [], "cas": 0.0, "num_alignments": 0}

    # Use obligation_desc (model's extracted text) if available, else fall back to text
    obls_by_doc = {}
    for doc_id, G in graphs.items():
        obls_by_doc[doc_id] = []
        for node, data in G.nodes(data=True):
            if data.get("type") == "obligation":
                # Prefer model-specific obligation_desc over original text
                obl_text = data.get("obligation_desc", "")
                if not obl_text or obl_text == "":
                    obl_text = data.get("text", "")
                if obl_text:
                    obls_by_doc[doc_id].append({"node_id": node, "text": obl_text})

    st_model = SentenceTransformer(EMBEDDING_MODEL)

    doc_ids = list(obls_by_doc.keys())
    alignments = []

    for di in range(len(doc_ids)):
        for dj in range(di + 1, len(doc_ids)):
            da, db = doc_ids[di], doc_ids[dj]
            obls_a, obls_b = obls_by_doc[da], obls_by_doc[db]
            if not obls_a or not obls_b:
                continue

            emb_a = st_model.encode([o["text"] for o in obls_a],
                                     normalize_embeddings=True, show_progress_bar=False)
            emb_b = st_model.encode([o["text"] for o in obls_b],
                                     normalize_embeddings=True, show_progress_bar=False)

            sim_matrix = emb_a @ emb_b.T
            rows, cols = np.where(sim_matrix >= SIMILARITY_THRESHOLD)

            for r, c in zip(rows, cols):
                alignments.append({
                    "doc_a": da, "doc_b": db,
                    "similarity": round(float(sim_matrix[r, c]), 4),
                })

    total_pairs = len(obls_by_doc.get(doc_ids[0], [])) * len(obls_by_doc.get(doc_ids[1], []))
    cas_result = compute_cas(alignments, total_pairs)

    return {
        "cas": cas_result["value"],
        "num_alignments": len(alignments),
        "total_pairs": total_pairs,
    }


# =============================================================================
# Per-model HRR: adversarial re-parse through each model
# =============================================================================

def reparse_with_gpt4o(texts: list[str]) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []
    for text in texts:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": PARSE_PROMPT.replace("{text}", "")},
                    {"role": "user", "content": text},
                ],
                temperature=0.0, seed=RANDOM_SEED,
                response_format={"type": "json_object"},
            )
            results.append(json.loads(resp.choices[0].message.content))
        except:
            results.append({f: None for f in TUPLE_FIELDS})
    return results


def reparse_with_claude(texts: list[str]) -> list[dict]:
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    for text in texts:
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": PARSE_PROMPT.format(text=text)}],
            )
            content = resp.content[0].text
            s, e = content.find("{"), content.rfind("}") + 1
            results.append(json.loads(content[s:e]) if s >= 0 and e > s else {f: None for f in TUPLE_FIELDS})
        except:
            results.append({f: None for f in TUPLE_FIELDS})
    return results


def reparse_with_llama(texts: list[str]) -> list[dict]:
    import requests
    results = []
    for text in texts:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": PARSE_PROMPT.format(text=text[:500]),
                    "stream": False,
                    "options": {"temperature": 0.0, "seed": RANDOM_SEED, "num_predict": 200},
                },
                timeout=600,
            )
            content = resp.json().get("response", "{}")
            m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            results.append(json.loads(m.group()) if m else {f: None for f in TUPLE_FIELDS})
        except:
            results.append({f: None for f in TUPLE_FIELDS})
    return results


def tuple_similarity(t1: dict, t2: dict) -> float:
    """Compute field-level match between two tuples."""
    matches = 0
    total = 0
    for f in TUPLE_FIELDS:
        v1, v2 = t1.get(f), t2.get(f)
        if v1 is None and v2 is None:
            continue  # Both absent — skip
        total += 1
        if v1 is not None and v2 is not None:
            # Check if they share content
            s1, s2 = str(v1).lower().strip(), str(v2).lower().strip()
            if s1 == s2 or s1 in s2 or s2 in s1:
                matches += 1
    return matches / max(total, 1)


def compute_per_model_hrr(model_name: str, parsed_tuples: list[dict]) -> dict:
    """Compute HRR by re-parsing adversarial samples through the model."""
    import random
    random.seed(RANDOM_SEED)

    # Select clauses that have non-empty tuples for meaningful testing
    good_tuples = [t for t in parsed_tuples
                   if sum(1 for f in TUPLE_FIELDS if t.get("tuple", {}).get(f) is not None) >= 2]

    if len(good_tuples) < 5:
        # Not enough tuples — return baseline HRR from structural analysis
        samples = generate_adversarial_samples(parsed_tuples)
        from src.adversarial_tester import evaluate_robustness
        results = evaluate_robustness(parsed_tuples, samples)
        return compute_hrr(results, samples)

    # Sample clauses for adversarial testing
    sample = random.sample(good_tuples, min(ADV_REPARSE_SAMPLE, len(good_tuples)))

    # Generate one perturbation per type for each sampled clause
    preserving_results = []  # (original_tuple, perturbed_tuple, is_same)
    altering_results = []

    reparse_fn = {
        "gpt-4o": reparse_with_gpt4o,
        "claude-sonnet": reparse_with_claude,
        "llama3.2": reparse_with_llama,
    }[model_name]

    # Collect all texts to parse (originals + perturbations)
    original_texts = [s.get("text", "") for s in sample]
    perturbation_pairs = []  # (original_idx, perturbed_text, is_preserving)

    for idx, clause in enumerate(sample):
        text = clause.get("text", "")
        for ptype in ["synonym_swap", "negation_injection", "threshold_mutation"]:
            fn = PERTURBATION_FUNCTIONS.get(ptype)
            if fn:
                result = fn(text)
                if result.get("changes"):
                    perturbation_pairs.append((idx, result["text"], result["semantic_preserving"]))

    print(f"    Re-parsing {len(original_texts)} originals + {len(perturbation_pairs)} perturbations")

    # Parse originals
    original_tuples = reparse_fn(original_texts)

    # Parse perturbations
    perturbed_texts = [p[1] for p in perturbation_pairs]
    if perturbed_texts:
        perturbed_tuples = reparse_fn(perturbed_texts)
    else:
        perturbed_tuples = []

    # Compare original vs perturbed tuples
    for i, (orig_idx, _, is_preserving) in enumerate(perturbation_pairs):
        if i >= len(perturbed_tuples):
            break
        sim = tuple_similarity(original_tuples[orig_idx], perturbed_tuples[i])
        if is_preserving:
            preserving_results.append(sim)
        else:
            altering_results.append(sim)

    # HRR = 0.5 * stability + 0.5 * detection
    # stability: for preserving perturbations, how similar are outputs? (higher = more stable)
    stability = float(np.mean(preserving_results)) if preserving_results else 0.5
    # detection: for altering perturbations, how different are outputs? (lower sim = better detection)
    detection = 1.0 - float(np.mean(altering_results)) if altering_results else 0.5

    hrr_value = round(0.5 * stability + 0.5 * detection, 4)

    return {
        "metric": "HRR",
        "value": hrr_value,
        "stability_rate": round(stability, 4),
        "detection_rate": round(detection, 4),
        "preserving_samples": len(preserving_results),
        "altering_samples": len(altering_results),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()
    print(f"\n{CYN}{BLD}=== Recomputing All 5 Metrics Per Model ==={RST}\n")

    # Load clauses and parsed tuples per model
    clauses_by_doc = {}
    for doc_id in DOCS:
        with open(PROCESSED_DIR / f"{doc_id}_clauses.json", encoding="utf-8") as f:
            clauses_by_doc[doc_id] = json.load(f).get("clauses", [])

    gold_path = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
    gold_standard = None
    if gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            gold_standard = list(csv.DictReader(f))

    results_table = []

    for model_name in MODELS:
        print(f"{BLD}--- {model_name} ---{RST}")

        # Load tuples
        parsed_by_doc = {}
        for doc_id in DOCS:
            if model_name == "gpt-4o":
                path = PROCESSED_DIR / f"{doc_id}_tuples.json"
            else:
                path = PROCESSED_DIR / f"{doc_id}_tuples_{model_name.replace('-', '_')}.json"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            parsed_by_doc[doc_id] = data.get("parsed_tuples", [])

        # RCC, OAL, RIF per document
        per_doc_metrics = {}
        for doc_id in DOCS:
            tuples = parsed_by_doc[doc_id]
            clauses = clauses_by_doc[doc_id]
            total_obl = len(clauses)

            rcc = compute_rcc(tuples, total_obl)["value"]
            oal = compute_oal(tuples, gold_standard)["value"]
            rif = compute_rif(tuples, clauses)["value"]
            per_doc_metrics[doc_id] = {"RCC": rcc, "OAL": oal, "RIF": rif}
            print(f"  {doc_id}: RCC={rcc:.4f}  OAL={oal:.4f}  RIF={rif:.4f}")

        # CAS (cross-doc, per model)
        print(f"  Computing CAS...")
        cas_result = compute_per_model_cas(model_name)
        cas_val = cas_result["cas"]
        print(f"  CAS={cas_val:.4f} ({cas_result['num_alignments']} alignments / {cas_result['total_pairs']} pairs)")

        # HRR (adversarial re-parse, per model)
        print(f"  Computing HRR (re-parsing {ADV_REPARSE_SAMPLE} adversarial samples)...")
        all_tuples = []
        for doc_id in DOCS:
            all_tuples.extend(parsed_by_doc[doc_id])
        hrr_result = compute_per_model_hrr(model_name, all_tuples)
        hrr_val = hrr_result["value"]
        print(f"  HRR={hrr_val:.4f} (stability={hrr_result.get('stability_rate', '?')}, "
              f"detection={hrr_result.get('detection_rate', '?')})")

        # Build per-doc rows
        for doc_id in DOCS:
            results_table.append({
                "model": model_name,
                "document": doc_id,
                "RCC": round(per_doc_metrics[doc_id]["RCC"], 4),
                "OAL": round(per_doc_metrics[doc_id]["OAL"], 4),
                "RIF": round(per_doc_metrics[doc_id]["RIF"], 4),
                "CAS": round(cas_val, 4),
                "HRR": round(hrr_val, 4),
            })

        print(f"  {CHECK} {model_name} done\n")

    # =========================================================================
    # Print final table
    # =========================================================================
    elapsed = round(time.time() - start, 1)

    print(f"\n{CYN}{BLD}{'='*80}")
    print(f"  COMPLETE RESULTS TABLE (all 5 metrics per model)")
    print(f"{'='*80}{RST}\n")

    header = f"  {'Model':<18} {'Document':<10} {'RCC':>7} {'OAL':>7} {'RIF':>7} {'CAS':>7} {'HRR':>7}"
    print(f"{BLD}{header}{RST}")
    print(f"  {'-'*70}")

    for row in results_table:
        line = (f"  {row['model']:<18} {row['document']:<10} "
                f"{row['RCC']:>7.4f} {row['OAL']:>7.4f} {row['RIF']:>7.4f} "
                f"{row['CAS']:>7.4f} {row['HRR']:>7.4f}")
        print(line)

    print(f"  {'-'*70}")
    for model_name in MODELS:
        model_rows = [r for r in results_table if r["model"] == model_name]
        avg = {m: np.mean([r[m] for r in model_rows]) for m in ["RCC", "OAL", "RIF", "CAS", "HRR"]}
        line = (f"  {model_name:<18} {'AVG':<10} "
                f"{avg['RCC']:>7.4f} {avg['OAL']:>7.4f} {avg['RIF']:>7.4f} "
                f"{avg['CAS']:>7.4f} {avg['HRR']:>7.4f}")
        print(f"{BLD}{line}{RST}")

    # Save CSV
    table_path = OUTPUTS_DIR / "reports" / "multimodel_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "document", "RCC", "OAL", "RIF", "CAS", "HRR"])
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\n  {CHECK} Saved to {table_path.relative_to(Path(__file__).parent)}")

    # Save JSON
    json_path = OUTPUTS_DIR / "reports" / "multimodel_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"  {CHECK} Saved to {json_path.relative_to(Path(__file__).parent)}")

    # Regenerate heatmap
    print(f"\n{BLD}Regenerating Figure 2 heatmap...{RST}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["RCC", "OAL", "RIF", "CAS", "HRR"]
        model_labels = ["GPT-4o", "Claude Sonnet", "Llama3.2"]
        model_keys = ["gpt-4o", "claude-sonnet", "llama3.2"]

        matrix = []
        for key in model_keys:
            rows = [r for r in results_table if r["model"] == key]
            matrix.append([np.mean([r[m] for r in rows]) for m in metrics])
        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=12)

        for i in range(len(model_labels)):
            for j in range(len(metrics)):
                val = matrix[i, j]
                color = "white" if val < 0.3 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        ax.set_title("Figure 2: RIG Metrics by Model (Cross-Document Average)",
                     fontsize=14, fontweight="bold", pad=15)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Score", fontsize=11)
        plt.tight_layout()

        fig_path = OUTPUTS_DIR / "graphs" / "figure2_heatmap.png"
        fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  {CHECK} Figure 2 saved to {fig_path.relative_to(Path(__file__).parent)}")
    except Exception as e:
        print(f"  [ERROR] Heatmap: {e}")

    print(f"\n{DIM}  Completed in {elapsed}s{RST}\n")


if __name__ == "__main__":
    main()
