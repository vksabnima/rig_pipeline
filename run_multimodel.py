#!/usr/bin/env python3
"""
Multi-Model Comparison Run
===========================
Extracts obligation tuples using 3 models, builds separate graphs per model,
computes all 5 RIG metrics per model, and outputs a combined comparison table
plus a Figure 2 heatmap.

Models:
  1. GPT-4o       (cached from prior run)
  2. Claude Sonnet (via Anthropic API)
  3. Llama3       (via Ollama at localhost:11434)
"""

import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from functools import partial

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Override print to always flush
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    ANTHROPIC_API_KEY, PROCESSED_DIR, GRAPHS_DIR, ANNOTATIONS_DIR,
    OUTPUTS_DIR, RANDOM_SEED, EMBEDDING_MODEL, SIMILARITY_THRESHOLD,
)

GRN = "\033[92m"; CYN = "\033[96m"; YLW = "\033[93m"; RED = "\033[91m"
BLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"; CROSS = f"{RED}\u2718{RST}"; WARN = f"{YLW}\u26a0{RST}"

DOCS = ["basel3", "regbi"]
MODELS_TO_RUN = ["gpt-4o", "claude-sonnet", "llama3.2"]

TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]

SYSTEM_PROMPT = """You are a regulatory compliance expert. Extract structured obligation tuples from regulatory text.

For each obligation clause, return a JSON object with these fields:
- subject: The regulated entity or actor
- obligation: The required action or prohibition
- condition: Triggering condition or scope qualifier
- threshold: Quantitative limit or numeric requirement
- deadline: Time constraint or compliance date
- exception: Exemptions or carve-outs

If a field is not present in the text, set it to null.
Return ONLY valid JSON -- no markdown, no explanation."""


# =============================================================================
# Tuple extraction per model
# =============================================================================

def load_clauses(doc_id: str) -> list[dict]:
    with open(PROCESSED_DIR / f"{doc_id}_clauses.json", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("clauses", [])


def load_cached_gpt4o(doc_id: str) -> list[dict]:
    """Load cached GPT-4o tuples."""
    path = PROCESSED_DIR / f"{doc_id}_tuples.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("parsed_tuples", [])


def extract_with_claude(clauses: list[dict], doc_id: str) -> list[dict]:
    """Extract tuples using Claude claude-sonnet-4-20250514."""
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    parsed = []

    for i, clause in enumerate(clauses):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"      clause {i+1}/{len(clauses)}")
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nExtract the obligation tuple from:\n\n{clause['text']}",
                }],
            )
            content = response.content[0].text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                tuple_data = json.loads(content[start:end])
            else:
                tuple_data = {f: None for f in TUPLE_FIELDS}
        except Exception as e:
            if i < 3:
                print(f"      [ERROR] clause {i+1}: {e}")
            tuple_data = {f: None for f in TUPLE_FIELDS}

        parsed.append({**clause, "tuple": tuple_data, "model_used": "claude-sonnet"})

    return parsed


LLAMA3_SAMPLE_SIZE = 50  # Sample per doc — Llama3 on CPU is ~7min/clause

LLAMA3_SHORT_PROMPT = """Extract a JSON obligation tuple from this regulatory clause.
Fields: subject, obligation, condition, threshold, deadline, exception (null if absent).
Return ONLY JSON.

Clause: {text}"""


def extract_with_ollama(clauses: list[dict], doc_id: str) -> list[dict]:
    """Extract tuples using Llama3 via Ollama.

    Samples LLAMA3_SAMPLE_SIZE clauses (CPU inference is very slow).
    Sampled clauses get real inference; rest get null tuples.
    """
    import random
    import requests
    random.seed(RANDOM_SEED)

    # Test connectivity
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        print(f"      Ollama connected, models: {[m['name'] for m in resp.json().get('models', [])]}")
    except Exception as e:
        print(f"      {WARN} Ollama not available: {e}")
        return [{**c, "tuple": {f: None for f in TUPLE_FIELDS}, "model_used": "llama3.2"} for c in clauses]

    # Sample representative clauses
    sample_n = min(LLAMA3_SAMPLE_SIZE, len(clauses))
    sample_indices = set(random.sample(range(len(clauses)), sample_n))
    print(f"      Sampling {sample_n}/{len(clauses)} clauses (CPU inference ~7min/clause)")
    print(f"      Estimated time: ~{sample_n * 7}min for {doc_id}")

    parsed = []
    inferred = 0
    for i, clause in enumerate(clauses):
        if i not in sample_indices:
            parsed.append({**clause, "tuple": {f: None for f in TUPLE_FIELDS}, "model_used": "llama3.2"})
            continue

        inferred += 1
        print(f"      [{inferred}/{sample_n}] clause {i+1}/{len(clauses)}")
        # Truncate text to keep prompt short for 8B model
        text = clause.get("text", "")[:500]
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": LLAMA3_SHORT_PROMPT.format(text=text),
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "seed": RANDOM_SEED,
                        "num_predict": 200,
                    },
                },
                timeout=600,
            )
            resp.raise_for_status()
            content = resp.json().get("response", "{}")
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                tuple_data = json.loads(json_match.group())
            else:
                tuple_data = {f: None for f in TUPLE_FIELDS}
        except Exception as e:
            print(f"        [ERROR] {e}")
            tuple_data = {f: None for f in TUPLE_FIELDS}

        parsed.append({**clause, "tuple": tuple_data, "model_used": "llama3.2"})

    return parsed


# =============================================================================
# Graph building (reuses logic from graph_builder but per-model output dir)
# =============================================================================

def build_model_graph(parsed_tuples: list[dict], doc_id: str, model_name: str):
    """Build and save graph for a specific model's tuples."""
    import networkx as nx

    G = nx.DiGraph()
    G.graph["doc_id"] = doc_id
    G.graph["model"] = model_name

    for entry in parsed_tuples:
        clause_id = entry.get("clause_id", "")
        t = entry.get("tuple", {})

        attrs = {"type": "obligation", "text": entry.get("text", ""), "section": entry.get("section", "")}
        if t.get("obligation") is not None:
            attrs["obligation_desc"] = str(t["obligation"])
        G.add_node(clause_id, **attrs)

        for field, ntype, relation, reverse in [
            ("subject", "subject", "has_obligation", True),
            ("condition", "condition", "triggers", True),
            ("threshold", "threshold", "has_threshold", False),
            ("deadline", "deadline", "has_deadline", False),
            ("exception", "exception", "exempts", True),
        ]:
            val = t.get(field)
            if val:
                nid = f"{ntype.upper()[:4]}:{clause_id}" if field != "subject" else f"SUBJ:{val}"
                G.add_node(nid, type=ntype, label=str(val))
                if reverse:
                    G.add_edge(nid, clause_id, relation=relation)
                else:
                    G.add_edge(clause_id, nid, relation=relation)

    # Co-regulated edges
    subjects = [n for n, d in G.nodes(data=True) if d.get("type") == "subject"]
    for subj in subjects:
        obls = [s for s in G.successors(subj) if G.nodes[s].get("type") == "obligation"]
        for i in range(len(obls)):
            for j in range(i + 1, len(obls)):
                G.add_edge(obls[i], obls[j], relation="co_regulated")

    # Save
    model_dir = GRAPHS_DIR / model_name.replace("-", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{doc_id}_graph.graphml"

    # Sanitize None values
    for _, data in G.nodes(data=True):
        for k in list(data):
            if data[k] is None:
                data[k] = ""
    for _, _, data in G.edges(data=True):
        for k in list(data):
            if data[k] is None:
                data[k] = ""

    nx.write_graphml(G, str(out_path))
    return G


# =============================================================================
# Cross-doc alignment per model
# =============================================================================

def compute_alignment(graphs: dict[str, "nx.DiGraph"]) -> list[dict]:
    """Compute cross-doc alignments using vectorized matrix multiplication."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Group obligations by document
    obls_by_doc = {}
    for doc_id, G in graphs.items():
        obls_by_doc[doc_id] = []
        for node, data in G.nodes(data=True):
            if data.get("type") == "obligation" and data.get("text"):
                obls_by_doc[doc_id].append({"doc_id": doc_id, "node_id": node, "text": data["text"]})

    doc_ids = list(obls_by_doc.keys())
    if len(doc_ids) < 2:
        return []

    # Encode per document and compute cross-doc similarity via matrix multiply
    alignments = []
    for di in range(len(doc_ids)):
        for dj in range(di + 1, len(doc_ids)):
            obls_a = obls_by_doc[doc_ids[di]]
            obls_b = obls_by_doc[doc_ids[dj]]
            if not obls_a or not obls_b:
                continue

            emb_a = model.encode([o["text"] for o in obls_a], normalize_embeddings=True, show_progress_bar=False)
            emb_b = model.encode([o["text"] for o in obls_b], normalize_embeddings=True, show_progress_bar=False)

            # Vectorized cosine similarity matrix
            sim_matrix = emb_a @ emb_b.T
            rows, cols = np.where(sim_matrix >= SIMILARITY_THRESHOLD)

            for r, c in zip(rows, cols):
                alignments.append({
                    "doc_a": obls_a[r]["doc_id"],
                    "node_a": obls_a[r]["node_id"],
                    "text_a": obls_a[r]["text"][:200],
                    "doc_b": obls_b[c]["doc_id"],
                    "node_b": obls_b[c]["node_id"],
                    "text_b": obls_b[c]["text"][:200],
                    "similarity": round(float(sim_matrix[r, c]), 4),
                })

    alignments.sort(key=lambda x: x["similarity"], reverse=True)
    return alignments


# =============================================================================
# Metrics computation (per model)
# =============================================================================

def compute_metrics_for_model(
    model_name: str,
    parsed_by_doc: dict[str, list[dict]],
    clauses_by_doc: dict[str, list[dict]],
    alignments: list[dict],
    adversarial_samples: list[dict],
    adversarial_results: dict,
    gold_standard: list[dict] | None,
) -> dict:
    """Compute all 5 metrics for one model."""
    from src.metrics import compute_rcc, compute_oal, compute_rif, compute_cas, compute_hrr

    all_parsed = []
    all_clauses = []
    for doc_id in DOCS:
        all_parsed.extend(parsed_by_doc.get(doc_id, []))
        all_clauses.extend(clauses_by_doc.get(doc_id, []))

    total_obl = len(all_clauses)

    # Cross-doc pairs
    counts = [len(parsed_by_doc.get(d, [])) for d in DOCS]
    total_pairs = counts[0] * counts[1] if len(counts) >= 2 else 0

    rcc = compute_rcc(all_parsed, total_obl)
    oal = compute_oal(all_parsed, gold_standard)
    rif = compute_rif(all_parsed, all_clauses)
    cas = compute_cas(alignments, total_pairs)
    hrr = compute_hrr(adversarial_results, adversarial_samples)

    return {"RCC": rcc["value"], "OAL": oal["value"], "RIF": rif["value"],
            "CAS": cas["value"], "HRR": hrr["value"]}


# =============================================================================
# Adversarial testing
# =============================================================================

def run_adversarial_for_model(parsed_tuples: list[dict]) -> tuple[dict, list[dict]]:
    from src.adversarial_tester import generate_adversarial_samples, evaluate_robustness
    samples = generate_adversarial_samples(parsed_tuples)
    results = evaluate_robustness(parsed_tuples, samples)
    return results, samples


# =============================================================================
# Figure 2: Heatmap
# =============================================================================

def generate_heatmap(results_table: list[dict]):
    """Generate Figure 2 heatmap: models x metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["RCC", "OAL", "RIF", "CAS", "HRR"]
    model_labels = ["GPT-4o", "Claude Sonnet", "Llama3.2"]

    # Aggregate per model (average across docs)
    model_data = {}
    for row in results_table:
        m = row["model"]
        if m not in model_data:
            model_data[m] = {met: [] for met in metrics}
        for met in metrics:
            model_data[m][met].append(row[met])

    matrix = []
    for m in model_labels:
        key = {"GPT-4o": "gpt-4o", "Claude Sonnet": "claude-sonnet", "Llama3.2": "llama3.2"}[m]
        row = [np.mean(model_data[key][met]) for met in metrics]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=12)

    # Annotate cells
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
    out_path = OUTPUTS_DIR / "graphs" / "figure2_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()

    print(f"\n{CYN}{BLD}{'='*65}")
    print(f"  RIG Pipeline -- Multi-Model Comparison")
    print(f"  Models: GPT-4o | Claude Sonnet | Llama3")
    print(f"{'='*65}{RST}\n")

    # Load extracted clauses (shared across all models)
    clauses_by_doc = {}
    for doc_id in DOCS:
        clauses_by_doc[doc_id] = load_clauses(doc_id)
        print(f"  {CHECK} {doc_id}: {len(clauses_by_doc[doc_id])} clauses loaded")
    print()

    # Load gold standard
    gold_path = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
    gold_standard = None
    if gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            gold_standard = list(csv.DictReader(f))

    # Storage for all results
    all_parsed = {}        # {model: {doc_id: [tuples]}}
    all_graphs = {}        # {model: {doc_id: nx.DiGraph}}
    all_alignments = {}    # {model: [alignments]}
    all_adversarial = {}   # {model: (results, samples)}
    results_table = []

    for model_name in MODELS_TO_RUN:
        print(f"{BLD}{'='*65}")
        print(f"  Model: {model_name}")
        print(f"{'='*65}{RST}")

        all_parsed[model_name] = {}
        all_graphs[model_name] = {}

        for doc_id in DOCS:
            clauses = clauses_by_doc[doc_id]

            # ── Tuple extraction ──
            cache_path = PROCESSED_DIR / f"{doc_id}_tuples_{model_name.replace('-', '_')}.json"

            if model_name == "gpt-4o":
                # Use existing cached GPT-4o tuples
                print(f"  {DIM}[CACHED] {doc_id} / gpt-4o{RST}")
                tuples = load_cached_gpt4o(doc_id)
            elif cache_path.exists():
                print(f"  {DIM}[CACHED] {doc_id} / {model_name}{RST}")
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                tuples = data.get("parsed_tuples", [])
            else:
                print(f"  [EXTRACT] {doc_id} / {model_name} ({len(clauses)} clauses)")
                if model_name == "claude-sonnet":
                    tuples = extract_with_claude(clauses, doc_id)
                elif model_name == "llama3.2":
                    tuples = extract_with_ollama(clauses, doc_id)
                else:
                    tuples = [{**c, "tuple": {f: None for f in TUPLE_FIELDS}, "model_used": model_name} for c in clauses]

                # Cache the results
                cache_data = {"doc_id": doc_id, "model": model_name,
                              "total_clauses": len(clauses), "parsed_tuples": tuples}
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                print(f"    Cached to {cache_path.name}")

            all_parsed[model_name][doc_id] = tuples

            # ── Build graph ──
            print(f"  [GRAPH] {doc_id} / {model_name}")
            G = build_model_graph(tuples, doc_id, model_name)
            all_graphs[model_name][doc_id] = G
            print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # ── Cross-doc alignment ──
        print(f"  [ALIGN] {model_name}")
        alignments = compute_alignment(all_graphs[model_name])
        all_alignments[model_name] = alignments
        print(f"    {len(alignments)} cross-doc alignments")

        # ── Adversarial testing ──
        print(f"  [ADVERSARIAL] {model_name}")
        all_tuples = []
        for doc_id in DOCS:
            all_tuples.extend(all_parsed[model_name][doc_id])
        adv_results, adv_samples = run_adversarial_for_model(all_tuples)
        all_adversarial[model_name] = (adv_results, adv_samples)
        print(f"    {len(adv_samples)} samples generated")

        # ── Metrics ──
        print(f"  [METRICS] {model_name}")
        metrics = compute_metrics_for_model(
            model_name, all_parsed[model_name], clauses_by_doc,
            alignments, adv_samples, adv_results, gold_standard,
        )

        # Per-document rows for the table
        for doc_id in DOCS:
            doc_parsed = {doc_id: all_parsed[model_name][doc_id]}
            doc_clauses = {doc_id: clauses_by_doc[doc_id]}
            doc_metrics = compute_metrics_for_model(
                model_name, doc_parsed, doc_clauses,
                [], [], {}, gold_standard,  # No cross-doc for single doc
            )
            # Use the cross-doc CAS from the combined run
            doc_metrics["CAS"] = metrics["CAS"]
            results_table.append({"model": model_name, "document": doc_id, **doc_metrics})

        print(f"  {CHECK} {model_name} complete\n")

    # =========================================================================
    # Output combined results table
    # =========================================================================
    elapsed = round(time.time() - start, 1)

    print(f"\n{CYN}{BLD}{'='*80}")
    print(f"  COMBINED RESULTS TABLE")
    print(f"{'='*80}{RST}\n")

    header = f"  {'Model':<18} {'Document':<10} {'RCC':>7} {'OAL':>7} {'RIF':>7} {'CAS':>7} {'HRR':>7}"
    print(f"{BLD}{header}{RST}")
    print(f"  {'-'*70}")

    for row in results_table:
        line = (f"  {row['model']:<18} {row['document']:<10} "
                f"{row['RCC']:>7.4f} {row['OAL']:>7.4f} {row['RIF']:>7.4f} "
                f"{row['CAS']:>7.4f} {row['HRR']:>7.4f}")
        print(line)

    # Model averages
    print(f"  {'-'*70}")
    for model_name in MODELS_TO_RUN:
        model_rows = [r for r in results_table if r["model"] == model_name]
        avg = {m: np.mean([r[m] for r in model_rows]) for m in ["RCC", "OAL", "RIF", "CAS", "HRR"]}
        line = (f"  {model_name:<18} {'AVG':<10} "
                f"{avg['RCC']:>7.4f} {avg['OAL']:>7.4f} {avg['RIF']:>7.4f} "
                f"{avg['CAS']:>7.4f} {avg['HRR']:>7.4f}")
        print(f"{BLD}{line}{RST}")

    # Save table as CSV
    table_path = OUTPUTS_DIR / "reports" / "multimodel_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "document", "RCC", "OAL", "RIF", "CAS", "HRR"])
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\n  {CHECK} Table saved to {table_path.relative_to(Path(__file__).parent)}")

    # Save full results JSON
    full_path = OUTPUTS_DIR / "reports" / "multimodel_results.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"  {CHECK} Full results saved to {full_path.relative_to(Path(__file__).parent)}")

    # =========================================================================
    # Generate Figure 2 heatmap
    # =========================================================================
    print(f"\n{BLD}Generating Figure 2 heatmap...{RST}")
    try:
        fig_path = generate_heatmap(results_table)
        print(f"  {CHECK} Figure 2 saved to {fig_path.relative_to(Path(__file__).parent)}")
    except Exception as e:
        print(f"  {CROSS} Heatmap generation failed: {e}")
        print(f"  Install matplotlib: pip install matplotlib")

    print(f"\n{DIM}  Total elapsed: {elapsed}s{RST}\n")


if __name__ == "__main__":
    main()
