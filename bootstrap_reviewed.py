#!/usr/bin/env python3
"""
Bootstrap CI on RIF using validated gold_standard_reviewed.csv.
n=1000, 80% sample with replacement, 95% CI.
"""

import csv
import io
import json
import os
import random
import sys
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

from config.config import PROCESSED_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR

random.seed(42)
np.random.seed(42)

TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]
GT_FIELDS = ["subject_gt", "obligation_gt", "condition_gt", "threshold_gt", "deadline_gt", "exception_gt"]
REGULATORY_KEYWORDS = {
    "shall", "must", "required", "prohibited", "ensure", "comply",
    "obligation", "mandatory", "limit", "minimum", "maximum", "capital",
    "risk", "exposure", "deadline", "within", "before", "after",
}

MODELS = ["gpt-4o", "claude-sonnet", "llama3.2"]
DOCS = ["basel3", "regbi"]
N_BOOT = 1000
SAMPLE_FRAC = 0.80

GRN = "\033[92m"; CYN = "\033[96m"; BLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"


def compute_rif_single(parsed_tuple: dict, original_text: str, gold_row: dict) -> float:
    """Compute RIF for a single clause using gold-aligned formula."""
    t = parsed_tuple.get("tuple", {}) if "tuple" in parsed_tuple else parsed_tuple

    non_null = sum(1 for f in TUPLE_FIELDS if t.get(f) is not None)
    completeness = non_null / len(TUPLE_FIELDS)

    orig_words = set(original_text.lower().split())
    orig_kw = orig_words & REGULATORY_KEYWORDS

    if orig_kw:
        tuple_text = " ".join(str(v).lower() for v in t.values() if v is not None)
        retained = sum(1 for kw in orig_kw if kw in tuple_text)
        retention = retained / len(orig_kw)
    else:
        retention = 1.0

    # Gold alignment bonus
    if gold_row:
        match_bonus = 0.0
        match_count = 0
        for tf, gf in zip(TUPLE_FIELDS, GT_FIELDS):
            tv = t.get(tf)
            gv = gold_row.get(gf)
            if tv is not None and gv is not None and str(gv).strip():
                match_count += 1
                if str(tv).lower().strip() in str(gv).lower().strip() or \
                   str(gv).lower().strip() in str(tv).lower().strip():
                    match_bonus += 1
        if match_count > 0:
            gold_alignment = match_bonus / match_count
            return 0.4 * completeness + 0.3 * retention + 0.3 * gold_alignment

    return 0.5 * completeness + 0.5 * retention


def main():
    print(f"\n{CYN}{BLD}=== Bootstrap CI on RIF (Validated Gold Standard, n={N_BOOT}) ==={RST}\n")

    # Load reviewed gold standard
    gold_path = ANNOTATIONS_DIR / "gold_standard_reviewed.csv"
    with open(gold_path, encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))

    reviewed = [r for r in gold_rows if r.get("human_reviewed", "").upper() == "TRUE"]
    print(f"  Gold standard: {len(reviewed)} rows (human_reviewed=TRUE)")

    gold_lookup = {r["clause_id"]: r for r in reviewed}

    # Load clauses
    clauses_by_doc = {}
    for doc_id in DOCS:
        cpath = PROCESSED_DIR / f"{doc_id}_clauses.json"
        if cpath.exists():
            with open(cpath, encoding="utf-8") as f:
                data = json.load(f)
            clause_list = data.get("clauses", [])
            clauses_by_doc[doc_id] = {c.get("clause_id", ""): c for c in clause_list}

    # Bootstrap per model per doc
    results = {}

    for model_name in MODELS:
        results[model_name] = {}
        for doc_id in DOCS:
            if model_name == "gpt-4o":
                tpath = PROCESSED_DIR / f"{doc_id}_tuples.json"
            else:
                tpath = PROCESSED_DIR / f"{doc_id}_tuples_{model_name.replace('-', '_')}.json"

            if not tpath.exists():
                print(f"  [SKIP] {model_name}/{doc_id} — no tuples file")
                continue

            with open(tpath, encoding="utf-8") as f:
                data = json.load(f)
            tuples = data.get("parsed_tuples", [])
            doc_clauses = clauses_by_doc.get(doc_id, {})

            # Build matched pairs: (tuple, original_text, gold_row)
            pairs = []
            for t in tuples:
                cid = t.get("clause_id", "")
                clause = doc_clauses.get(cid, {})
                gold = gold_lookup.get(cid, {})
                orig_text = clause.get("text", t.get("text", ""))
                rif = compute_rif_single(t, orig_text, gold)
                pairs.append(rif)

            if not pairs:
                continue

            pairs = np.array(pairs)
            sample_size = int(len(pairs) * SAMPLE_FRAC)

            # Bootstrap
            boot_rifs = []
            for _ in range(N_BOOT):
                idx = np.random.choice(len(pairs), size=sample_size, replace=True)
                boot_rifs.append(float(np.mean(pairs[idx])))

            boot_rifs = np.array(boot_rifs)
            mean_rif = float(np.mean(boot_rifs))
            lower = float(np.percentile(boot_rifs, 2.5))
            upper = float(np.percentile(boot_rifs, 97.5))

            results[model_name][doc_id] = {
                "mean": round(mean_rif, 4),
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "point_estimate": round(float(np.mean(pairs)), 4),
                "n_clauses": len(pairs),
            }

    # Print results
    print(f"\n{BLD}  {'Model':<18} {'Doc':<8} {'N':>5} {'RIF_point':>10} {'Boot Mean':>10} {'CI Lower':>9} {'CI Upper':>9}{RST}")
    print(f"  {'-'*72}")

    for model_name in MODELS:
        for doc_id in DOCS:
            r = results.get(model_name, {}).get(doc_id)
            if r:
                print(f"  {model_name:<18} {doc_id:<8} {r['n_clauses']:>5} "
                      f"{r['point_estimate']:>10.4f} {r['mean']:>10.4f} "
                      f"{r['lower']:>9.4f} {r['upper']:>9.4f}")

    # Overlap analysis: Claude vs GPT-4o
    print(f"\n{BLD}  CI Overlap Analysis (Claude Sonnet vs GPT-4o):{RST}")
    for doc_id in DOCS:
        cs = results.get("claude-sonnet", {}).get(doc_id)
        gpt = results.get("gpt-4o", {}).get(doc_id)
        if cs and gpt:
            overlap = cs["lower"] <= gpt["upper"] and gpt["lower"] <= cs["upper"]
            status = "OVERLAPPING" if overlap else "NON-OVERLAPPING"
            symbol = f"{GRN}***{RST}" if not overlap else ""
            print(f"    {doc_id}: Claude [{cs['lower']:.4f}, {cs['upper']:.4f}] "
                  f"vs GPT-4o [{gpt['lower']:.4f}, {gpt['upper']:.4f}] "
                  f"-> {status} {symbol}")

    # Update multimodel_comparison.csv
    csv_path = OUTPUTS_DIR / "reports" / "multimodel_comparison.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        for row in csv_rows:
            model = row.get("model", "")
            doc = row.get("document", "")
            r = results.get(model, {}).get(doc)
            if r:
                row["RIF_lower"] = r["lower"]
                row["RIF_upper"] = r["upper"]

        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\n  {CHECK} Updated {csv_path.name} with new RIF_lower/RIF_upper")

    # Print final CSV
    print(f"\n{BLD}  Updated multimodel_comparison.csv:{RST}\n")
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        cols = list(csv_rows[0].keys())
        header = "  " + " | ".join(f"{c:>12}" for c in cols)
        print(header)
        print(f"  {'-' * len(header)}")
        for row in csv_rows:
            line = "  " + " | ".join(f"{row[c]:>12}" for c in cols)
            print(line)

    print()


if __name__ == "__main__":
    main()
