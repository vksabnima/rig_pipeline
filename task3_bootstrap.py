"""
Task 3 — Bootstrap 95% CI for RIF (Regulatory Intent Fidelity) across 3 models.
"""

import io, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import json
import random
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJ = Path(__file__).parent
DATA = PROJ / "data" / "processed"

MODELS = ["gpt-4o", "claude-sonnet", "llama3.2"]
DOCS = ["basel3", "regbi"]

TUPLE_FILES = {
    ("gpt-4o", "basel3"):       DATA / "basel3_tuples.json",
    ("gpt-4o", "regbi"):        DATA / "regbi_tuples.json",
    ("claude-sonnet", "basel3"): DATA / "basel3_tuples_claude_sonnet.json",
    ("claude-sonnet", "regbi"):  DATA / "regbi_tuples_claude_sonnet.json",
    ("llama3.2", "basel3"):      DATA / "basel3_tuples_llama3.2.json",
    ("llama3.2", "regbi"):       DATA / "regbi_tuples_llama3.2.json",
}

CLAUSE_FILES = {
    "basel3": DATA / "basel3_clauses.json",
    "regbi":  DATA / "regbi_clauses.json",
}

FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]
REGULATORY_KEYWORDS = {
    "shall", "must", "required", "prohibited", "ensure", "comply",
    "obligation", "mandatory", "limit", "minimum", "maximum", "capital",
    "risk", "exposure", "deadline", "within", "before", "after",
}

N_BOOT = 1000
SAMPLE_FRAC = 0.80
random.seed(42)
np.random.seed(42)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rif_score(parsed_tuple, original_clause):
    """Compute per-pair intent score (same formula as compute_rif)."""
    t = parsed_tuple.get("tuple", {})
    non_null = sum(1 for f in FIELDS if t.get(f) is not None)
    completeness = non_null / len(FIELDS)

    orig_words = set(original_clause.get("text", "").lower().split())
    orig_keywords = orig_words & REGULATORY_KEYWORDS

    if orig_keywords:
        tuple_text = " ".join(str(v).lower() for v in t.values() if v is not None)
        retained = sum(1 for kw in orig_keywords if kw in tuple_text)
        retention = retained / len(orig_keywords)
    else:
        retention = 1.0

    return 0.5 * completeness + 0.5 * retention


def main():
    print("=" * 60, flush=True)
    print("Task 3 — Bootstrap 95 % CI for RIF", flush=True)
    print("=" * 60, flush=True)

    # Note: human_reviewed is all FALSE, so we use ALL gold_standard rows
    print("\nNOTE: human_reviewed column is all FALSE — using ALL gold_standard rows as population.\n", flush=True)

    # Load clause lookup by clause_id for each doc
    clause_lookup = {}
    for doc in DOCS:
        data = load_json(CLAUSE_FILES[doc])
        for c in data["clauses"]:
            clause_lookup[c["clause_id"]] = c

    print(f"Loaded {len(clause_lookup)} original clauses across {len(DOCS)} documents.\n", flush=True)

    results = {}  # (model, doc) -> {mean, lower, upper}

    for model in MODELS:
        for doc in DOCS:
            key = (model, doc)
            tup_data = load_json(TUPLE_FILES[key])
            parsed_tuples = tup_data["parsed_tuples"]

            # Match tuples to original clauses by clause_id
            pairs = []
            for pt in parsed_tuples:
                cid = pt.get("clause_id")
                if cid and cid in clause_lookup:
                    pairs.append((pt, clause_lookup[cid]))

            n_sample = max(1, int(len(pairs) * SAMPLE_FRAC))
            print(f"[{model} / {doc}]  matched pairs: {len(pairs)},  sample size per iter: {n_sample}", flush=True)

            boot_rifs = []
            for _ in range(N_BOOT):
                sample = random.choices(pairs, k=n_sample)
                scores = [rif_score(p, c) for p, c in sample]
                boot_rifs.append(float(np.mean(scores)) if scores else 0.0)

            boot_arr = np.array(boot_rifs)
            mean_rif = float(np.mean(boot_arr))
            lower = float(np.percentile(boot_arr, 2.5))
            upper = float(np.percentile(boot_arr, 97.5))

            results[key] = {"mean": round(mean_rif, 4), "lower": round(lower, 4), "upper": round(upper, 4)}
            print(f"         RIF mean={mean_rif:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]", flush=True)

    # ---- Update multimodel_comparison.csv ----
    csv_path = PROJ / "outputs" / "reports" / "multimodel_comparison.csv"
    print(f"\nUpdating {csv_path} ...", flush=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    # Add new columns if missing
    for col in ["RIF_lower", "RIF_upper"]:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        key = (row["model"], row["document"])
        if key in results:
            row["RIF_lower"] = results[key]["lower"]
            row["RIF_upper"] = results[key]["upper"]
        else:
            row.setdefault("RIF_lower", "")
            row.setdefault("RIF_upper", "")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.\n", flush=True)

    # ---- Print final table ----
    print("=" * 100, flush=True)
    print("Updated multimodel_comparison.csv:", flush=True)
    print("-" * 100, flush=True)
    header = ",".join(fieldnames)
    print(header, flush=True)
    for row in rows:
        print(",".join(str(row.get(c, "")) for c in fieldnames), flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
