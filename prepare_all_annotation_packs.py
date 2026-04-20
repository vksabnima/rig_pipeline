#!/usr/bin/env python3
"""
Generate three human-annotation worksheets for the paper revision:

  1. Basel III kappa expansion (300 clauses, stratified short/med/long)
  2. Reg BI gold standard (100 clauses, stratified)
  3. GDPR independent human validation (50 clauses, stratified)

Each produces:
  data/annotations/{name}_worksheet.csv  — the CSV to hand to the annotator
  data/annotations/{name}_GUIDE.md       — one-page protocol with examples

The annotator fills the human_* columns, saves, and you run the matching
score script (compute_kappa.py for Basel, compute_regbi_gold.py for Reg BI,
compute_gdpr_gold.py for GDPR) to recompute the metrics.
"""

import csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, ANNOTATIONS_DIR

random.seed(42)

GT_FIELDS = ["subject_gt", "obligation_gt", "condition_gt",
             "threshold_gt", "deadline_gt", "exception_gt"]
HUMAN_FIELDS = ["human_subject", "human_obligation", "human_condition",
                "human_threshold", "human_deadline", "human_exception"]


def length_bucket(s):
    n = len(s.split())
    if n < 25: return "short"
    if n < 60: return "medium"
    return "long"


def stratified_sample(clauses_with_gold, n_total, rng):
    buckets = {"short": [], "medium": [], "long": []}
    for c in clauses_with_gold:
        buckets[length_bucket(c["text"])].append(c)
    per = n_total // 3
    picked = []
    for k in ("short", "medium", "long"):
        pool = buckets[k]
        picked.extend(rng.sample(pool, min(per, len(pool))))
    if len(picked) < n_total:
        rest = [c for k in buckets for c in buckets[k] if c not in picked]
        picked.extend(rng.sample(rest, min(n_total - len(picked), len(rest))))
    return picked


def write_worksheet(out_path, clauses, gold_lookup, show_existing=True):
    """
    If show_existing=True: include the auto-annotated columns alongside (for
    kappa where we want to measure agreement with Claude's pass).
    If show_existing=False: only source_text + blank human_* columns (for
    producing a fresh independent gold standard).
    """
    rows = []
    for c in clauses:
        g = gold_lookup.get(c["clause_id"], {}) if gold_lookup else {}
        row = {"clause_id": c["clause_id"],
               "section": c.get("section", ""),
               "source_text": c["text"][:1200].replace("\n", " ")}
        if show_existing and g:
            for f in GT_FIELDS:
                key = f"claude_{f.replace('_gt','')}"
                row[key] = g.get(f, "")
        for hf in HUMAN_FIELDS:
            row[hf] = ""
        row["notes"] = ""
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


# ── 1. Basel III kappa expansion (300 clauses) ───────────────────────────────
with open(PROCESSED_DIR / "basel3_clauses.json", encoding="utf-8") as f:
    basel_clauses = json.load(f)["clauses"]
with open(ANNOTATIONS_DIR / "gold_standard_reviewed.csv", encoding="utf-8") as f:
    basel_gold = {r["clause_id"]: r for r in csv.DictReader(f)}

basel_pool = [c for c in basel_clauses if c["clause_id"] in basel_gold]
# Exclude the 100 already in annotated_kappa to avoid re-annotating them
try:
    with open(ANNOTATIONS_DIR / "annotated_kappa.csv", encoding="utf-8") as f:
        already = {r["clause_id"] for r in csv.DictReader(f)}
    basel_pool = [c for c in basel_pool if c["clause_id"] not in already]
    exclusion_note = f"(excluded {len(already)} already in annotated_kappa.csv)"
except FileNotFoundError:
    already = set(); exclusion_note = ""

rng1 = random.Random(42)
basel_sample = stratified_sample(basel_pool, 300, rng1)
n1 = write_worksheet(ANNOTATIONS_DIR / "kappa_300_worksheet.csv",
                     basel_sample, basel_gold, show_existing=True)
print(f"1. Basel III kappa (additional 300 clauses) written: n={n1} {exclusion_note}")

# ── 2. Reg BI gold standard (100 clauses) ────────────────────────────────────
with open(PROCESSED_DIR / "regbi_clauses.json", encoding="utf-8") as f:
    regbi_clauses = json.load(f)["clauses"]

# No pre-existing gold to filter on — sample from all
rng2 = random.Random(43)
regbi_sample = stratified_sample(regbi_clauses, 100, rng2)
n2 = write_worksheet(ANNOTATIONS_DIR / "regbi_gold_worksheet.csv",
                     regbi_sample, None, show_existing=False)
print(f"2. Reg BI fresh gold (100 clauses) written: n={n2}")

# ── 3. GDPR independent gold (50 clauses) ────────────────────────────────────
with open(PROCESSED_DIR / "gdpr_clauses.json", encoding="utf-8") as f:
    gdpr_clauses = json.load(f)["clauses"]

# Exclude the 50 in gdpr_gold_standard.csv (those are Claude-auto-annotated)
try:
    with open(ANNOTATIONS_DIR / "gdpr_gold_standard.csv", encoding="utf-8") as f:
        already_gdpr = {r["clause_id"] for r in csv.DictReader(f)}
    gdpr_pool = [c for c in gdpr_clauses if c["clause_id"] not in already_gdpr]
    gdpr_note = f"(excluded {len(already_gdpr)} already in gdpr_gold_standard.csv)"
except FileNotFoundError:
    gdpr_pool = gdpr_clauses; gdpr_note = ""

rng3 = random.Random(44)
gdpr_sample = stratified_sample(gdpr_pool, 50, rng3)
n3 = write_worksheet(ANNOTATIONS_DIR / "gdpr_human_worksheet.csv",
                     gdpr_sample, None, show_existing=False)
print(f"3. GDPR fresh gold (50 clauses) written: n={n3} {gdpr_note}")

print("\nDone. Files in data/annotations/:")
print("  - kappa_300_worksheet.csv        (Basel III kappa expansion)")
print("  - regbi_gold_worksheet.csv       (Reg BI fresh gold)")
print("  - gdpr_human_worksheet.csv       (GDPR independent gold)")
