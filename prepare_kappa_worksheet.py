#!/usr/bin/env python3
"""
Prepares a 100-clause stratified random sample from Basel III for human
annotation. Output is a CSV with the source text + Claude Sonnet's
auto-annotation side-by-side and 6 blank "human_*" columns for the
human annotator to fill in.

Run:   python prepare_kappa_worksheet.py
Out:   data/annotations/kappa_worksheet.csv

After human fills it in, run compute_kappa.py to get Cohen's kappa
per field (and overall Krippendorff alpha).
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, ANNOTATIONS_DIR

random.seed(42)

GT_FIELDS = ["subject_gt", "obligation_gt", "condition_gt",
             "threshold_gt", "deadline_gt", "exception_gt"]
HUMAN_FIELDS = ["human_subject", "human_obligation", "human_condition",
                "human_threshold", "human_deadline", "human_exception"]

N_SAMPLE = 100

# Load Basel III clauses and the auto-annotated gold
with open(PROCESSED_DIR / "basel3_clauses.json", encoding="utf-8") as f:
    clauses = json.load(f).get("clauses", [])

with open(ANNOTATIONS_DIR / "gold_standard_reviewed.csv", encoding="utf-8") as f:
    gold_rows = {r["clause_id"]: r for r in csv.DictReader(f)}

# Stratify by clause text length (short / medium / long) so the sample is
# representative — annotation difficulty correlates with length.
def length_bucket(s):
    n = len(s.split())
    if n < 25: return "short"
    if n < 60: return "medium"
    return "long"

buckets = {"short": [], "medium": [], "long": []}
for c in clauses:
    if c["clause_id"] in gold_rows:
        buckets[length_bucket(c["text"])].append(c)

per_bucket = N_SAMPLE // 3
sample = []
for k in ("short", "medium", "long"):
    pool = buckets[k]
    sample.extend(random.sample(pool, min(per_bucket, len(pool))))
# Top up to N_SAMPLE if integer division left a gap
if len(sample) < N_SAMPLE:
    rest = [c for k in buckets for c in buckets[k] if c not in sample]
    sample.extend(random.sample(rest, min(N_SAMPLE - len(sample), len(rest))))

# Build the worksheet
out_rows = []
for c in sample:
    g = gold_rows[c["clause_id"]]
    row = {
        "clause_id": c["clause_id"],
        "section": c.get("section", ""),
        "source_text": c["text"][:1000].replace("\n", " "),
    }
    # Auto-annotation (Claude) side
    for f in GT_FIELDS:
        row[f"claude_{f.replace('_gt','')}"] = g.get(f, "")
    # Blank human columns
    for hf in HUMAN_FIELDS:
        row[hf] = ""
    row["notes"] = ""
    out_rows.append(row)

# Write
out_path = ANNOTATIONS_DIR / "kappa_worksheet.csv"
fieldnames = ["clause_id", "section", "source_text"] + \
             [f"claude_{f.replace('_gt','')}" for f in GT_FIELDS] + \
             HUMAN_FIELDS + ["notes"]
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

print(f"Wrote {len(out_rows)} clauses to {out_path}")
print(f"\nINSTRUCTIONS:")
print(f"  Open in Excel / a CSV editor.")
print(f"  For each row, read source_text and Claude's annotation,")
print(f"  then fill the human_* columns with what YOU think the")
print(f"  correct value is (or leave blank if the field is not present).")
print(f"  After completion, run: python compute_kappa.py")
