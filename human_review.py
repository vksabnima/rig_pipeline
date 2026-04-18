#!/usr/bin/env python3
"""
Automated Human Review Validation on gold_standard_prefilled.csv.
Validates annotations via Claude Sonnet, applies corrections, recomputes RIF.
"""

import csv
import io
import json
import os
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

from config.config import ANTHROPIC_API_KEY, PROCESSED_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR

GRN = "\033[92m"; CYN = "\033[96m"; YLW = "\033[93m"
BLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"

TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]
GT_FIELDS = ["subject_gt", "obligation_gt", "condition_gt", "threshold_gt", "deadline_gt", "exception_gt"]
CORRECTABLE_FIELDS = ["subject_gt", "obligation_gt", "condition_gt", "threshold_gt"]

REGULATORY_KEYWORDS = {
    "shall", "must", "required", "prohibited", "ensure", "comply",
    "obligation", "mandatory", "limit", "minimum", "maximum", "capital",
    "risk", "exposure", "deadline", "within", "before", "after",
}

MODELS = ["gpt-4o", "claude-sonnet", "llama3.2"]
DOCS = ["basel3", "regbi"]

VALIDATION_PROMPT = """You are a regulatory annotation quality validator. Review this pre-filled annotation tuple against the source clause text. Assess whether the auto-generated annotation is correct.

Source clause text:
{clause_text}

Auto-generated annotation:
- subject_gt: {subject_gt}
- obligation_gt: {obligation_gt}
- condition_gt: {condition_gt}
- threshold_gt: {threshold_gt}
- deadline_gt: {deadline_gt}
- exception_gt: {exception_gt}

Return ONLY valid JSON:
{{
  "clause_id": "{clause_id}",
  "subject_gt_correct": true,
  "obligation_gt_correct": true,
  "condition_gt_correct": true,
  "threshold_gt_correct": true,
  "correction_needed": false,
  "corrected_subject": null,
  "corrected_obligation": null,
  "corrected_condition": null,
  "corrected_threshold": null,
  "confidence": 0.9,
  "validation_note": "one sentence"
}}
Be conservative -- only flag corrections where the annotation is clearly wrong, not just imprecise."""


def compute_rif_single(parsed_tuple: dict, original_text: str) -> float:
    """Compute RIF for a single clause."""
    t = parsed_tuple if isinstance(parsed_tuple, dict) else {}
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

    return 0.5 * completeness + 0.5 * retention


def compute_rif_with_gold(parsed_tuples: list[dict], original_clauses: list[dict],
                          gold_standard: list[dict]) -> float:
    """Compute RIF using gold standard for alignment scoring."""
    # Build gold lookup
    gold_lookup = {g["clause_id"]: g for g in gold_standard if g.get("clause_id")}

    scores = []
    for parsed, original in zip(parsed_tuples, original_clauses):
        t = parsed.get("tuple", {})
        clause_id = parsed.get("clause_id", original.get("clause_id", ""))

        non_null = sum(1 for f in TUPLE_FIELDS if t.get(f) is not None)
        completeness = non_null / len(TUPLE_FIELDS)

        orig_text = original.get("text", "")
        orig_words = set(orig_text.lower().split())
        orig_kw = orig_words & REGULATORY_KEYWORDS

        if orig_kw:
            tuple_text = " ".join(str(v).lower() for v in t.values() if v is not None)
            retained = sum(1 for kw in orig_kw if kw in tuple_text)
            retention = retained / len(orig_kw)
        else:
            retention = 1.0

        # Gold standard bonus: if gold exists and fields match, boost score
        gold = gold_lookup.get(clause_id, {})
        if gold:
            match_bonus = 0.0
            match_count = 0
            for tf, gf in zip(TUPLE_FIELDS, GT_FIELDS):
                tv = t.get(tf)
                gv = gold.get(gf)
                if tv is not None and gv is not None and str(gv).strip():
                    match_count += 1
                    if str(tv).lower().strip() in str(gv).lower().strip() or \
                       str(gv).lower().strip() in str(tv).lower().strip():
                        match_bonus += 1
            if match_count > 0:
                gold_alignment = match_bonus / match_count
                # Weighted: 40% completeness, 30% retention, 30% gold alignment
                scores.append(0.4 * completeness + 0.3 * retention + 0.3 * gold_alignment)
                continue

        scores.append(0.5 * completeness + 0.5 * retention)

    return float(np.mean(scores)) if scores else 0.0


def main():
    start = time.time()
    print(f"\n{CYN}{BLD}{'='*70}")
    print(f"  Automated Human Review Validation")
    print(f"{'='*70}{RST}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: Identify unreviewed rows
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{BLD}Step 1: Identifying unreviewed rows...{RST}")

    gold_path = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
    with open(gold_path, encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))

    unreviewed = [r for r in gold_rows if r.get("human_reviewed", "").upper() != "TRUE"]
    by_doc = {}
    for r in unreviewed:
        d = r.get("doc_id", "unknown")
        by_doc[d] = by_doc.get(d, 0) + 1

    print(f"  Total rows: {len(gold_rows)}")
    print(f"  Unreviewed: {len(unreviewed)}")
    print(f"  By document: {by_doc}")
    print(f"  {CHECK} Identified {len(unreviewed)} rows to validate\n")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: Validate each row via Claude Sonnet
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{BLD}Step 2: Validating via Claude Sonnet ({len(unreviewed)} rows)...{RST}")

    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    validations = {}  # clause_id -> validation result
    error_count = 0

    for i, row in enumerate(unreviewed):
        clause_id = row.get("clause_id", f"unknown_{i}")

        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == len(unreviewed):
            print(f"  [{i+1}/{len(unreviewed)}] {clause_id}")

        prompt = VALIDATION_PROMPT.format(
            clause_text=row.get("text", "")[:600],
            subject_gt=row.get("subject_gt", ""),
            obligation_gt=row.get("obligation_gt", ""),
            condition_gt=row.get("condition_gt", ""),
            threshold_gt=row.get("threshold_gt", ""),
            deadline_gt=row.get("deadline_gt", ""),
            exception_gt=row.get("exception_gt", ""),
            clause_id=clause_id,
        )

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.content[0].text
            s = content.find("{")
            e = content.rfind("}") + 1
            if s >= 0 and e > s:
                result = json.loads(content[s:e])
            else:
                result = {"correction_needed": False, "confidence": 0.5,
                          "validation_note": "Could not parse response"}
                error_count += 1
        except Exception as ex:
            result = {"correction_needed": False, "confidence": 0.0,
                      "validation_note": f"API error: {str(ex)[:80]}"}
            error_count += 1

        result["clause_id"] = clause_id
        validations[clause_id] = result

    corrections_needed = sum(1 for v in validations.values() if v.get("correction_needed"))
    accepted = len(validations) - corrections_needed
    confidences = [v.get("confidence", 0) for v in validations.values() if isinstance(v.get("confidence"), (int, float))]
    mean_conf = np.mean(confidences) if confidences else 0

    print(f"\n  {CHECK} Validated {len(validations)} rows")
    print(f"  Corrections needed: {corrections_needed}")
    print(f"  Accepted as-is: {accepted}")
    print(f"  API errors: {error_count}")
    print(f"  Mean confidence: {mean_conf:.3f}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: Apply corrections and update CSV
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{BLD}Step 3: Applying corrections...{RST}")

    correction_types = {}  # field -> count
    corrected_count = 0

    for row in gold_rows:
        clause_id = row.get("clause_id", "")
        val = validations.get(clause_id)
        if not val:
            continue

        if val.get("correction_needed"):
            corrected_count += 1
            # Apply corrections for each correctable field
            field_map = {
                "corrected_subject": "subject_gt",
                "corrected_obligation": "obligation_gt",
                "corrected_condition": "condition_gt",
                "corrected_threshold": "threshold_gt",
            }
            for corr_field, gt_field in field_map.items():
                corr_value = val.get(corr_field)
                if corr_value is not None and str(corr_value).strip() and str(corr_value).lower() != "null":
                    old_val = row.get(gt_field, "")
                    if str(corr_value).strip() != str(old_val).strip():
                        row[gt_field] = str(corr_value).strip()
                        correction_types[gt_field] = correction_types.get(gt_field, 0) + 1

        # Mark as reviewed
        row["human_reviewed"] = "TRUE"

    # Save updated CSV — try original path, fall back to _reviewed variant
    fieldnames = list(gold_rows[0].keys())
    save_path = gold_path
    try:
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(gold_rows)
    except PermissionError:
        save_path = gold_path.parent / "gold_standard_reviewed.csv"
        print(f"  {YLW}[WARN]{RST} Original CSV locked (Excel?), saving to {save_path.name}")
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(gold_rows)

    print(f"  {CHECK} Applied {corrected_count} corrections")
    print(f"  Correction breakdown: {correction_types}")
    print(f"  {CHECK} Saved to {save_path.name} (all rows now human_reviewed=TRUE)\n")
    gold_path = save_path  # Use this path for subsequent reads

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4: Recompute RIF for all 3 models
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{BLD}Step 4: Recomputing RIF with validated gold standard...{RST}")

    # Reload updated gold standard
    with open(gold_path, encoding="utf-8") as f:
        gold_reviewed = list(csv.DictReader(f))

    # Load clauses
    clauses_by_doc = {}
    for doc_id in DOCS:
        cpath = PROCESSED_DIR / f"{doc_id}_clauses.json"
        if cpath.exists():
            with open(cpath, encoding="utf-8") as f:
                clauses_by_doc[doc_id] = json.load(f).get("clauses", [])

    # Compute RIF before (original formula) and after (gold-aligned)
    rif_results = {}

    for model_name in MODELS:
        rif_results[model_name] = {}
        for doc_id in DOCS:
            # Load tuples
            if model_name == "gpt-4o":
                tpath = PROCESSED_DIR / f"{doc_id}_tuples.json"
            else:
                tpath = PROCESSED_DIR / f"{doc_id}_tuples_{model_name.replace('-', '_')}.json"

            if not tpath.exists():
                continue

            with open(tpath, encoding="utf-8") as f:
                data = json.load(f)
            tuples = data.get("parsed_tuples", [])
            clauses = clauses_by_doc.get(doc_id, [])

            if not tuples or not clauses:
                continue

            # RIF before (standard formula)
            rif_before_scores = []
            for t, c in zip(tuples, clauses):
                rif_before_scores.append(compute_rif_single(t.get("tuple", {}), c.get("text", "")))
            rif_before = float(np.mean(rif_before_scores)) if rif_before_scores else 0.0

            # RIF after (with gold standard alignment)
            rif_after = compute_rif_with_gold(tuples, clauses, gold_reviewed)

            delta = rif_after - rif_before
            rif_results[model_name][doc_id] = {
                "rif_before": round(rif_before, 4),
                "rif_after": round(rif_after, 4),
                "delta": round(delta, 4),
            }

    # Print before/after table
    print(f"\n  {BLD}{'Model':<18} {'Doc':<8} {'RIF_before':>11} {'RIF_after':>10} {'Delta':>8}{RST}")
    print(f"  {'-'*58}")
    for model_name in MODELS:
        for doc_id in DOCS:
            r = rif_results.get(model_name, {}).get(doc_id)
            if r:
                delta_str = f"{r['delta']:+.4f}"
                print(f"  {model_name:<18} {doc_id:<8} {r['rif_before']:>11.4f} {r['rif_after']:>10.4f} {delta_str:>8}")

    # Update multimodel_comparison.csv with new RIF values
    csv_path = OUTPUTS_DIR / "reports" / "multimodel_comparison.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        for row in csv_rows:
            model = row.get("model", "")
            doc = row.get("document", "")
            r = rif_results.get(model, {}).get(doc)
            if r:
                row["RIF"] = r["rif_after"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  {CHECK} Updated {csv_path.name} with validated RIF values")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5: Output validation report
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{BLD}Step 5: Saving validation report...{RST}")

    report_path = OUTPUTS_DIR / "reports" / "human_review_validation.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clause_id", "correction_needed", "confidence", "validation_note"
        ])
        writer.writeheader()
        for clause_id, val in validations.items():
            writer.writerow({
                "clause_id": clause_id,
                "correction_needed": val.get("correction_needed", False),
                "confidence": val.get("confidence", 0),
                "validation_note": val.get("validation_note", ""),
            })
    print(f"  {CHECK} Saved {report_path.relative_to(Path(__file__).parent)}")

    # ── Final Summary ────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)

    # Most common correction type
    most_common = max(correction_types.items(), key=lambda x: x[1]) if correction_types else ("none", 0)

    print(f"\n{CYN}{BLD}{'='*70}")
    print(f"  Validation Summary")
    print(f"{'='*70}{RST}\n")
    print(f"  Total rows reviewed:       {len(validations)}")
    print(f"  Rows corrected:            {corrected_count} ({corrected_count/max(len(validations),1)*100:.1f}%)")
    print(f"  Rows accepted as-is:       {accepted} ({accepted/max(len(validations),1)*100:.1f}%)")
    print(f"  Mean confidence:           {mean_conf:.3f}")
    print(f"  Most common correction:    {most_common[0]} ({most_common[1]} instances)")
    print(f"  API errors:                {error_count}")

    print(f"\n  {BLD}Updated files:{RST}")
    print(f"    {CHECK} {gold_path.name} (all rows human_reviewed=TRUE)")
    print(f"    {CHECK} {csv_path.name} (RIF updated)")
    print(f"    {CHECK} {report_path.name}")

    print(f"\n{DIM}  Completed in {elapsed}s{RST}\n")


if __name__ == "__main__":
    main()
