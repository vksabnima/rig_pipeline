#!/usr/bin/env python3
"""
Automated Error Analysis on the 30 lowest-RIF clauses from GPT-4o.
Classifies each into one of 5 failure types using Claude Sonnet.
"""

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

sys.path.insert(0, str(Path(__file__).parent))

from config.config import ANTHROPIC_API_KEY, PROCESSED_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR

GRN = "\033[92m"; CYN = "\033[96m"; BLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"

TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]
REGULATORY_KEYWORDS = {
    "shall", "must", "required", "prohibited", "ensure", "comply",
    "obligation", "mandatory", "limit", "minimum", "maximum", "capital",
    "risk", "exposure", "deadline", "within", "before", "after",
}

CLASSIFICATION_PROMPT = """You are a regulatory NLP error analyst. Classify this extraction failure into exactly one category:

1. NESTED_OBLIGATION -- clause contains 2+ distinct obligations merged into one tuple (signal: long obligation_gt with 'and' connecting independent requirements)

2. IMPLICIT_SUBJECT -- subject is implied by context not stated in clause (signal: subject_gt blank or generic like 'regulated entity')

3. CROSS_REFERENCE -- obligation only interpretable via another paragraph (signal: text contains 'as specified in paragraph', 'pursuant to', 'in accordance with paragraph [N]')

4. THRESHOLD_AMBIGUITY -- numeric threshold present but units or reference frame unclear (signal: threshold_gt partially populated or contains approximations)

5. EXCEPTION_DOMINANCE -- clause is primarily an exception, obligation is secondary (signal: exception_gt longer than obligation_gt, or text starts with 'unless', 'except', 'provided that')

Here is the clause text:
{clause_text}

Here is the GPT-4o extracted tuple:
{tuple_json}

Here is the gold standard annotation:
{gold_json}

The RIF score for this extraction is {rif_score:.4f} (low = poor extraction quality).

Return ONLY valid JSON:
{{
  "clause_id": "{clause_id}",
  "failure_type": "one of: NESTED_OBLIGATION, IMPLICIT_SUBJECT, CROSS_REFERENCE, THRESHOLD_AMBIGUITY, EXCEPTION_DOMINANCE",
  "confidence": 0.0,
  "reason": "one sentence explanation"
}}
Never return more than one failure type."""


def compute_clause_rif(parsed_tuple: dict, original_text: str) -> float:
    """Compute RIF for a single clause."""
    t = parsed_tuple.get("tuple", {})
    non_null = sum(1 for f in TUPLE_FIELDS if t.get(f) is not None)
    completeness = non_null / len(TUPLE_FIELDS)

    orig_words = set(original_text.lower().split())
    orig_keywords = orig_words & REGULATORY_KEYWORDS

    if orig_keywords:
        tuple_text = " ".join(str(v).lower() for v in t.values() if v is not None)
        retained = sum(1 for kw in orig_keywords if kw in tuple_text)
        retention = retained / len(orig_keywords)
    else:
        retention = 1.0

    return 0.5 * completeness + 0.5 * retention


def main():
    start = time.time()
    print(f"\n{CYN}{BLD}=== Automated Error Analysis: 30 Lowest-RIF GPT-4o Clauses ==={RST}\n")

    # ── STEP 1: Identify 30 lowest-RIF clauses ──────────────────────────────
    print(f"{BLD}Step 1: Identifying lowest-RIF clauses...{RST}")

    # Load gold standard
    gold_path = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
    with open(gold_path, encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))

    reviewed = [r for r in gold_rows if r.get("human_reviewed", "").upper() == "TRUE"]
    if not reviewed:
        print(f"  Note: No human_reviewed=TRUE rows found. Using all {len(gold_rows)} rows.")
        reviewed = gold_rows

    # Load GPT-4o tuples for both docs
    gpt4o_tuples = {}
    for doc_id in ["basel3", "regbi"]:
        path = PROCESSED_DIR / f"{doc_id}_tuples.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for t in data.get("parsed_tuples", []):
                gpt4o_tuples[t.get("clause_id", "")] = t

    # Compute per-clause RIF and match with gold standard
    clause_rifs = []
    for gold_row in reviewed:
        clause_id = gold_row.get("clause_id", "")
        if clause_id not in gpt4o_tuples:
            continue

        parsed = gpt4o_tuples[clause_id]
        original_text = parsed.get("text", gold_row.get("text", ""))
        rif = compute_clause_rif(parsed, original_text)

        clause_rifs.append({
            "clause_id": clause_id,
            "text": original_text,
            "parsed_tuple": parsed.get("tuple", {}),
            "gold_row": gold_row,
            "rif_score": rif,
        })

    # Sort by RIF ascending, take bottom 30
    clause_rifs.sort(key=lambda x: x["rif_score"])
    bottom_30 = clause_rifs[:30]

    print(f"  Total matched clauses: {len(clause_rifs)}")
    print(f"  Lowest RIF: {bottom_30[0]['rif_score']:.4f}")
    print(f"  Highest in bottom 30: {bottom_30[-1]['rif_score']:.4f}")
    print(f"  {CHECK} Selected 30 lowest-RIF clauses\n")

    # ── STEP 2: Classify each clause via Claude Sonnet ───────────────────────
    print(f"{BLD}Step 2: Classifying failure types via Claude Sonnet...{RST}")

    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    classifications = []
    for i, item in enumerate(bottom_30):
        clause_id = item["clause_id"]
        print(f"  [{i+1}/30] {clause_id} (RIF={item['rif_score']:.4f})")

        gold_fields = {
            "subject_gt": item["gold_row"].get("subject_gt"),
            "obligation_gt": item["gold_row"].get("obligation_gt"),
            "condition_gt": item["gold_row"].get("condition_gt"),
            "threshold_gt": item["gold_row"].get("threshold_gt"),
            "deadline_gt": item["gold_row"].get("deadline_gt"),
            "exception_gt": item["gold_row"].get("exception_gt"),
        }

        prompt = CLASSIFICATION_PROMPT.format(
            clause_text=item["text"][:800],
            tuple_json=json.dumps(item["parsed_tuple"], indent=2),
            gold_json=json.dumps(gold_fields, indent=2),
            rif_score=item["rif_score"],
            clause_id=clause_id,
        )

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.content[0].text
            s = content.find("{")
            e = content.rfind("}") + 1
            if s >= 0 and e > s:
                result = json.loads(content[s:e])
            else:
                result = {"clause_id": clause_id, "failure_type": "UNKNOWN",
                          "confidence": 0.0, "reason": "Failed to parse response"}
        except Exception as ex:
            result = {"clause_id": clause_id, "failure_type": "UNKNOWN",
                      "confidence": 0.0, "reason": str(ex)}

        result["clause_id"] = clause_id  # Ensure correct ID
        result["rif_score"] = item["rif_score"]
        result["text_snippet"] = item["text"][:80]
        classifications.append(result)

    print(f"  {CHECK} All 30 clauses classified\n")

    # ── STEP 3: Aggregate and save ───────────────────────────────────────────
    print(f"{BLD}Step 3: Aggregating results...{RST}")

    # Detail CSV
    detail_path = OUTPUTS_DIR / "reports" / "error_analysis_detail.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clause_id", "raw_text_snippet", "failure_type", "confidence", "reason", "rif_score"
        ])
        writer.writeheader()
        for c in classifications:
            writer.writerow({
                "clause_id": c["clause_id"],
                "raw_text_snippet": c.get("text_snippet", ""),
                "failure_type": c.get("failure_type", "UNKNOWN"),
                "confidence": c.get("confidence", 0.0),
                "reason": c.get("reason", ""),
                "rif_score": round(c["rif_score"], 4),
            })
    print(f"  {CHECK} Saved {detail_path.relative_to(Path(__file__).parent)}")

    # Summary aggregation
    failure_types = ["NESTED_OBLIGATION", "IMPLICIT_SUBJECT", "CROSS_REFERENCE",
                     "THRESHOLD_AMBIGUITY", "EXCEPTION_DOMINANCE"]

    summary = []
    for ftype in failure_types:
        matches = [c for c in classifications if c.get("failure_type") == ftype]
        if not matches:
            continue
        count = len(matches)
        pct = count / len(classifications) * 100
        mean_rif = sum(m["rif_score"] for m in matches) / count
        # Best example = highest confidence
        best = max(matches, key=lambda x: x.get("confidence", 0))
        summary.append({
            "failure_type": ftype,
            "count": count,
            "percentage": round(pct, 1),
            "mean_rif": round(mean_rif, 4),
            "example_clause_id": best["clause_id"],
        })

    # Handle any UNKNOWN
    unknowns = [c for c in classifications if c.get("failure_type") not in failure_types]
    if unknowns:
        summary.append({
            "failure_type": "UNKNOWN",
            "count": len(unknowns),
            "percentage": round(len(unknowns) / len(classifications) * 100, 1),
            "mean_rif": round(sum(u["rif_score"] for u in unknowns) / len(unknowns), 4),
            "example_clause_id": unknowns[0]["clause_id"],
        })

    # Sort by count descending
    summary.sort(key=lambda x: x["count"], reverse=True)

    summary_path = OUTPUTS_DIR / "reports" / "error_analysis_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "failure_type", "count", "percentage", "mean_rif", "example_clause_id"
        ])
        writer.writeheader()
        writer.writerows(summary)
    print(f"  {CHECK} Saved {summary_path.relative_to(Path(__file__).parent)}")

    # ── STEP 4: Print summary table ─────────────────────────────────────────
    elapsed = round(time.time() - start, 1)

    print(f"\n{CYN}{BLD}{'='*75}")
    print(f"  Error Analysis Summary: GPT-4o Lowest-RIF Clauses")
    print(f"{'='*75}{RST}\n")

    print(f"{BLD}  {'Failure Type':<24} {'Count':>6} {'%':>7} {'Mean RIF':>9} {'Example':>12}{RST}")
    print(f"  {'-'*65}")
    for s in summary:
        print(f"  {s['failure_type']:<24} {s['count']:>6} {s['percentage']:>6.1f}% {s['mean_rif']:>9.4f} {s['example_clause_id']:>12}")
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<24} {len(classifications):>6} {100.0:>6.1f}%")

    # Print best example per failure type
    print(f"\n{BLD}  Most Representative Example Per Failure Type:{RST}\n")
    for s in summary:
        ftype = s["failure_type"]
        matches = [c for c in classifications if c.get("failure_type") == ftype]
        best = max(matches, key=lambda x: x.get("confidence", 0))
        print(f"  {BLD}{ftype}{RST} (confidence={best.get('confidence', 0):.2f})")
        print(f"    Clause: {best['clause_id']}")
        print(f"    Text:   {best.get('text_snippet', '')[:75]}...")
        print(f"    Reason: {best.get('reason', '')}")
        print()

    print(f"{DIM}  Completed in {elapsed}s{RST}\n")


if __name__ == "__main__":
    main()
