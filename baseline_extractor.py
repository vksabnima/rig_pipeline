#!/usr/bin/env python3
"""
Rule-based baseline extractor for the RIG benchmark.
Zero LLM calls. Extracts 6-field obligation tuples from regulatory text
using regex + light heuristics.

Run:   python baseline_extractor.py
Out:   data/processed/{doc}_tuples_baseline.json (one per doc)
       outputs/reports/baseline_results.json (RCC/OAL/RIF per doc)
"""

import csv
import io
import json
import os
import re
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR
from src.metrics import compute_rcc, compute_oal, compute_rif
from human_review import compute_rif_with_gold

DOCS = ["basel3", "regbi", "gdpr"]
TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]

# ── Subject patterns ────────────────────────────────────────────────────────
SUBJECT_NPS = [
    r"\b(?:a|the|each|every|all)\s+(?:bank|institution|firm|broker[- ]?dealer|"
    r"investment\s+adviser|controller|processor|data\s+controller|data\s+processor|"
    r"member\s+state|supervisory\s+authority|competent\s+authority|natural\s+person|"
    r"legal\s+person|covered\s+entity|business\s+associate|undertaking|regulated\s+entity)s?\b",
    r"\b(?:banks?|institutions?|firms?|broker[- ]?dealers?|investment\s+advisers?|"
    r"controllers?|processors?|undertakings?|member\s+states?|"
    r"supervisory\s+authorities|competent\s+authorities)\b",
]
SUBJECT_RE = re.compile("|".join(SUBJECT_NPS), re.IGNORECASE)

# ── Obligation triggers ─────────────────────────────────────────────────────
OBL_TRIGGER_RE = re.compile(
    r"\b(shall|must|is\s+required\s+to|are\s+required\s+to|"
    r"shall\s+not|must\s+not|is\s+prohibited\s+from|are\s+prohibited\s+from|"
    r"may\s+not|shall\s+ensure|must\s+ensure|shall\s+comply|must\s+comply)\b",
    re.IGNORECASE,
)

# ── Condition triggers ──────────────────────────────────────────────────────
COND_RE = re.compile(
    r"\b(?:if|where|when|whenever|in\s+the\s+event\s+that|in\s+case\s+of|"
    r"provided\s+that|subject\s+to|in\s+respect\s+of|with\s+respect\s+to|"
    r"for\s+the\s+purposes\s+of|in\s+relation\s+to)\b\s+([^.;:]{8,200})",
    re.IGNORECASE,
)

# ── Threshold patterns ──────────────────────────────────────────────────────
THRESH_RE = re.compile(
    r"(?:(?:at\s+least|no\s+(?:less|more)\s+than|minimum|maximum|"
    r"not\s+(?:less|more|exceed)|exceed(?:s|ing)?|less\s+than|"
    r"greater\s+than|equal\s+to|up\s+to)\s+)?"
    r"(?:USD|EUR|GBP|\$|€|£)?\s?\d[\d,]*(?:\.\d+)?\s?(?:%|percent|"
    r"basis\s+points?|bps|million|billion|days?|months?|years?|"
    r"hours?)?",
    re.IGNORECASE,
)

# ── Deadline patterns ───────────────────────────────────────────────────────
DEADLINE_RE = re.compile(
    r"\b(?:within|by|before|after|no\s+later\s+than|prior\s+to|"
    r"on\s+or\s+before|not\s+later\s+than)\s+"
    r"(?:\d+\s+(?:days?|months?|years?|hours?|business\s+days?|"
    r"working\s+days?)|"
    r"(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(?:\s+\d{4})?|"
    r"\d{4}|"
    r"the\s+end\s+of\s+(?:the\s+)?(?:year|month|quarter|reporting\s+period))",
    re.IGNORECASE,
)

# ── Exception patterns ──────────────────────────────────────────────────────
EXCEPT_RE = re.compile(
    r"\b(?:unless|except|except\s+(?:that|where|when|for|as)|"
    r"with\s+the\s+exception\s+of|other\s+than|excluding|"
    r"save\s+(?:that|where|for))\s+([^.;:]{5,200})",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_tuple(text: str) -> dict:
    """Apply rules; return 6-field dict (None where no match)."""
    t = {f: None for f in TUPLE_FIELDS}
    if not text or not text.strip():
        return t
    text = text.replace("\n", " ")

    # Subject — first matching noun phrase
    m = SUBJECT_RE.search(text)
    if m:
        t["subject"] = _norm(m.group(0)).lower()

    # Obligation — verb after the trigger, up to next clause boundary
    m = OBL_TRIGGER_RE.search(text)
    if m:
        tail = text[m.end():m.end() + 250]
        # Clip at next strong boundary
        end = re.search(r"[.;:](?:\s|$)|\bunless\b|\bexcept\b|\bif\b|\bwhere\b", tail, re.IGNORECASE)
        obl = tail[: end.start()] if end else tail
        if obl.strip():
            t["obligation"] = _norm(obl)

    # Condition
    m = COND_RE.search(text)
    if m:
        cond_body = m.group(1) or ""
        if cond_body.strip():
            t["condition"] = _norm(cond_body)

    # Threshold — pick the first numeric pattern that looks regulatory
    th = THRESH_RE.findall(text)
    if th:
        # Prefer one with %/units; else first one
        with_units = [x for x in th if re.search(r"%|percent|basis|million|billion|days?|months?|years?", x, re.I)]
        chosen = with_units[0] if with_units else th[0]
        if chosen and chosen.strip():
            t["threshold"] = _norm(chosen)

    # Deadline
    m = DEADLINE_RE.search(text)
    if m:
        t["deadline"] = _norm(m.group(0))

    # Exception
    m = EXCEPT_RE.search(text)
    if m:
        exc = m.group(1) or ""
        if exc.strip():
            t["exception"] = _norm(exc)

    return t


def process_doc(doc_id: str) -> tuple[list, dict]:
    cpath = PROCESSED_DIR / f"{doc_id}_clauses.json"
    with open(cpath, encoding="utf-8") as f:
        clauses = json.load(f).get("clauses", [])

    parsed = []
    for c in clauses:
        tup = extract_tuple(c.get("text", ""))
        parsed.append({**c, "tuple": tup, "model_used": "baseline-regex"})

    out = PROCESSED_DIR / f"{doc_id}_tuples_baseline.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"doc_id": doc_id, "model": "baseline-regex",
                   "total_clauses": len(clauses), "parsed_tuples": parsed},
                  f, indent=2, ensure_ascii=False)

    rcc = compute_rcc(parsed, len(clauses))["value"]
    oal = compute_oal(parsed, None)["value"]   # Branch-B (matches the corrected paper)

    # RIF: use gold-aligned 0.4/0.3/0.3 if gold exists for this doc, else heuristic
    if doc_id == "basel3":
        with open(ANNOTATIONS_DIR / "gold_standard_reviewed.csv", encoding="utf-8") as f:
            gold = list(csv.DictReader(f))
        rif = compute_rif_with_gold(parsed, clauses, gold)
    elif doc_id == "gdpr":
        gpath = ANNOTATIONS_DIR / "gdpr_gold_standard.csv"
        if gpath.exists():
            with open(gpath, encoding="utf-8") as f:
                gold = list(csv.DictReader(f))
            rif = compute_rif_with_gold(parsed, clauses, gold)
        else:
            rif = compute_rif(parsed, clauses)["value"]
    else:  # regbi: same Basel3 gold (matches paper's regbi RIF approach)
        with open(ANNOTATIONS_DIR / "gold_standard_reviewed.csv", encoding="utf-8") as f:
            gold = list(csv.DictReader(f))
        rif = compute_rif_with_gold(parsed, clauses, gold)

    return parsed, {"doc": doc_id, "n_clauses": len(clauses),
                    "RCC": round(rcc, 4), "OAL": round(oal, 4), "RIF": round(rif, 4)}


def main():
    print("=== Rule-based baseline (no LLM calls) ===\n")
    results = []
    for doc_id in DOCS:
        print(f"  Processing {doc_id}...")
        _, m = process_doc(doc_id)
        results.append(m)
        print(f"    n={m['n_clauses']}  RCC={m['RCC']:.4f}  OAL={m['OAL']:.4f}  RIF={m['RIF']:.4f}")

    out_path = OUTPUTS_DIR / "reports" / "baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {out_path.name}")


if __name__ == "__main__":
    main()
