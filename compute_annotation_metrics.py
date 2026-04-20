#!/usr/bin/env python3
"""
Run this AFTER your friends return the three filled worksheets.
Computes:
  1. Cohen's kappa on the combined 100+300=400-clause Basel III kappa set
  2. Reg BI RIF against the new human-reviewed Reg BI gold (100 clauses)
  3. GDPR RIF against the new independent human gold (50 clauses),
     excluding Claude-Sonnet-vs-Claude-gold circularity

Outputs:
  outputs/reports/kappa_400_results.json
  outputs/reports/regbi_rif_human_gold.json
  outputs/reports/gdpr_rif_human_gold.json

If any worksheet has < 50% of rows filled, that section is skipped with a warning.
"""

import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]
GT_FIELDS = ["subject_gt", "obligation_gt", "condition_gt", "threshold_gt", "deadline_gt", "exception_gt"]
HUMAN_FIELDS = [f"human_{f}" for f in FIELDS]
REGULATORY_KEYWORDS = {
    "shall", "must", "required", "prohibited", "ensure", "comply",
    "obligation", "mandatory", "limit", "minimum", "maximum", "capital",
    "risk", "exposure", "deadline", "within", "before", "after",
}


# ── Helpers ────────────────────────────────────────────────────────────────
def is_present(v):
    return v is not None and str(v).strip() != "" and str(v).strip().lower() != "null"


def substring_match(a, b):
    a_n = str(a).lower().strip()
    b_n = str(b).lower().strip()
    if not a_n or not b_n:
        return False
    return a_n in b_n or b_n in a_n


def cohen_kappa_binary(pairs):
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    a_yes = sum(1 for a, _ in pairs if a == 1) / n
    b_yes = sum(1 for _, b in pairs if b == 1) / n
    pe = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def is_completed(r):
    """A row is completed if EITHER any human_* field is filled, OR the
    annotator marked it as non-obligation (all human_* blank, notes filled)."""
    if any(is_present(r.get(h)) for h in HUMAN_FIELDS):
        return True
    note = (r.get("notes") or "").strip()
    return bool(note)


def count_filled(rows):
    return sum(1 for r in rows if is_completed(r))


# ── 1. Combined kappa on 100 + 300 = 400 clauses ───────────────────────────
def compute_combined_kappa():
    all_rows = []
    for name in ("annotated_kappa.csv", "kappa_300_worksheet.csv"):
        p = ANNOTATIONS_DIR / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            n_filled = count_filled(rows)
            print(f"  {name}: {len(rows)} rows, {n_filled} filled")
            if n_filled / max(len(rows), 1) < 0.5:
                print(f"    SKIP — less than half filled.")
                continue
            all_rows.extend(rows)
    if len(all_rows) < 100:
        print("  Not enough combined rows; skipping combined kappa.")
        return None

    per_field = {}
    all_pairs = []
    all_both_present_match = []
    for f in FIELDS:
        pairs = []
        both_present = []
        for r in all_rows:
            c_val = r.get(f"claude_{f}", "")
            h_val = r.get(f"human_{f}", "")
            c = 1 if is_present(c_val) else 0
            h = 1 if is_present(h_val) else 0
            pairs.append((c, h))
            if c and h:
                both_present.append(1 if substring_match(c_val, h_val) else 0)
        kappa = cohen_kappa_binary(pairs)
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        content_align = sum(both_present) / len(both_present) if both_present else None
        per_field[f] = {
            "n": len(pairs), "presence_agreement": round(agree, 4),
            "presence_kappa": round(kappa, 4) if kappa is not None else None,
            "n_both_present": len(both_present),
            "content_alignment_when_both_present":
                round(content_align, 4) if content_align is not None else None,
        }
        all_pairs.extend(pairs)
        all_both_present_match.extend(both_present)

    overall_k = cohen_kappa_binary(all_pairs)
    overall_ca = sum(all_both_present_match) / len(all_both_present_match) if all_both_present_match else None
    out = {
        "n_clauses": len(all_rows),
        "n_judgments": len(all_pairs),
        "per_field": per_field,
        "overall_presence_kappa": round(overall_k, 4) if overall_k is not None else None,
        "overall_content_alignment_when_both_present":
            round(overall_ca, 4) if overall_ca is not None else None,
    }
    out_p = OUTPUTS_DIR / "reports" / "kappa_400_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Combined kappa: {overall_k:.4f} over {len(all_pairs)} judgments (content align {overall_ca:.4f})")
    print(f"  Saved {out_p.name}")
    return out


# ── 2. Reg BI RIF against new human gold ──────────────────────────────────
def rif_single(t, text, gold):
    non_null = sum(1 for f in FIELDS if t.get(f) is not None)
    comp = non_null / 6
    orig_kw = set(text.lower().split()) & REGULATORY_KEYWORDS
    if orig_kw:
        tt = " ".join(str(v).lower() for v in t.values() if v is not None)
        ret = sum(1 for kw in orig_kw if kw in tt) / len(orig_kw)
    else:
        ret = 1.0
    if gold:
        mb, mc = 0, 0
        for tf, gf in zip(FIELDS, GT_FIELDS):
            tv, gv = t.get(tf), gold.get(gf)
            if tv is not None and gv and str(gv).strip():
                mc += 1
                a = str(tv).lower().strip(); b = str(gv).lower().strip()
                if a in b or b in a: mb += 1
        if mc > 0:
            return 0.4 * comp + 0.3 * ret + 0.3 * (mb / mc)
    return 0.5 * comp + 0.5 * ret


def human_to_gold_lookup(rows):
    """Convert completed rows into gold_standard-shaped lookup with *_gt keys.
    Rows that the annotator marked as non-obligation (all human_* blank but
    notes filled) are included as empty-gold rows, which falls through to
    the non-gold RIF branch when scored."""
    g = {}
    for r in rows:
        if not is_completed(r):
            continue
        cid = r.get("clause_id", "")
        g[cid] = {f"{f}_gt": r.get(f"human_{f}", "") for f in FIELDS}
    return g


def compute_rif_on_new_gold(doc_id, gold_worksheet, out_name, exclude_models=None):
    exclude_models = set(exclude_models or [])
    p = ANNOTATIONS_DIR / gold_worksheet
    if not p.exists():
        print(f"  {gold_worksheet} not found — skipping.")
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if count_filled(rows) / max(len(rows), 1) < 0.5:
        print(f"  {gold_worksheet}: less than half filled; skipping.")
        return None
    gold_lookup = human_to_gold_lookup(rows)
    if not gold_lookup:
        print(f"  {gold_worksheet}: no human annotations; skipping.")
        return None

    with open(PROCESSED_DIR / f"{doc_id}_clauses.json", encoding="utf-8") as f:
        clauses = json.load(f)["clauses"]
    clause_by_id = {c["clause_id"]: c for c in clauses}

    models = {
        "gpt-4o": f"{doc_id}_tuples.json",
        "claude-sonnet": f"{doc_id}_tuples_claude_sonnet.json",
        "qwen": f"{doc_id}_tuples_qwen.json",
        "llama3_8b": f"{doc_id}_tuples_llama3_8b.json",
    }
    results = {}
    for m, fname in models.items():
        if m in exclude_models:
            continue
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            tuples = json.load(f)["parsed_tuples"]
        scored = []
        for t in tuples:
            cid = t.get("clause_id", "")
            if cid not in gold_lookup:
                continue
            c = clause_by_id.get(cid, {})
            scored.append(rif_single(t.get("tuple", {}), c.get("text", ""), gold_lookup[cid]))
        if not scored:
            continue
        arr = np.array(scored)
        n_samp = max(int(0.8 * len(arr)), 1)
        boot = [float(np.mean(arr[np.random.choice(len(arr), n_samp, replace=True)]))
                for _ in range(1000)]
        results[m] = {
            "n": len(scored),
            "rif": round(float(np.mean(arr)), 4),
            "ci_95": [round(float(np.percentile(boot, 2.5)), 4),
                      round(float(np.percentile(boot, 97.5)), 4)],
        }
        print(f"    {m}: n={len(scored)}, RIF={results[m]['rif']} CI={results[m]['ci_95']}")

    out_p = OUTPUTS_DIR / "reports" / out_name
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"doc": doc_id, "n_gold": len(gold_lookup), "models": results},
                  f, indent=2)
    print(f"  Saved {out_p.name}")
    return results


def main():
    print("=== 1. Combined Basel III kappa (100 + 300 = 400 clauses) ===")
    compute_combined_kappa()
    print("\n=== 2. Reg BI RIF against new human gold ===")
    compute_rif_on_new_gold("regbi", "regbi_gold_worksheet.csv", "regbi_rif_human_gold.json")
    print("\n=== 3. GDPR RIF against independent human gold ===")
    print("    (excluding Claude Sonnet to avoid circularity)")
    compute_rif_on_new_gold("gdpr", "gdpr_human_worksheet.csv",
                             "gdpr_rif_human_gold.json",
                             exclude_models=["claude-sonnet"])


if __name__ == "__main__":
    main()
