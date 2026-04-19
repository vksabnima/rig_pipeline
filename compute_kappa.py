#!/usr/bin/env python3
"""
Compute Cohen's kappa between human annotations and Claude Sonnet
auto-annotations on the 100-clause kappa_worksheet.csv.

For each field f and each row i, both annotators emit a label that is
mapped to one of three categories used for kappa computation:
  AGREE_NULL     — both said "field absent"
  AGREE_PRESENT  — both said "field present" with substring overlap
                   (lowercase + strip; either direction containment)
  DISAGREE       — one said absent and the other present, OR
                   both present but no substring overlap

Cohen's kappa is then computed on the 2x2 (or 3x3 collapsed) table.

Run:   python compute_kappa.py
Out:   outputs/reports/kappa_results.json
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config.config import ANNOTATIONS_DIR, OUTPUTS_DIR

FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]


def cohen_kappa_binary(pairs):
    """Cohen's kappa for binary agree/disagree judgments."""
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n

    # Marginals
    a_yes = sum(1 for a, _ in pairs if a == 1) / n
    b_yes = sum(1 for _, b in pairs if b == 1) / n
    pe = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def is_present(s):
    return s is not None and str(s).strip() != "" and str(s).strip().lower() != "null"


def substring_match(a, b):
    a_norm = str(a).lower().strip()
    b_norm = str(b).lower().strip()
    if not a_norm or not b_norm:
        return False
    return a_norm in b_norm or b_norm in a_norm


def main():
    # Look for filled worksheet under either name
    candidates = ["annotated_kappa.csv", "kappa_worksheet.csv"]
    path = None
    for name in candidates:
        p = ANNOTATIONS_DIR / name
        if p.exists():
            path = p
            break
    if path is None:
        print(f"No worksheet found. Looked for: {candidates}")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Reading: {path.name}")

    n_filled = sum(1 for r in rows if any(is_present(r.get(f"human_{f}")) for f in FIELDS))
    print(f"Loaded {len(rows)} rows; {n_filled} have at least one human field filled.")

    if n_filled == 0:
        print("No human annotations found. Fill kappa_worksheet.csv first.")
        return

    per_field = {}
    for f in FIELDS:
        # Presence κ: binary "field present" judgment per rater.
        pairs = []
        for r in rows:
            claude_val = r.get(f"claude_{f}", "")
            human_val = r.get(f"human_{f}", "")
            c_present = 1 if is_present(claude_val) else 0
            h_present = 1 if is_present(human_val) else 0
            pairs.append((c_present, h_present))
        presence_kappa = cohen_kappa_binary(pairs)
        agree = sum(1 for a, b in pairs if a == b)

        # Content alignment | both present:
        # Among rows where BOTH raters marked the field present, what fraction
        # have bidirectional substring containment? (Raw fraction, not κ —
        # there is no chance baseline once both have committed to "present".)
        both_present = []
        for r in rows:
            cv = r.get(f"claude_{f}", "")
            hv = r.get(f"human_{f}", "")
            if is_present(cv) and is_present(hv):
                both_present.append(1 if substring_match(cv, hv) else 0)
        content_align = (sum(both_present) / len(both_present)) if both_present else None

        per_field[f] = {
            "n": len(pairs),
            "presence_agreement": round(agree / len(pairs), 4),
            "presence_kappa": round(presence_kappa, 4) if presence_kappa is not None else None,
            "n_both_present": len(both_present),
            "content_alignment_when_both_present":
                round(content_align, 4) if content_align is not None else None,
        }

    # Overall presence κ pooled across all 6 fields
    all_pairs = []
    all_both_present = []
    for r in rows:
        for f in FIELDS:
            c = is_present(r.get(f"claude_{f}", ""))
            h = is_present(r.get(f"human_{f}", ""))
            all_pairs.append((1 if c else 0, 1 if h else 0))
            if c and h:
                all_both_present.append(
                    1 if substring_match(r.get(f"claude_{f}", ""),
                                         r.get(f"human_{f}", "")) else 0
                )
    overall_presence_kappa = cohen_kappa_binary(all_pairs)
    overall_content_align = (sum(all_both_present) / len(all_both_present)) \
                             if all_both_present else None

    print(f"\n{'Field':<14} {'n':>4} {'Pres.Agree':>11} {'Pres.κ':>8} {'BothPres':>9} {'Cont.Align':>11}")
    print("-" * 64)
    for f in FIELDS:
        r = per_field[f]
        pa = f"{r['presence_agreement']:.3f}" if r['presence_agreement'] is not None else "—"
        pk = f"{r['presence_kappa']:.3f}" if r['presence_kappa'] is not None else "—"
        ca = f"{r['content_alignment_when_both_present']:.3f}" \
             if r['content_alignment_when_both_present'] is not None else "—"
        print(f"{f:<14} {r['n']:>4} {pa:>11} {pk:>8} {r['n_both_present']:>9} {ca:>11}")
    print("-" * 64)
    overall_ca = f"{overall_content_align:.3f}" if overall_content_align is not None else "—"
    print(f"{'OVERALL':<14} {len(all_pairs):>4} {'':>11} {overall_presence_kappa:>8.3f} {len(all_both_present):>9} {overall_ca:>11}")

    out = {
        "n_clauses": len(rows),
        "n_with_any_human_label": n_filled,
        "per_field": per_field,
        "overall_presence_kappa": round(overall_presence_kappa, 4),
        "overall_content_alignment_when_both_present":
            round(overall_content_align, 4) if overall_content_align is not None else None,
    }
    out_path = OUTPUTS_DIR / "reports" / "kappa_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path.name}")


if __name__ == "__main__":
    main()
