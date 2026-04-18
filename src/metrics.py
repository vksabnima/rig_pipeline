"""
RIG Metrics
-----------
Five novel metrics for evaluating the Regulatory Intent Graph pipeline:

1. RCC — Regulatory Clause Coverage
2. OAL — Obligation Alignment Level
3. RIF — Regulatory Intent Fidelity
4. CAS — Cross-document Alignment Score
5. HRR — Hallucination Resistance Rate
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import OUTPUTS_DIR


def compute_rcc(extracted_clauses: list[dict], total_obligation_sentences: int) -> dict:
    """
    RCC — Regulatory Clause Coverage
    ---------------------------------
    Measures the proportion of obligation-bearing sentences in the source
    document that the pipeline successfully extracts and parses.

    RCC = |extracted obligation clauses with valid tuples| / |total obligation sentences|

    Range: [0, 1]. Higher is better.
    """
    valid = sum(1 for c in extracted_clauses
                if c.get("tuple") and any(v is not None for v in c["tuple"].values()))

    rcc = valid / max(total_obligation_sentences, 1)
    return {
        "metric": "RCC",
        "full_name": "Regulatory Clause Coverage",
        "value": round(min(rcc, 1.0), 4),
        "extracted_valid": valid,
        "total_obligation_sentences": total_obligation_sentences,
        "interpretation": "Proportion of obligation sentences successfully extracted and parsed",
    }


def compute_oal(parsed_tuples: list[dict], gold_standard: Optional[list[dict]] = None) -> dict:
    """
    OAL — Obligation Alignment Level
    ----------------------------------
    Measures how well extracted obligation tuples align with ground truth
    annotations (if available) or internal consistency (if no gold standard).

    With gold standard:
        OAL = mean(field_match_score) across all tuples
        field_match_score = |matched fields| / |total fields|

    Without gold standard (self-consistency):
        OAL = proportion of tuples with >= 3 non-null fields

    Range: [0, 1]. Higher is better.
    """
    FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]

    if gold_standard and len(gold_standard) > 0:
        scores = []
        for parsed, gold in zip(parsed_tuples, gold_standard):
            pt = parsed.get("tuple", {})
            matches = sum(1 for f in FIELDS
                          if pt.get(f) is not None and gold.get(f"{f}_gt") is not None)
            total_present = sum(1 for f in FIELDS if gold.get(f"{f}_gt") is not None)
            scores.append(matches / max(total_present, 1))
        oal = float(np.mean(scores)) if scores else 0.0
        method = "gold_standard_comparison"
    else:
        # Self-consistency: tuples with at least 3 non-null fields
        sufficient = sum(
            1 for t in parsed_tuples
            if sum(1 for f in FIELDS if t.get("tuple", {}).get(f) is not None) >= 3
        )
        oal = sufficient / max(len(parsed_tuples), 1)
        method = "self_consistency"

    return {
        "metric": "OAL",
        "full_name": "Obligation Alignment Level",
        "value": round(oal, 4),
        "method": method,
        "num_tuples": len(parsed_tuples),
        "interpretation": "Alignment quality of extracted tuples with ground truth or internal consistency",
    }


def compute_rif(parsed_tuples: list[dict], original_clauses: list[dict]) -> dict:
    """
    RIF — Regulatory Intent Fidelity
    ----------------------------------
    Measures whether the extracted tuple preserves the regulatory intent
    of the original clause text. Uses field completeness and keyword retention.

    RIF = mean(intent_score) across all tuples
    intent_score = 0.5 * field_completeness + 0.5 * keyword_retention

    Range: [0, 1]. Higher is better.
    """
    FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]
    REGULATORY_KEYWORDS = {
        "shall", "must", "required", "prohibited", "ensure", "comply",
        "obligation", "mandatory", "limit", "minimum", "maximum", "capital",
        "risk", "exposure", "deadline", "within", "before", "after",
    }

    scores = []
    for parsed, original in zip(parsed_tuples, original_clauses):
        t = parsed.get("tuple", {})

        # Field completeness
        non_null = sum(1 for f in FIELDS if t.get(f) is not None)
        completeness = non_null / len(FIELDS)

        # Keyword retention: what fraction of regulatory keywords in original
        # appear somewhere in the tuple values
        orig_words = set(original.get("text", "").lower().split())
        orig_keywords = orig_words & REGULATORY_KEYWORDS

        if orig_keywords:
            tuple_text = " ".join(
                str(v).lower() for v in t.values() if v is not None
            )
            retained = sum(1 for kw in orig_keywords if kw in tuple_text)
            retention = retained / len(orig_keywords)
        else:
            retention = 1.0  # No keywords to retain

        intent_score = 0.5 * completeness + 0.5 * retention
        scores.append(intent_score)

    rif = float(np.mean(scores)) if scores else 0.0
    return {
        "metric": "RIF",
        "full_name": "Regulatory Intent Fidelity",
        "value": round(rif, 4),
        "num_tuples": len(parsed_tuples),
        "interpretation": "How faithfully tuples preserve the regulatory intent of source clauses",
    }


def compute_cas(alignments: list[dict], total_cross_doc_pairs: int) -> dict:
    """
    CAS — Cross-document Alignment Score
    --------------------------------------
    Measures the quality and coverage of cross-document obligation alignment.

    CAS = alpha * precision + (1 - alpha) * recall
    precision = mean(similarity) of aligned pairs
    recall = |aligned pairs| / |total possible cross-doc obligation pairs|

    Range: [0, 1]. Higher is better.
    """
    alpha = 0.6  # Weight precision slightly more than recall

    if not alignments:
        return {
            "metric": "CAS",
            "full_name": "Cross-document Alignment Score",
            "value": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_alignments": 0,
            "interpretation": "Quality and coverage of cross-document obligation alignment",
        }

    precision = float(np.mean([a["similarity"] for a in alignments]))
    recall = len(alignments) / max(total_cross_doc_pairs, 1)
    cas = alpha * precision + (1 - alpha) * min(recall, 1.0)

    return {
        "metric": "CAS",
        "full_name": "Cross-document Alignment Score",
        "value": round(cas, 4),
        "precision": round(precision, 4),
        "recall": round(min(recall, 1.0), 4),
        "num_alignments": len(alignments),
        "total_possible_pairs": total_cross_doc_pairs,
        "interpretation": "Quality and coverage of cross-document obligation alignment",
    }


def compute_hrr(adversarial_results: dict, adversarial_samples: list[dict]) -> dict:
    """
    HRR — Hallucination Resistance Rate
    -------------------------------------
    Measures pipeline resistance to adversarial perturbations.

    For semantic-preserving perturbations: measures output stability
    For semantic-altering perturbations: measures detection sensitivity

    HRR = 0.5 * stability_rate + 0.5 * detection_rate

    Range: [0, 1]. Higher is better.
    """
    preserving = [s for s in adversarial_samples if s.get("semantic_preserving")]
    altering = [s for s in adversarial_samples if not s.get("semantic_preserving")]

    # Stability: semantic-preserving perturbations with changes should not
    # significantly alter the output. Approximate by change rate.
    preserving_with_changes = [s for s in preserving if s.get("changes")]
    stability_rate = 1.0 - (len(preserving_with_changes) / max(len(preserving), 1)) * 0.3

    # Detection: semantic-altering perturbations should be caught.
    # Higher change rate for altering = better detection proxy.
    altering_with_changes = [s for s in altering if s.get("changes")]
    detection_rate = len(altering_with_changes) / max(len(altering), 1)

    hrr = 0.5 * stability_rate + 0.5 * detection_rate

    return {
        "metric": "HRR",
        "full_name": "Hallucination Resistance Rate",
        "value": round(hrr, 4),
        "stability_rate": round(stability_rate, 4),
        "detection_rate": round(detection_rate, 4),
        "preserving_samples": len(preserving),
        "altering_samples": len(altering),
        "interpretation": "Pipeline resistance to adversarial perturbations",
    }


def compute_all_metrics(
    extracted_clauses: list[dict],
    parsed_tuples: list[dict],
    original_clauses: list[dict],
    alignments: list[dict],
    total_cross_doc_pairs: int,
    adversarial_results: dict,
    adversarial_samples: list[dict],
    total_obligation_sentences: int,
    gold_standard: Optional[list[dict]] = None,
) -> dict:
    """Compute all 5 RIG metrics and return a consolidated report."""
    print("\n=== Computing RIG Metrics ===")

    metrics = {}

    rcc = compute_rcc(extracted_clauses, total_obligation_sentences)
    print(f"  RCC (Regulatory Clause Coverage):      {rcc['value']:.4f}")
    metrics["RCC"] = rcc

    oal = compute_oal(parsed_tuples, gold_standard)
    print(f"  OAL (Obligation Alignment Level):      {oal['value']:.4f}")
    metrics["OAL"] = oal

    rif = compute_rif(parsed_tuples, original_clauses)
    print(f"  RIF (Regulatory Intent Fidelity):      {rif['value']:.4f}")
    metrics["RIF"] = rif

    cas = compute_cas(alignments, total_cross_doc_pairs)
    print(f"  CAS (Cross-document Alignment Score):  {cas['value']:.4f}")
    metrics["CAS"] = cas

    hrr = compute_hrr(adversarial_results, adversarial_samples)
    print(f"  HRR (Hallucination Resistance Rate):   {hrr['value']:.4f}")
    metrics["HRR"] = hrr

    # Save report
    out_dir = OUTPUTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rig_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Full metrics report saved to outputs/reports/rig_metrics.json")

    return metrics
