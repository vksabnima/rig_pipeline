#!/usr/bin/env python3
"""
Recompute HRR split into HRR-Detection and HRR-Stability per model.

HRR-Detection: % of hallucination bait clauses where model correctly
               returns obligation_present=false or confidence < 0.4.

HRR-Stability: % of adversarial perturbations where model output
               matches the baseline (non-perturbed) output.
"""

import csv
import io
import json
import os
import random
import re
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

from config.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    PROCESSED_DIR, OUTPUTS_DIR, ANNOTATIONS_DIR, RANDOM_SEED,
)
from src.adversarial_tester import PERTURBATION_FUNCTIONS

random.seed(RANDOM_SEED)

GRN = "\033[92m"; CYN = "\033[96m"; BLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"

DOCS = ["basel3", "regbi"]
MODELS = ["gpt-4o", "claude-sonnet", "llama3.2"]
TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]

STABILITY_SAMPLE = 20       # clauses to test stability on
PERTURBATION_TYPES = ["synonym_swap", "negation_injection", "clause_reorder",
                      "threshold_mutation", "entity_substitution"]

# =============================================================================
# Hallucination bait clauses — non-obligation text that looks regulatory
# =============================================================================

HALLUCINATION_BAITS = [
    # Descriptive/informational — no obligation
    "The Basel Committee on Banking Supervision was established in 1974 by the central bank governors of the Group of Ten countries.",
    "This section provides an overview of the regulatory framework and its historical development since the 2008 financial crisis.",
    "Market participants have expressed a range of views on the proposed amendments during the public consultation period.",
    "The standardised approach calculates risk-weighted assets by assigning exposures to predefined categories based on external credit ratings.",
    "Several commenters noted that the existing framework already addresses many of the concerns raised in the proposal.",
    # Opinion/analysis — no obligation
    "In the view of the Commission, the current regulatory environment is conducive to maintaining financial stability across jurisdictions.",
    "Industry stakeholders have generally supported the direction of reform, while noting implementation challenges.",
    "The impact assessment suggests that smaller institutions may face proportionally higher compliance costs.",
    # Aspirational/non-binding
    "The Committee encourages national supervisors to share best practices in the implementation of these standards.",
    "Firms are invited to consider whether additional voluntary measures might further strengthen their risk management frameworks.",
    # Random non-regulatory
    "The quarterly report on economic trends was published in March 2024 and is available on the central bank website.",
    "Annual conference attendance has grown by 15% over the past three years, reflecting increased interest in regulatory technology.",
    # Near-miss: looks like obligation but isn't
    "Previous versions of this regulation had required annual reporting, but that provision was removed in the 2019 amendment.",
    "Some jurisdictions may choose to implement stricter requirements, although this is not mandated by the international standard.",
    "The working group discussed whether institutions should be required to submit quarterly reports, but no decision was reached.",
]


# =============================================================================
# Detection prompt — asks model to assess obligation presence
# =============================================================================

DETECT_PROMPT = """Analyze this text from a financial regulation. Determine if it contains a genuine regulatory obligation.

Return ONLY a JSON object with these fields:
- obligation_present: true if the text contains a binding regulatory obligation (must/shall/required), false otherwise
- confidence: float 0.0-1.0 indicating how confident you are
- subject: the regulated entity (null if no obligation)
- obligation: the required action (null if no obligation)
- reasoning: one sentence explaining your judgment

Text: {text}"""

# =============================================================================
# Stability prompt — standard tuple extraction
# =============================================================================

PARSE_PROMPT = """Extract a JSON obligation tuple from this regulatory clause.
Fields: subject, obligation, condition, threshold, deadline, exception (null if absent).
Return ONLY JSON.

Clause: {text}"""


# =============================================================================
# Model parsers
# =============================================================================

def parse_batch_gpt4o(texts: list[str], prompt_template: str) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []
    for text in texts:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_template.format(text=text)}],
                temperature=0.0, seed=RANDOM_SEED,
                response_format={"type": "json_object"},
            )
            results.append(json.loads(resp.choices[0].message.content))
        except Exception as e:
            results.append({"error": str(e)})
    return results


def parse_batch_claude(texts: list[str], prompt_template: str) -> list[dict]:
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    for text in texts:
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=400,
                messages=[{"role": "user", "content": prompt_template.format(text=text)}],
            )
            content = resp.content[0].text
            s, e = content.find("{"), content.rfind("}") + 1
            results.append(json.loads(content[s:e]) if s >= 0 and e > s else {"error": "no json"})
        except Exception as e:
            results.append({"error": str(e)})
    return results


def parse_batch_llama(texts: list[str], prompt_template: str) -> list[dict]:
    import requests
    results = []
    for text in texts:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt_template.format(text=text[:500]),
                    "stream": False,
                    "options": {"temperature": 0.0, "seed": RANDOM_SEED, "num_predict": 250},
                },
                timeout=600,
            )
            content = resp.json().get("response", "{}")
            m = re.search(r"\{.*\}", content, re.DOTALL)
            results.append(json.loads(m.group()) if m else {"error": "no json"})
        except Exception as e:
            results.append({"error": str(e)})
    return results


PARSERS = {
    "gpt-4o": parse_batch_gpt4o,
    "claude-sonnet": parse_batch_claude,
    "llama3.2": parse_batch_llama,
}


# =============================================================================
# HRR-Detection
# =============================================================================

def compute_hrr_detection(model_name: str) -> dict:
    """Run hallucination baits through the model and measure rejection rate."""
    print(f"    Sending {len(HALLUCINATION_BAITS)} bait clauses...")
    parser = PARSERS[model_name]
    results = parser(HALLUCINATION_BAITS, DETECT_PROMPT)

    correct_rejections = 0
    details = []

    for i, (bait, result) in enumerate(zip(HALLUCINATION_BAITS, results)):
        obl_present = result.get("obligation_present", True)
        confidence = result.get("confidence", 1.0)

        # Normalize: handle string "false"/"true"
        if isinstance(obl_present, str):
            obl_present = obl_present.lower() not in ("false", "no", "0")

        # Try to parse confidence as float
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0

        # Correct if: obligation_present == false OR confidence < 0.4
        is_correct = (not obl_present) or (confidence < 0.4)
        if is_correct:
            correct_rejections += 1

        details.append({
            "bait_idx": i,
            "obligation_present": obl_present,
            "confidence": confidence,
            "correct_rejection": is_correct,
            "bait_snippet": bait[:80],
        })

    rate = correct_rejections / max(len(HALLUCINATION_BAITS), 1)
    print(f"    Correct rejections: {correct_rejections}/{len(HALLUCINATION_BAITS)} ({rate:.1%})")

    return {
        "hrr_detection": round(rate, 4),
        "correct_rejections": correct_rejections,
        "total_baits": len(HALLUCINATION_BAITS),
        "details": details,
    }


# =============================================================================
# HRR-Stability
# =============================================================================

def tuples_match(t1: dict, t2: dict) -> bool:
    """Check if two tuples are functionally equivalent."""
    for f in TUPLE_FIELDS:
        v1 = t1.get(f)
        v2 = t2.get(f)
        # Both None = match
        if v1 is None and v2 is None:
            continue
        # One None = mismatch
        if v1 is None or v2 is None:
            return False
        # Compare as lowercase strings
        if str(v1).lower().strip() != str(v2).lower().strip():
            return False
    return True


def compute_hrr_stability(model_name: str, all_tuples: list[dict]) -> dict:
    """Perturb clauses, re-parse, compare to baseline."""
    # Select clauses with at least 2 non-null tuple fields
    good = [t for t in all_tuples
            if sum(1 for f in TUPLE_FIELDS if t.get("tuple", {}).get(f) is not None) >= 2]

    if len(good) < 5:
        # Fallback for sparse models (Llama3.2): use all tuples with any content
        good = [t for t in all_tuples
                if any(t.get("tuple", {}).get(f) is not None for f in TUPLE_FIELDS)]

    sample = random.sample(good, min(STABILITY_SAMPLE, len(good))) if good else []

    if not sample:
        print(f"    No tuples with content to test stability")
        return {"hrr_stability": 0.0, "stable": 0, "total_tests": 0}

    parser = PARSERS[model_name]

    # Step 1: Parse originals to get baseline
    original_texts = [s.get("text", "") for s in sample]
    print(f"    Parsing {len(original_texts)} baseline clauses...")
    baseline_tuples = parser(original_texts, PARSE_PROMPT)

    # Step 2: Generate perturbations for each original
    perturbed_jobs = []  # (original_idx, ptype, perturbed_text)
    for idx, clause in enumerate(sample):
        text = clause.get("text", "")
        for ptype in PERTURBATION_TYPES:
            fn = PERTURBATION_FUNCTIONS.get(ptype)
            if fn:
                result = fn(text)
                if result.get("changes"):
                    perturbed_jobs.append((idx, ptype, result["text"]))

    print(f"    Parsing {len(perturbed_jobs)} perturbed variants...")
    perturbed_texts = [j[2] for j in perturbed_jobs]
    perturbed_tuples = parser(perturbed_texts, PARSE_PROMPT) if perturbed_texts else []

    # Step 3: Compare
    stable_count = 0
    total_tests = len(perturbed_jobs)

    for i, (orig_idx, ptype, _) in enumerate(perturbed_jobs):
        if i >= len(perturbed_tuples):
            break
        if tuples_match(baseline_tuples[orig_idx], perturbed_tuples[i]):
            stable_count += 1

    rate = stable_count / max(total_tests, 1)
    print(f"    Stable outputs: {stable_count}/{total_tests} ({rate:.1%})")

    return {
        "hrr_stability": round(rate, 4),
        "stable": stable_count,
        "total_tests": total_tests,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()
    print(f"\n{CYN}{BLD}=== HRR Split: Detection + Stability Per Model ==={RST}\n")

    # Load parsed tuples per model
    all_tuples_by_model = {}
    for model_name in MODELS:
        tuples = []
        for doc_id in DOCS:
            if model_name == "gpt-4o":
                path = PROCESSED_DIR / f"{doc_id}_tuples.json"
            else:
                path = PROCESSED_DIR / f"{doc_id}_tuples_{model_name.replace('-', '_')}.json"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            tuples.extend(data.get("parsed_tuples", []))
        all_tuples_by_model[model_name] = tuples

    hrr_results = {}

    for model_name in MODELS:
        print(f"{BLD}--- {model_name} ---{RST}")

        # Detection
        print(f"  HRR-Detection:")
        det = compute_hrr_detection(model_name)

        # Stability
        print(f"  HRR-Stability:")
        stab = compute_hrr_stability(model_name, all_tuples_by_model[model_name])

        hrr_results[model_name] = {
            "HRR_detection": det["hrr_detection"],
            "HRR_stability": stab["hrr_stability"],
            "detection_details": det,
            "stability_details": stab,
        }
        print(f"  {CHECK} {model_name}: detection={det['hrr_detection']:.4f}, stability={stab['hrr_stability']:.4f}\n")

    # =========================================================================
    # Update multimodel_comparison.csv
    # =========================================================================
    csv_path = OUTPUTS_DIR / "reports" / "multimodel_comparison.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Add new columns
    for row in rows:
        model = row["model"]
        row["HRR_detection"] = hrr_results[model]["HRR_detection"]
        row["HRR_stability"] = hrr_results[model]["HRR_stability"]

    fieldnames = ["model", "document", "RCC", "OAL", "RIF", "CAS", "HRR", "HRR_detection", "HRR_stability"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save detailed HRR results
    hrr_path = OUTPUTS_DIR / "reports" / "hrr_detailed.json"
    with open(hrr_path, "w", encoding="utf-8") as f:
        json.dump(hrr_results, f, indent=2, default=str)

    # =========================================================================
    # Print final table
    # =========================================================================
    elapsed = round(time.time() - start, 1)

    print(f"\n{CYN}{BLD}{'='*95}")
    print(f"  UPDATED RESULTS TABLE (with HRR_detection + HRR_stability)")
    print(f"{'='*95}{RST}\n")

    header = (f"  {'Model':<18} {'Doc':<8} {'RCC':>7} {'OAL':>7} {'RIF':>7} "
              f"{'CAS':>7} {'HRR':>7} {'HRR_det':>8} {'HRR_stab':>9}")
    print(f"{BLD}{header}{RST}")
    print(f"  {'-'*88}")

    for row in rows:
        line = (f"  {row['model']:<18} {row['document']:<8} "
                f"{float(row['RCC']):>7.4f} {float(row['OAL']):>7.4f} {float(row['RIF']):>7.4f} "
                f"{float(row['CAS']):>7.4f} {float(row['HRR']):>7.4f} "
                f"{float(row['HRR_detection']):>8.4f} {float(row['HRR_stability']):>9.4f}")
        print(line)

    print(f"  {'-'*88}")
    for model_name in MODELS:
        model_rows = [r for r in rows if r["model"] == model_name]
        avg = {}
        for m in ["RCC", "OAL", "RIF", "CAS", "HRR", "HRR_detection", "HRR_stability"]:
            avg[m] = np.mean([float(r[m]) for r in model_rows])
        line = (f"  {model_name:<18} {'AVG':<8} "
                f"{avg['RCC']:>7.4f} {avg['OAL']:>7.4f} {avg['RIF']:>7.4f} "
                f"{avg['CAS']:>7.4f} {avg['HRR']:>7.4f} "
                f"{avg['HRR_detection']:>8.4f} {avg['HRR_stability']:>9.4f}")
        print(f"{BLD}{line}{RST}")

    print(f"\n  {CHECK} Updated {csv_path.relative_to(Path(__file__).parent)}")
    print(f"  {CHECK} Detailed HRR saved to {hrr_path.relative_to(Path(__file__).parent)}")
    print(f"\n{DIM}  Completed in {elapsed}s{RST}\n")


if __name__ == "__main__":
    main()
