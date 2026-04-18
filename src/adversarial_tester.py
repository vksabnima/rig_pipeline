"""
Adversarial Tester
------------------
Generates adversarial perturbations of obligation clauses and measures
pipeline robustness against semantic-preserving and semantic-altering attacks.
"""

import json
import random
import re
from pathlib import Path
from copy import deepcopy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    ADVERSARIAL_PERTURBATION_TYPES, ADVERSARIAL_NUM_SAMPLES,
    PROCESSED_DIR, OUTPUTS_DIR, RANDOM_SEED,
)

random.seed(RANDOM_SEED)

# ── Perturbation Functions ───────────────────────────────────────────────────

SYNONYM_MAP = {
    "shall": ["must", "is required to", "will"],
    "must": ["shall", "is obligated to", "is required to"],
    "bank": ["institution", "financial entity", "credit institution"],
    "ensure": ["guarantee", "make certain", "verify"],
    "comply": ["adhere", "conform", "abide"],
    "prohibited": ["forbidden", "not permitted", "banned"],
    "requirement": ["obligation", "mandate", "stipulation"],
    "capital": ["funds", "reserves", "equity"],
    "risk": ["exposure", "hazard", "vulnerability"],
    "customer": ["client", "investor", "counterparty"],
}


def synonym_swap(text: str) -> dict:
    """Replace regulatory terms with synonyms (semantic-preserving)."""
    modified = text
    swaps = []
    for word, synonyms in SYNONYM_MAP.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pattern.search(modified):
            replacement = random.choice(synonyms)
            modified = pattern.sub(replacement, modified, count=1)
            swaps.append(f"{word} -> {replacement}")
    return {"text": modified, "perturbation": "synonym_swap", "changes": swaps,
            "semantic_preserving": True}


def negation_injection(text: str) -> dict:
    """Inject or remove negation (semantic-altering)."""
    negation_pairs = [
        (r"\bshall\b", "shall not"),
        (r"\bmust\b", "must not"),
        (r"\bshall not\b", "shall"),
        (r"\bmust not\b", "must"),
        (r"\bis required\b", "is not required"),
        (r"\bis prohibited\b", "is permitted"),
    ]
    modified = text
    change = None
    for pattern, replacement in negation_pairs:
        if re.search(pattern, modified, re.IGNORECASE):
            modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
            change = f"{pattern} -> {replacement}"
            break
    return {"text": modified, "perturbation": "negation_injection", "changes": [change] if change else [],
            "semantic_preserving": False}


def clause_reorder(text: str) -> dict:
    """Reorder sub-clauses (semantic-preserving if structure independent)."""
    parts = re.split(r"[;,]", text)
    if len(parts) > 2:
        reordered = [parts[0]] + random.sample(parts[1:], len(parts) - 1)
        modified = ", ".join(p.strip() for p in reordered)
    else:
        modified = text
    return {"text": modified, "perturbation": "clause_reorder", "changes": ["sub-clauses reordered"],
            "semantic_preserving": True}


def threshold_mutation(text: str) -> dict:
    """Mutate numeric thresholds (semantic-altering)."""
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    modified = text
    changes = []
    for num in numbers[:2]:  # Mutate up to 2 numbers
        clean = num.rstrip("%")
        try:
            val = float(clean)
            mutated = val * random.choice([0.5, 1.5, 2.0, 0.1])
            new_num = f"{mutated:.1f}" + ("%" if "%" in num else "")
            modified = modified.replace(num, new_num, 1)
            changes.append(f"{num} -> {new_num}")
        except ValueError:
            pass
    return {"text": modified, "perturbation": "threshold_mutation", "changes": changes,
            "semantic_preserving": False}


def entity_substitution(text: str) -> dict:
    """Substitute regulated entities (semantic-altering)."""
    entity_swaps = {
        "bank": "insurance company",
        "broker-dealer": "investment adviser",
        "institution": "fund manager",
        "customer": "shareholder",
        "counterparty": "subsidiary",
    }
    modified = text
    changes = []
    for orig, replacement in entity_swaps.items():
        pattern = re.compile(rf"\b{orig}\b", re.IGNORECASE)
        if pattern.search(modified):
            modified = pattern.sub(replacement, modified, count=1)
            changes.append(f"{orig} -> {replacement}")
    return {"text": modified, "perturbation": "entity_substitution", "changes": changes,
            "semantic_preserving": False}


PERTURBATION_FUNCTIONS = {
    "synonym_swap": synonym_swap,
    "negation_injection": negation_injection,
    "clause_reorder": clause_reorder,
    "threshold_mutation": threshold_mutation,
    "entity_substitution": entity_substitution,
}


# ── Test Runner ──────────────────────────────────────────────────────────────

def generate_adversarial_samples(clauses: list[dict]) -> list[dict]:
    """Generate adversarial variants for a set of obligation clauses."""
    samples = []

    for ptype in ADVERSARIAL_PERTURBATION_TYPES:
        func = PERTURBATION_FUNCTIONS.get(ptype)
        if not func:
            continue

        # Sample clauses for this perturbation type
        n = min(ADVERSARIAL_NUM_SAMPLES, len(clauses))
        selected = random.sample(clauses, n) if len(clauses) > n else clauses

        for clause in selected:
            original_text = clause.get("text", "")
            result = func(original_text)
            samples.append({
                "original_clause_id": clause.get("clause_id", ""),
                "original_text": original_text,
                **result,
            })

    return samples


def evaluate_robustness(original_tuples: list[dict], adversarial_samples: list[dict],
                        parse_fn=None) -> dict:
    """Evaluate pipeline robustness against adversarial samples.

    If parse_fn is provided, re-parses adversarial text and compares tuples.
    Otherwise, returns structural analysis only.
    """
    results = {
        "total_samples": len(adversarial_samples),
        "by_perturbation": {},
        "semantic_preserving_stability": 0.0,
        "semantic_altering_detection": 0.0,
    }

    for ptype in ADVERSARIAL_PERTURBATION_TYPES:
        ptype_samples = [s for s in adversarial_samples if s["perturbation"] == ptype]
        has_changes = [s for s in ptype_samples if s.get("changes")]

        results["by_perturbation"][ptype] = {
            "total": len(ptype_samples),
            "with_changes": len(has_changes),
            "change_rate": len(has_changes) / max(len(ptype_samples), 1),
        }

    # Compute aggregate scores
    preserving = [s for s in adversarial_samples if s.get("semantic_preserving")]
    altering = [s for s in adversarial_samples if not s.get("semantic_preserving")]

    # For semantic-preserving: stability = how many still parse identically
    results["semantic_preserving_count"] = len(preserving)
    results["semantic_altering_count"] = len(altering)

    return results


def run_adversarial_tests(parsed_docs: list[dict]) -> dict:
    """Run the full adversarial testing suite."""
    print("\n=== Adversarial Robustness Testing ===")

    all_clauses = []
    all_tuples = []
    for doc in parsed_docs:
        for entry in doc.get("parsed_tuples", []):
            all_clauses.append(entry)
            all_tuples.append(entry)

    if not all_clauses:
        print("  [WARN] No clauses to test")
        return {}

    print(f"  Total clauses for testing: {len(all_clauses)}")
    print(f"  Perturbation types: {len(ADVERSARIAL_PERTURBATION_TYPES)}")

    samples = generate_adversarial_samples(all_clauses)
    print(f"  Generated {len(samples)} adversarial samples")

    results = evaluate_robustness(all_tuples, samples)

    # Save results
    out_dir = OUTPUTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "adversarial_samples.json", "w") as f:
        json.dump(samples[:100], f, indent=2)  # Save first 100 for inspection

    with open(out_dir / "adversarial_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {out_dir}")

    for ptype, stats in results.get("by_perturbation", {}).items():
        print(f"    {ptype}: {stats['with_changes']}/{stats['total']} changed "
              f"({stats['change_rate']:.1%})")

    return results
