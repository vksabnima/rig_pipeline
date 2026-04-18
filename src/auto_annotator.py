"""
Auto Annotator
--------------
Uses Claude claude-sonnet-4-20250514 to pre-fill ground truth annotations for obligation
clauses, and sentence-transformers to find top cross-document obligation pairs.

Outputs:
  - data/annotations/gold_standard_prefilled.csv
  - data/annotations/cross_doc_pairs_prefilled.csv
"""

import csv
import json
from pathlib import Path

import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    ANTHROPIC_API_KEY, MODELS, EMBEDDING_MODEL,
    PROCESSED_DIR, ANNOTATIONS_DIR, RANDOM_SEED,
)

ANNOTATION_FIELDS = [
    "subject_gt", "obligation_gt", "condition_gt",
    "threshold_gt", "deadline_gt", "exception_gt",
]

ANNOTATION_PROMPT = """You are a regulatory compliance expert. Analyze this obligation clause from a financial regulation and extract the ground truth annotation.

Clause: {clause_text}

Return ONLY a JSON object with these fields (use null if not present):
- subject_gt: The regulated entity or actor
- obligation_gt: The required action or prohibition
- condition_gt: Triggering condition or scope qualifier
- threshold_gt: Quantitative limit or numeric requirement
- deadline_gt: Time constraint or compliance date
- exception_gt: Exemptions or carve-outs

Return only valid JSON, no markdown or explanation."""


def annotate_clauses_with_claude(clauses: list[dict]) -> list[dict]:
    """Use Claude claude-sonnet-4-20250514 to pre-fill ground truth for each clause."""
    if not ANTHROPIC_API_KEY:
        print("  [WARN] ANTHROPIC_API_KEY not set — generating empty annotations")
        return [
            {
                "clause_id": c.get("clause_id", ""),
                "doc_id": c.get("doc_id", ""),
                "section": c.get("section", ""),
                "text": c.get("text", ""),
                **{f: None for f in ANNOTATION_FIELDS},
                "human_reviewed": "FALSE",
            }
            for c in clauses
        ]

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    annotated = []

    for i, clause in enumerate(clauses):
        print(f"    Annotating clause {i+1}/{len(clauses)}: {clause.get('clause_id', '')}")
        try:
            response = client.messages.create(
                model=MODELS["annotator"],
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": ANNOTATION_PROMPT.format(clause_text=clause.get("text", "")),
                }],
            )
            content = response.content[0].text
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                annotation = json.loads(content[start:end])
            else:
                annotation = {f: None for f in ANNOTATION_FIELDS}
        except Exception as e:
            print(f"      [ERROR] {e}")
            annotation = {f: None for f in ANNOTATION_FIELDS}

        annotated.append({
            "clause_id": clause.get("clause_id", ""),
            "doc_id": clause.get("doc_id", ""),
            "section": clause.get("section", ""),
            "text": clause.get("text", "")[:500],
            **{f: annotation.get(f) for f in ANNOTATION_FIELDS},
            "human_reviewed": "FALSE",
        })

    return annotated


def save_gold_standard(annotated: list[dict], output_dir: Path) -> Path:
    """Save annotations to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gold_standard_prefilled.csv"

    fieldnames = ["clause_id", "doc_id", "section", "text"] + ANNOTATION_FIELDS + ["human_reviewed"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)

    print(f"  Saved {len(annotated)} annotations to {out_path.name}")
    return out_path


def find_cross_doc_pairs(clauses_by_doc: dict[str, list[dict]], top_k: int = 20) -> list[dict]:
    """Find top-k cross-document obligation pairs using sentence similarity."""
    print("  Loading sentence transformer for cross-doc pairing...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Collect all clauses with doc_id
    all_clauses = []
    for doc_id, clauses in clauses_by_doc.items():
        for clause in clauses:
            all_clauses.append({**clause, "doc_id": doc_id})

    if len(all_clauses) < 2:
        return []

    texts = [c.get("text", "") for c in all_clauses]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Compute cross-doc similarities
    pairs = []
    for i in range(len(all_clauses)):
        for j in range(i + 1, len(all_clauses)):
            if all_clauses[i]["doc_id"] == all_clauses[j]["doc_id"]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            pairs.append({
                "doc_a": all_clauses[i]["doc_id"],
                "clause_a": all_clauses[i].get("clause_id", ""),
                "text_a": all_clauses[i].get("text", "")[:300],
                "doc_b": all_clauses[j]["doc_id"],
                "clause_b": all_clauses[j].get("clause_id", ""),
                "text_b": all_clauses[j].get("text", "")[:300],
                "similarity": round(sim, 4),
                "match_confirmed": "FALSE",
            })

    # Return top-k by similarity
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_k]


def save_cross_doc_pairs(pairs: list[dict], output_dir: Path) -> Path:
    """Save cross-document pairs to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cross_doc_pairs_prefilled.csv"

    fieldnames = [
        "doc_a", "clause_a", "text_a",
        "doc_b", "clause_b", "text_b",
        "similarity", "match_confirmed",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    print(f"  Saved {len(pairs)} cross-doc pairs to {out_path.name}")
    return out_path


def run_auto_annotation(processed_docs: list[dict]) -> dict:
    """Run the full auto-annotation pipeline."""
    print("\n=== Auto Annotation (Claude + Sentence-Transformers) ===")

    # Collect all clauses across documents
    all_clauses = []
    clauses_by_doc = {}

    for doc in processed_docs:
        doc_id = doc.get("doc_id", "unknown")
        clauses = doc.get("clauses", [])
        for c in clauses:
            c["doc_id"] = doc_id
        all_clauses.extend(clauses)
        clauses_by_doc[doc_id] = clauses

    if not all_clauses:
        print("  [WARN] No clauses found for annotation")
        return {"gold_standard": [], "cross_doc_pairs": []}

    print(f"  Total clauses to annotate: {len(all_clauses)}")

    # Step 1: Annotate with Claude
    print("\n  --- Step 1: Claude Annotation ---")
    annotated = annotate_clauses_with_claude(all_clauses)
    gold_path = save_gold_standard(annotated, ANNOTATIONS_DIR)

    # Step 2: Find cross-document pairs
    print("\n  --- Step 2: Cross-Document Pairing ---")
    pairs = find_cross_doc_pairs(clauses_by_doc, top_k=20)
    pairs_path = save_cross_doc_pairs(pairs, ANNOTATIONS_DIR)

    return {
        "gold_standard_path": str(gold_path),
        "gold_standard_count": len(annotated),
        "cross_doc_pairs_path": str(pairs_path),
        "cross_doc_pairs_count": len(pairs),
    }


if __name__ == "__main__":
    # Load processed documents
    docs = []
    for f in sorted(PROCESSED_DIR.glob("*_clauses.json")):
        with open(f) as fh:
            docs.append(json.load(fh))
    run_auto_annotation(docs)
