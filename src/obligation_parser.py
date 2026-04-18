"""
Obligation Parser
-----------------
Parses obligation clauses into structured 6-tuples:
  (Subject, Obligation, Condition, Threshold, Deadline, Exception)

Uses GPT-4o via OpenAI API, with optional Ollama/Llama 3.2 fallback.
"""

import json
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    OPENAI_API_KEY, OLLAMA_AVAILABLE, MODELS,
    PROCESSED_DIR, RANDOM_SEED,
)

TUPLE_SCHEMA = {
    "subject": "The regulated entity or actor (e.g., 'bank', 'broker-dealer')",
    "obligation": "The required action or prohibition",
    "condition": "Triggering condition or scope qualifier",
    "threshold": "Quantitative limit or numeric requirement",
    "deadline": "Time constraint or compliance date",
    "exception": "Exemptions or carve-outs",
}

SYSTEM_PROMPT = f"""You are a regulatory compliance expert. Extract structured obligation tuples from regulatory text.

For each obligation clause, return a JSON object with these fields:
{json.dumps(TUPLE_SCHEMA, indent=2)}

If a field is not present in the text, set it to null.
Return ONLY valid JSON — no markdown, no explanation."""


def _parse_with_openai(clauses: list[dict]) -> list[dict]:
    """Parse clauses using OpenAI GPT-4o."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    parsed = []

    for clause in clauses:
        try:
            response = client.chat.completions.create(
                model=MODELS["extractor"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract the obligation tuple from:\n\n{clause['text']}"},
                ],
                temperature=0.0,
                seed=RANDOM_SEED,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            tuple_data = json.loads(content)
            parsed.append({
                **clause,
                "tuple": tuple_data,
                "model_used": MODELS["extractor"],
            })
        except Exception as e:
            parsed.append({
                **clause,
                "tuple": {k: None for k in TUPLE_SCHEMA},
                "model_used": "error",
                "error": str(e),
            })

    return parsed


def _parse_with_ollama(clauses: list[dict]) -> list[dict]:
    """Parse clauses using local Llama3 via Ollama."""
    import requests

    parsed = []
    for clause in clauses:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": MODELS["local"],
                    "prompt": f"{SYSTEM_PROMPT}\n\nExtract the obligation tuple from:\n\n{clause['text']}",
                    "stream": False,
                    "options": {"temperature": 0.0, "seed": RANDOM_SEED},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json().get("response", "{}")
            # Try to extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                tuple_data = json.loads(json_match.group())
            else:
                tuple_data = {k: None for k in TUPLE_SCHEMA}
            parsed.append({
                **clause,
                "tuple": tuple_data,
                "model_used": MODELS["local"],
            })
        except Exception as e:
            parsed.append({
                **clause,
                "tuple": {k: None for k in TUPLE_SCHEMA},
                "model_used": "error",
                "error": str(e),
            })

    return parsed


def parse_obligations(clauses: list[dict], use_ollama: bool = False) -> list[dict]:
    """Parse obligation clauses into structured tuples."""
    if not clauses:
        return []

    if use_ollama and OLLAMA_AVAILABLE:
        print(f"  Using Ollama ({MODELS['local']}) for parsing...")
        return _parse_with_ollama(clauses)
    elif OPENAI_API_KEY:
        print(f"  Using OpenAI ({MODELS['extractor']}) for parsing...")
        return _parse_with_openai(clauses)
    else:
        print("  [WARN] No API keys configured — returning empty tuples")
        return [
            {**c, "tuple": {k: None for k in TUPLE_SCHEMA}, "model_used": "none"}
            for c in clauses
        ]


def parse_document(doc_result: dict, use_ollama: bool = False) -> dict:
    """Parse all clauses from a processed document."""
    doc_id = doc_result["doc_id"]
    clauses = doc_result.get("clauses", [])
    print(f"  [PARSE] {doc_id}: {len(clauses)} clauses")

    parsed = parse_obligations(clauses, use_ollama=use_ollama)

    output = {
        "doc_id": doc_id,
        "total_clauses": len(clauses),
        "parsed_tuples": parsed,
    }

    out_path = PROCESSED_DIR / f"{doc_id}_tuples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"    Saved to {out_path.name}")

    return output
