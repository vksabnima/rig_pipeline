#!/usr/bin/env python3
"""
HRR-Detection and HRR-Stability for Qwen.
Mirrors recompute_hrr.py logic; uses Qwen via Together.ai.
Requires Qwen tuples (basel3, regbi) to already exist.
~100 API calls, < $0.05.
"""

import json, os, random, re, sys, time, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.adversarial_tester import PERTURBATION_FUNCTIONS

random.seed(RANDOM_SEED)
DOCS = ["basel3", "regbi"]
TUPLE_FIELDS = ["subject","obligation","condition","threshold","deadline","exception"]
STABILITY_SAMPLE = 20
PERT_TYPES = ["synonym_swap","negation_injection","clause_reorder","threshold_mutation","entity_substitution"]

MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

# Same baits as recompute_hrr.py (must be identical for paper consistency)
HALLUCINATION_BAITS = [
    "The Basel Committee on Banking Supervision was established in 1974 by the central bank governors of the Group of Ten countries.",
    "This section provides an overview of the regulatory framework and its historical development since the 2008 financial crisis.",
    "Market participants have expressed a range of views on the proposed amendments during the public consultation period.",
    "The standardised approach calculates risk-weighted assets by assigning exposures to predefined categories based on external credit ratings.",
    "Several commenters noted that the existing framework already addresses many of the concerns raised in the proposal.",
    "In the view of the Commission, the current regulatory environment is conducive to maintaining financial stability across jurisdictions.",
    "Industry stakeholders have generally supported the direction of reform, while noting implementation challenges.",
    "The impact assessment suggests that smaller institutions may face proportionally higher compliance costs.",
    "The Committee encourages national supervisors to share best practices in the implementation of these standards.",
    "Firms are invited to consider whether additional voluntary measures might further strengthen their risk management frameworks.",
    "The quarterly report on economic trends was published in March 2024 and is available on the central bank website.",
    "Annual conference attendance has grown by 15% over the past three years, reflecting increased interest in regulatory technology.",
    "Previous versions of this regulation had required annual reporting, but that provision was removed in the 2019 amendment.",
    "Some jurisdictions may choose to implement stricter requirements, although this is not mandated by the international standard.",
    "The working group discussed whether institutions should be required to submit quarterly reports, but no decision was reached.",
]

DETECT_PROMPT = """Analyze this text from a financial regulation. Determine if it contains a genuine regulatory obligation.

Return ONLY a JSON object with these fields:
- obligation_present: true if the text contains a binding regulatory obligation (must/shall/required), false otherwise
- confidence: float 0.0-1.0 indicating how confident you are
- subject: the regulated entity (null if no obligation)
- obligation: the required action (null if no obligation)
- reasoning: one sentence explaining your judgment

Text: {text}"""

PARSE_PROMPT = """Extract a JSON obligation tuple from this regulatory clause.
Fields: subject, obligation, condition, threshold, deadline, exception (null if absent).
Return ONLY JSON.

Clause: {text}"""


def call(text_payload, max_tokens=400, retries=3):
    key = os.environ["TOGETHER_API_KEY"]
    payload = {"model": MODEL_ID,
               "messages":[{"role":"user","content":text_payload}],
               "max_tokens": max_tokens, "temperature": 0.0,
               "seed": RANDOM_SEED, "response_format":{"type":"json_object"}}
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions", method="POST",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","User-Agent":"Mozilla/5.0 rig-hrr"},
            data=json.dumps(payload).encode("utf-8"))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"]["content"]
            s, e = content.find("{"), content.rfind("}")+1
            return json.loads(content[s:e]) if s >= 0 else {}
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,504) and attempt < retries-1:
                time.sleep(2**attempt); continue
            print(f"      HTTP {e.code}: {e.read().decode()[:120]}")
            return {}
        except Exception as ex:
            if attempt < retries-1: time.sleep(1); continue
            return {}
    return {}


def hrr_detection():
    print("HRR-Detection (15 baits)...")
    correct = 0
    details = []
    for i, bait in enumerate(HALLUCINATION_BAITS):
        r = call(DETECT_PROMPT.format(text=bait))
        op = r.get("obligation_present", True)
        if isinstance(op, str): op = op.lower() not in ("false","no","0")
        try: conf = float(r.get("confidence", 1.0))
        except: conf = 1.0
        is_correct = (not op) or (conf < 0.4)
        if is_correct: correct += 1
        details.append({"bait_idx":i,"obligation_present":op,"confidence":conf,"correct_rejection":is_correct,"bait_snippet":bait[:80]})
        print(f"  [{i+1}/15] op={op} conf={conf} correct={is_correct}")
    rate = correct / len(HALLUCINATION_BAITS)
    print(f"  HRR_det = {rate:.4f} ({correct}/15)")
    return {"hrr_detection": round(rate,4), "correct_rejections":correct,
            "total_baits": len(HALLUCINATION_BAITS), "details": details}


def tuples_match(t1, t2):
    for f in TUPLE_FIELDS:
        v1, v2 = t1.get(f), t2.get(f)
        if v1 is None and v2 is None: continue
        if v1 is None or v2 is None: return False
        if str(v1).lower().strip() != str(v2).lower().strip(): return False
    return True


def hrr_stability():
    print("HRR-Stability (20 clauses x 5 perturbation types)...")
    # Load Qwen tuples for sampling
    all_tuples = []
    for d in DOCS:
        with open(PROCESSED_DIR/f"{d}_tuples_qwen.json", encoding="utf-8") as f:
            all_tuples.extend(json.load(f)["parsed_tuples"])
    good = [t for t in all_tuples if sum(1 for f in TUPLE_FIELDS if t.get("tuple",{}).get(f) is not None) >= 2]
    if len(good) < 5:
        good = [t for t in all_tuples if any(t.get("tuple",{}).get(f) is not None for f in TUPLE_FIELDS)]
    sample = random.sample(good, min(STABILITY_SAMPLE, len(good)))

    # baseline parse
    print(f"  baseline parse {len(sample)} originals...")
    baseline = [call(PARSE_PROMPT.format(text=s.get("text","")[:1500])) for s in sample]

    # build perturbations
    pert_jobs = []
    for idx, c in enumerate(sample):
        text = c.get("text","")
        for ptype in PERT_TYPES:
            r = PERTURBATION_FUNCTIONS[ptype](text)
            if r.get("changes"): pert_jobs.append((idx, ptype, r["text"]))
    print(f"  parse {len(pert_jobs)} perturbed...")
    perts = [call(PARSE_PROMPT.format(text=t[:1500])) for _,_,t in pert_jobs]

    stable = 0
    for i, (orig_idx, _, _) in enumerate(pert_jobs):
        if i >= len(perts): break
        if tuples_match(baseline[orig_idx], perts[i]): stable += 1
    rate = stable / max(len(pert_jobs), 1)
    print(f"  HRR_stab = {rate:.4f} ({stable}/{len(pert_jobs)})")
    return {"hrr_stability": round(rate,4), "stable":stable, "total_tests":len(pert_jobs)}


def main():
    if "TOGETHER_API_KEY" not in os.environ:
        print("TOGETHER_API_KEY not set"); sys.exit(1)
    det = hrr_detection()
    stab = hrr_stability()

    out = {"HRR_detection": det["hrr_detection"], "HRR_stability": stab["hrr_stability"],
           "detection_details": det, "stability_details": stab}
    out_path = OUTPUTS_DIR / "reports" / "qwen_hrr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"\nQwen HRR_det = {det['hrr_detection']:.4f}, HRR_stab = {stab['hrr_stability']:.4f}")
    print(f"Composite (mean) = {(det['hrr_detection']+stab['hrr_stability'])/2:.4f}  (note: paper composite uses different formula)")


if __name__ == "__main__":
    main()
