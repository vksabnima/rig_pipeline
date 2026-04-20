#!/usr/bin/env python3
"""
HRR-Detection + HRR-Stability + Eq.11 composite HRR for Llama-3-8B-Lite.
Mirrors qwen_hrr.py + qwen_hrr_eq11.py for the replacement small open model.
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
PERT_TYPES_ALL = ["synonym_swap","negation_injection","clause_reorder","threshold_mutation","entity_substitution"]
PERT_TYPES_EQ11 = ["synonym_swap","negation_injection","threshold_mutation"]

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

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
Return ONLY one JSON object.

Clause: {text}"""


def call(prompt, max_tokens=400, retries=3):
    key = os.environ["TOGETHER_API_KEY"]
    payload = {"model": MODEL_ID, "messages":[{"role":"user","content":prompt}],
               "max_tokens": max_tokens, "temperature":0.0, "seed": RANDOM_SEED,
               "response_format":{"type":"json_object"}}
    for attempt in range(retries):
        req = urllib.request.Request("https://api.together.xyz/v1/chat/completions", method="POST",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","User-Agent":"Mozilla/5.0"},
            data=json.dumps(payload).encode("utf-8"))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"]["content"]
            s,e = content.find("{"), content.find("}", content.find("{"))+1
            if s >= 0 and e > s:
                try: return json.loads(content[s:e])
                except json.JSONDecodeError: return {}
            return {}
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,504) and attempt < retries-1:
                time.sleep(2**attempt); continue
            return {}
        except Exception:
            if attempt < retries-1: time.sleep(1); continue
            return {}
    return {}


def tuples_match(t1, t2):
    for f in TUPLE_FIELDS:
        v1,v2 = t1.get(f), t2.get(f)
        if v1 is None and v2 is None: continue
        if v1 is None or v2 is None: return False
        if str(v1).lower().strip() != str(v2).lower().strip(): return False
    return True


def tuple_similarity(t1, t2):
    matches, total = 0, 0
    for f in TUPLE_FIELDS:
        v1,v2 = t1.get(f), t2.get(f)
        if v1 is None and v2 is None: continue
        total += 1
        if v1 is not None and v2 is not None:
            s1,s2 = str(v1).lower().strip(), str(v2).lower().strip()
            if s1 == s2 or s1 in s2 or s2 in s1: matches += 1
    return matches / max(total, 1)


def hrr_detection():
    print("HRR-Detection (15 baits)...")
    correct = 0; details = []
    for i, bait in enumerate(HALLUCINATION_BAITS):
        r = call(DETECT_PROMPT.format(text=bait))
        op = r.get("obligation_present", True)
        if isinstance(op, str): op = op.lower() not in ("false","no","0")
        try: conf = float(r.get("confidence", 1.0))
        except: conf = 1.0
        is_correct = (not op) or (conf < 0.4)
        if is_correct: correct += 1
        details.append({"bait_idx":i,"op":op,"conf":conf,"correct":is_correct})
        print(f"  [{i+1}/15] op={op} conf={conf} correct={is_correct}")
    rate = correct/15
    print(f"  HRR_det = {rate:.4f} ({correct}/15)")
    return {"hrr_detection": round(rate,4), "correct_rejections":correct,
            "total_baits":15, "details":details}


def hrr_stability():
    print("HRR-Stability (5 perturbation types)...")
    all_t = []
    for d in DOCS:
        with open(PROCESSED_DIR/f"{d}_tuples_llama3_8b.json", encoding="utf-8") as f:
            all_t.extend(json.load(f)["parsed_tuples"])
    good = [t for t in all_t if sum(1 for f in TUPLE_FIELDS if t.get("tuple",{}).get(f) is not None) >= 2]
    if len(good) < 5:
        good = [t for t in all_t if any(t.get("tuple",{}).get(f) is not None for f in TUPLE_FIELDS)]
    sample = random.sample(good, min(STABILITY_SAMPLE, len(good)))

    baselines = [call(PARSE_PROMPT.format(text=s.get("text","")[:1500])) for s in sample]

    jobs = []
    for idx, c in enumerate(sample):
        text = c.get("text","")
        for pt in PERT_TYPES_ALL:
            r = PERTURBATION_FUNCTIONS[pt](text)
            if r.get("changes"): jobs.append((idx, pt, r["text"]))
    print(f"  Re-parsing {len(sample)} baselines + {len(jobs)} perturbations")
    perts = [call(PARSE_PROMPT.format(text=t[:1500])) for _,_,t in jobs]

    stable = sum(1 for i,(oi,_,_) in enumerate(jobs)
                 if i < len(perts) and tuples_match(baselines[oi], perts[i]))
    rate = stable / max(len(jobs), 1)
    print(f"  HRR_stab = {rate:.4f} ({stable}/{len(jobs)})")
    return {"hrr_stability": round(rate,4), "stable":stable, "total_tests":len(jobs)}


def hrr_eq11():
    print("HRR Eq.11 composite (3 perturbation types, continuous sim)...")
    rng = random.Random(RANDOM_SEED)
    all_t = []
    for d in DOCS:
        with open(PROCESSED_DIR/f"{d}_tuples_llama3_8b.json", encoding="utf-8") as f:
            all_t.extend(json.load(f)["parsed_tuples"])
    good = [t for t in all_t if sum(1 for f in TUPLE_FIELDS if t.get("tuple",{}).get(f) is not None) >= 2]
    if len(good) < 5:
        good = [t for t in all_t if any(t.get("tuple",{}).get(f) is not None for f in TUPLE_FIELDS)]
    sample = rng.sample(good, min(STABILITY_SAMPLE, len(good)))

    origs = [call(PARSE_PROMPT.format(text=s.get("text","")[:1500])) for s in sample]
    jobs = []
    for idx, c in enumerate(sample):
        text = c.get("text","")
        for pt in PERT_TYPES_EQ11:
            r = PERTURBATION_FUNCTIONS[pt](text)
            if r.get("changes"): jobs.append((idx, pt, r["text"], r["semantic_preserving"]))
    print(f"  Parsing {len(jobs)} perturbations")
    perts = [call(PARSE_PROMPT.format(text=t[:1500])) for _,_,t,_ in jobs]

    pres, alt = [], []
    for i,(oi,_,_,is_pres) in enumerate(jobs):
        if i >= len(perts): break
        sim = tuple_similarity(origs[oi], perts[i])
        (pres if is_pres else alt).append(sim)
    mu_p = float(np.mean(pres)) if pres else 0.5
    mu_a = float(np.mean(alt)) if alt else 0.5
    comp = round(0.5*mu_p + 0.5*(1.0-mu_a), 4)
    print(f"  preserve n={len(pres)}, mu={mu_p:.4f}; alter n={len(alt)}, mu={mu_a:.4f}")
    print(f"  HRR composite (Eq.11) = {comp:.4f}")
    return {"HRR_eq11": comp, "stability": round(mu_p,4), "detection": round(1.0-mu_a,4),
            "n_preserve":len(pres), "n_alter":len(alt)}


def main():
    if "TOGETHER_API_KEY" not in os.environ: sys.exit("Set TOGETHER_API_KEY")
    det = hrr_detection()
    stab = hrr_stability()
    eq11 = hrr_eq11()
    out = {"HRR_detection": det["hrr_detection"],
           "HRR_stability": stab["hrr_stability"],
           "HRR_eq11": eq11["HRR_eq11"],
           "detection_details": det,
           "stability_details": stab,
           "eq11_details": eq11}
    out_p = OUTPUTS_DIR/"reports"/"llama3_8b_hrr.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_p}")
    print(f"\nLlama3-8B  HRR_det={det['hrr_detection']:.4f}, HRR_stab={stab['hrr_stability']:.4f}, HRR_eq11={eq11['HRR_eq11']:.4f}")


if __name__ == "__main__":
    main()
