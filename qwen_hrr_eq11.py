#!/usr/bin/env python3
"""
Compute Qwen composite HRR using paper Eq.11 — the same formula
as recompute_metrics.compute_per_model_hrr (3 perturbation types,
continuous tuple_similarity, split by preserving/altering).
"""

import json, os, random, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.adversarial_tester import PERTURBATION_FUNCTIONS

random.seed(RANDOM_SEED)
DOCS = ["basel3","regbi"]
TUPLE_FIELDS = ["subject","obligation","condition","threshold","deadline","exception"]
ADV_REPARSE_SAMPLE = 20
PERT_TYPES = ["synonym_swap","negation_injection","threshold_mutation"]  # Eq 11 uses these 3

MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

PARSE_PROMPT = """Extract a JSON obligation tuple from this regulatory clause.
Fields: subject, obligation, condition, threshold, deadline, exception (null if absent).
Return ONLY JSON.

Clause: {text}"""

def call(text, retries=3):
    key = os.environ["TOGETHER_API_KEY"]
    payload = {"model": MODEL_ID,
               "messages":[{"role":"user","content":PARSE_PROMPT.format(text=text[:1500])}],
               "max_tokens": 400, "temperature": 0.0,
               "seed": RANDOM_SEED, "response_format":{"type":"json_object"}}
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions", method="POST",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","User-Agent":"Mozilla/5.0"},
            data=json.dumps(payload).encode("utf-8"))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"]["content"]
            s, e = content.find("{"), content.rfind("}")+1
            return json.loads(content[s:e]) if s >= 0 else {f: None for f in TUPLE_FIELDS}
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,504) and attempt < retries-1:
                time.sleep(2**attempt); continue
            return {f: None for f in TUPLE_FIELDS}
        except Exception:
            if attempt < retries-1: time.sleep(1); continue
            return {f: None for f in TUPLE_FIELDS}
    return {f: None for f in TUPLE_FIELDS}


def tuple_similarity(t1, t2):
    matches, total = 0, 0
    for f in TUPLE_FIELDS:
        v1, v2 = t1.get(f), t2.get(f)
        if v1 is None and v2 is None: continue
        total += 1
        if v1 is not None and v2 is not None:
            s1, s2 = str(v1).lower().strip(), str(v2).lower().strip()
            if s1 == s2 or s1 in s2 or s2 in s1: matches += 1
    return matches / max(total, 1)


def main():
    if "TOGETHER_API_KEY" not in os.environ:
        print("TOGETHER_API_KEY not set"); sys.exit(1)

    # Same sample as qwen_hrr.py — match the seed/state
    all_tuples = []
    for d in DOCS:
        with open(PROCESSED_DIR/f"{d}_tuples_qwen.json", encoding="utf-8") as f:
            all_tuples.extend(json.load(f)["parsed_tuples"])
    good = [t for t in all_tuples if sum(1 for f in TUPLE_FIELDS if t.get("tuple",{}).get(f) is not None) >= 2]
    if len(good) < 5:
        good = [t for t in all_tuples if any(t.get("tuple",{}).get(f) is not None for f in TUPLE_FIELDS)]
    sample = random.sample(good, min(ADV_REPARSE_SAMPLE, len(good)))

    print(f"Sampled {len(sample)} clauses")
    print(f"Parsing {len(sample)} originals via Qwen...")
    originals = [call(s.get("text","")) for s in sample]

    pert_jobs = []
    for idx, c in enumerate(sample):
        text = c.get("text","")
        for ptype in PERT_TYPES:
            r = PERTURBATION_FUNCTIONS[ptype](text)
            if r.get("changes"):
                pert_jobs.append((idx, ptype, r["text"], r["semantic_preserving"]))

    print(f"Parsing {len(pert_jobs)} perturbations via Qwen...")
    perts = [call(t) for _,_,t,_ in pert_jobs]

    preserve_sims = []
    alter_sims = []
    for i, (orig_idx, ptype, _, is_preserve) in enumerate(pert_jobs):
        sim = tuple_similarity(originals[orig_idx], perts[i])
        if is_preserve:
            preserve_sims.append(sim)
        else:
            alter_sims.append(sim)

    mu_preserve = float(np.mean(preserve_sims)) if preserve_sims else 0.5
    mu_alter = float(np.mean(alter_sims)) if alter_sims else 0.5
    composite = round(0.5*mu_preserve + 0.5*(1.0-mu_alter), 4)

    print()
    print(f"  preserve sample n={len(preserve_sims)}, mu_preserve={mu_preserve:.4f}")
    print(f"  altering sample n={len(alter_sims)}, mu_alter={mu_alter:.4f}")
    print(f"  Eq.11 composite HRR = {composite:.4f}")

    out = {"HRR_eq11": composite, "stability": round(mu_preserve,4),
           "detection": round(1.0-mu_alter, 4),
           "n_preserve": len(preserve_sims), "n_alter": len(alter_sims)}
    out_path = OUTPUTS_DIR/"reports"/"qwen_hrr_eq11.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
