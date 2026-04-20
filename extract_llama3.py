#!/usr/bin/env python3
"""
Extract obligation tuples using Llama-3-8B-Instruct-Lite via Together.ai.
Replaces the original Llama 3.2 3B / CPU-Ollama setup with a
properly-served 8B open-weights model on serverless infrastructure —
removing the 'under-configured small model' critique from the paper.

Run:   TOGETHER_API_KEY=... python extract_llama3.py [--doc basel3|regbi|gdpr|all]
Out:   data/processed/{doc}_tuples_llama3_8b.json
       data/graphs/llama3_8b/{doc}_graph.graphml
"""

import argparse, json, os, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROCESSED_DIR, GRAPHS_DIR, RANDOM_SEED

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
MODEL_NAME = "llama3_8b"
PRICE_PER_M = 0.10   # approx; Lite tier is cheaper than base
TUPLE_FIELDS = ["subject", "obligation", "condition", "threshold", "deadline", "exception"]

SYSTEM_PROMPT = """You are a regulatory compliance expert. Extract structured obligation tuples from regulatory text.

For each obligation clause, return a JSON object with these fields:
- subject: The regulated entity or actor
- obligation: The required action or prohibition
- condition: Triggering condition or scope qualifier
- threshold: Quantitative limit or numeric requirement
- deadline: Time constraint or compliance date
- exception: Exemptions or carve-outs

If a field is not present in the text, set it to null.
Return ONLY one valid JSON object -- no markdown, no explanation."""


def call(text, retries=3):
    key = os.environ["TOGETHER_API_KEY"]
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\nExtract the obligation tuple from:\n\n{text}"}],
        "max_tokens": 400, "temperature": 0.0, "seed": RANDOM_SEED,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions", method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            # Parse JSON — handle multi-object edge case
            s, e = content.find("{"), content.find("}", content.find("{")) + 1
            if s >= 0 and e > s:
                try:
                    tup = json.loads(content[s:e])
                    tup = {f: tup.get(f) if tup.get(f) not in ("", "null", "None") else None for f in TUPLE_FIELDS}
                    return tup, usage
                except json.JSONDecodeError:
                    pass
            return {f: None for f in TUPLE_FIELDS}, usage
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:150]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt); continue
            raise RuntimeError(f"HTTP {e.code}: {err}")
        except Exception:
            if attempt < retries - 1: time.sleep(1); continue
            return {f: None for f in TUPLE_FIELDS}, {}
    return {f: None for f in TUPLE_FIELDS}, {}


def _worker(idx_clause):
    idx, c = idx_clause
    tup, usage = call(c.get("text", "")[:1500])
    return idx, tup, usage


def extract_doc(doc_id, workers=8):
    cache_path = PROCESSED_DIR / f"{doc_id}_tuples_{MODEL_NAME}.json"
    if cache_path.exists():
        print(f"  [CACHED] {doc_id}/{MODEL_NAME}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f).get("parsed_tuples", []), {"cached": True}

    with open(PROCESSED_DIR / f"{doc_id}_clauses.json", encoding="utf-8") as f:
        clauses = json.load(f).get("clauses", [])
    print(f"  Extracting {len(clauses)} clauses for {doc_id} with {workers} workers...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [None] * len(clauses)
    total_in = total_out = done = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, (i, c)): i for i, c in enumerate(clauses)}
        for fut in as_completed(futures):
            try:
                idx, tup, usage = fut.result()
                results[idx] = (tup, usage)
                total_in += usage.get("prompt_tokens", 0)
                total_out += usage.get("completion_tokens", 0)
            except Exception as e:
                idx = futures[fut]
                print(f"    [ERROR] clause {idx}: {e}")
                results[idx] = ({f: None for f in TUPLE_FIELDS}, {})
            done += 1
            if done % 50 == 0 or done == len(clauses):
                elapsed = time.time() - start
                rate = done / max(elapsed, 1)
                eta = (len(clauses) - done) / max(rate, 0.001)
                print(f"    [{done}/{len(clauses)}] ({rate:.1f}/s, ETA {eta:.0f}s, tok={total_in}/{total_out})")

    parsed = [{**c, "tuple": results[i][0], "model_used": "llama-3-8b-instruct-lite"} for i, c in enumerate(clauses)]
    elapsed = time.time() - start
    print(f"  Done {doc_id} in {elapsed:.0f}s. Tokens: in={total_in}, out={total_out}")
    print(f"  Est cost: ${(total_in + total_out) * PRICE_PER_M / 1_000_000:.4f}")

    out = {"doc_id": doc_id, "model": MODEL_NAME, "model_id": MODEL_ID,
           "total_clauses": len(clauses), "parsed_tuples": parsed,
           "tokens_in": total_in, "tokens_out": total_out}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return parsed, {"in": total_in, "out": total_out, "elapsed": elapsed}


def build_graph(parsed, doc_id):
    import networkx as nx
    G = nx.DiGraph()
    G.graph["doc_id"] = doc_id; G.graph["model"] = MODEL_NAME
    for entry in parsed:
        cid = entry.get("clause_id", "")
        t = entry.get("tuple", {})
        attrs = {"type": "obligation", "text": entry.get("text", ""), "section": entry.get("section", "")}
        if t.get("obligation") is not None:
            attrs["obligation_desc"] = str(t["obligation"])
        G.add_node(cid, **attrs)
        for field, ntype, rel, rev in [("subject","subject","has_obligation",True),
                                        ("condition","condition","triggers",True),
                                        ("threshold","threshold","has_threshold",False),
                                        ("deadline","deadline","has_deadline",False),
                                        ("exception","exception","exempts",True)]:
            v = t.get(field)
            if v:
                nid = f"{ntype.upper()[:4]}:{cid}" if field != "subject" else f"SUBJ:{v}"
                G.add_node(nid, type=ntype, label=str(v))
                if rev: G.add_edge(nid, cid, relation=rel)
                else: G.add_edge(cid, nid, relation=rel)
    subjects = [n for n,d in G.nodes(data=True) if d.get("type") == "subject"]
    for s in subjects:
        obls = [x for x in G.successors(s) if G.nodes[x].get("type") == "obligation"]
        for i in range(len(obls)):
            for j in range(i+1, len(obls)):
                G.add_edge(obls[i], obls[j], relation="co_regulated")
    for _, d in G.nodes(data=True):
        for k in list(d):
            if d[k] is None: d[k] = ""
    for _,_,d in G.edges(data=True):
        for k in list(d):
            if d[k] is None: d[k] = ""
    out_dir = GRAPHS_DIR / MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(out_dir / f"{doc_id}_graph.graphml"))
    return G.number_of_nodes(), G.number_of_edges()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doc", default="all", choices=["basel3", "regbi", "gdpr", "all"])
    args = p.parse_args()
    docs = ["basel3", "regbi", "gdpr"] if args.doc == "all" else [args.doc]
    grand_in = grand_out = 0
    for doc_id in docs:
        print(f"\n=== {doc_id} ===")
        parsed, stats = extract_doc(doc_id)
        if "in" in stats:
            grand_in += stats["in"]; grand_out += stats["out"]
        n, e = build_graph(parsed, doc_id)
        print(f"  Graph: {n} nodes, {e} edges")
    if grand_in or grand_out:
        cost = (grand_in + grand_out) * PRICE_PER_M / 1_000_000
        print(f"\n=== TOTAL: in={grand_in}, out={grand_out} tokens, est cost ${cost:.4f} ===")


if __name__ == "__main__":
    main()
