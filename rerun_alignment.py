#!/usr/bin/env python3
"""
Targeted rerun: skip cached stages, run alignment + adversarial + metrics.
Uses cached extraction/parsing/graphs for both Basel III and Reg BI.
"""

import csv
import io
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    RAW_DIR, PROCESSED_DIR, GRAPHS_DIR, ANNOTATIONS_DIR, OUTPUTS_DIR,
)
from src.pdf_extractor import process_pdf
from src.obligation_parser import parse_document
from src.graph_builder import build_and_save
from src.graph_aligner import run_alignment
from src.adversarial_tester import run_adversarial_tests
from src.metrics import compute_all_metrics

GRN = "\033[92m"
CYN = "\033[96m"
BLD = "\033[1m"
DIM = "\033[2m"
RST = "\033[0m"
CHECK = f"{GRN}\u2714{RST}"

start = time.time()

print(f"\n{CYN}{BLD}=== RIG Pipeline -- Targeted Rerun ==={RST}")

# -- Load or build Reg BI data ------------------------------------------------
regbi_clauses_path = PROCESSED_DIR / "regbi_clauses.json"
regbi_tuples_path = PROCESSED_DIR / "regbi_tuples.json"
regbi_graph_path = GRAPHS_DIR / "regbi_graph.graphml"

if regbi_clauses_path.exists() and regbi_tuples_path.exists() and regbi_graph_path.exists():
    print(f"{DIM}  [CACHED] Reg BI extraction/parsing/graph{RST}")
    with open(regbi_clauses_path, encoding="utf-8") as f:
        regbi_extracted = json.load(f)
    with open(regbi_tuples_path, encoding="utf-8") as f:
        regbi_parsed = json.load(f)
    print(f"  {CHECK} Reg BI: {regbi_extracted.get('total_pages', '?')} pages, "
          f"{len(regbi_extracted.get('clauses', []))} clauses, "
          f"{regbi_parsed.get('total_clauses', '?')} tuples\n")
else:
    print(f"{BLD}[1/3] Processing Reg BI (extract + parse + graph)...{RST}")
    regbi_extracted = process_pdf("regbi", RAW_DIR / "regbi.pdf")
    regbi_parsed = parse_document(regbi_extracted)
    build_and_save(regbi_parsed)
    print(f"  {CHECK} Reg BI processed\n")

# -- Load cached Basel III data ------------------------------------------------
print(f"{DIM}  [CACHED] Basel III extraction/parsing/graph{RST}")
with open(PROCESSED_DIR / "basel3_clauses.json", encoding="utf-8") as f:
    basel3_extracted = json.load(f)
with open(PROCESSED_DIR / "basel3_tuples.json", encoding="utf-8") as f:
    basel3_parsed = json.load(f)
print(f"  {CHECK} Basel III: {basel3_extracted.get('total_pages', '?')} pages, "
      f"{len(basel3_extracted.get('clauses', []))} clauses, "
      f"{len(basel3_parsed.get('parsed_tuples', []))} tuples\n")

# -- Step 1: Cross-document alignment -----------------------------------------
print(f"{BLD}[1/3] Cross-document alignment (Basel III + Reg BI)...{RST}")
alignment_result = run_alignment()
print(f"  {CHECK} {alignment_result.get('num_alignments', 0)} alignments found\n")

# -- Step 2: Adversarial robustness testing ------------------------------------
print(f"{BLD}[2/3] Adversarial robustness testing (both docs)...{RST}")
all_parsed_docs = [basel3_parsed, regbi_parsed]
adversarial_result = run_adversarial_tests(all_parsed_docs)
print(f"  {CHECK} Adversarial tests complete\n")

# -- Step 3: Recompute all metrics ---------------------------------------------
print(f"{BLD}[3/3] Computing all RIG metrics...{RST}")

all_extracted_docs = [basel3_extracted, regbi_extracted]
all_original_clauses = []
all_parsed_tuples = []
all_extracted_tuples = []

for doc in all_extracted_docs:
    all_original_clauses.extend(doc.get("clauses", []))
for doc in all_parsed_docs:
    tuples = doc.get("parsed_tuples", [])
    all_parsed_tuples.extend(tuples)
    all_extracted_tuples.extend(tuples)

total_obligation_sentences = len(all_original_clauses)
alignments = alignment_result.get("alignments", [])

doc_tuple_counts = [len(d.get("parsed_tuples", [])) for d in all_parsed_docs]
total_cross_doc_pairs = 0
for i in range(len(doc_tuple_counts)):
    for j in range(i + 1, len(doc_tuple_counts)):
        total_cross_doc_pairs += doc_tuple_counts[i] * doc_tuple_counts[j]

adv_path = OUTPUTS_DIR / "reports" / "adversarial_samples.json"
adversarial_samples = []
if adv_path.exists():
    with open(adv_path, encoding="utf-8") as f:
        adversarial_samples = json.load(f)

gold_path = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
gold_standard = None
if gold_path.exists():
    with open(gold_path, encoding="utf-8") as f:
        gold_standard = list(csv.DictReader(f))

metrics = compute_all_metrics(
    extracted_clauses=all_extracted_tuples,
    parsed_tuples=all_parsed_tuples,
    original_clauses=all_original_clauses,
    alignments=alignments,
    total_cross_doc_pairs=total_cross_doc_pairs,
    adversarial_results=adversarial_result,
    adversarial_samples=adversarial_samples,
    total_obligation_sentences=total_obligation_sentences,
    gold_standard=gold_standard,
)

elapsed = round(time.time() - start, 1)

# -- Summary -------------------------------------------------------------------
print(f"\n{CYN}{BLD}{'='*60}")
print(f"  RIG Pipeline -- Final Results")
print(f"{'='*60}{RST}\n")

print(f"  {BLD}Documents:{RST}")
print(f"    Basel III: {basel3_extracted.get('total_pages', '?')} pages, "
      f"{len(basel3_extracted.get('clauses', []))} clauses, "
      f"{len(basel3_parsed.get('parsed_tuples', []))} tuples")
print(f"    Reg BI:    {regbi_extracted.get('total_pages', '?')} pages, "
      f"{len(regbi_extracted.get('clauses', []))} clauses, "
      f"{regbi_parsed.get('total_clauses', '?')} tuples")

print(f"\n  {BLD}Knowledge Graphs:{RST}")
print(f"    Merged:    {alignment_result.get('merged_nodes', '?')} nodes, "
      f"{alignment_result.get('merged_edges', '?')} edges")
print(f"    Cross-doc alignments: {alignment_result.get('num_alignments', 0)}")

print(f"\n  {BLD}Metrics:{RST}")
for name, data in metrics.items():
    if isinstance(data, dict) and "value" in data:
        val = data["value"]
        bar = "\u2588" * int(val * 20) + "\u2591" * (20 - int(val * 20))
        print(f"    {name}: {bar} {val:.4f}  ({data.get('full_name', '')})")

print(f"\n  {BLD}Output Files:{RST}")
for fpath in [
    ANNOTATIONS_DIR / "gold_standard_prefilled.csv",
    ANNOTATIONS_DIR / "cross_doc_pairs_prefilled.csv",
    GRAPHS_DIR / "basel3_graph.graphml",
    GRAPHS_DIR / "regbi_graph.graphml",
    GRAPHS_DIR / "merged_graph.graphml",
    OUTPUTS_DIR / "reports" / "rig_metrics.json",
    OUTPUTS_DIR / "reports" / "adversarial_results.json",
    OUTPUTS_DIR / "reports" / "cross_doc_alignments.json",
]:
    if fpath.exists():
        size = fpath.stat().st_size / 1024
        print(f"    {CHECK} {fpath.relative_to(Path(__file__).parent)}  ({size:.0f} KB)")
    else:
        print(f"    {DIM}  - {fpath.relative_to(Path(__file__).parent)}{RST}")

print(f"\n{DIM}  Completed in {elapsed}s{RST}\n")
