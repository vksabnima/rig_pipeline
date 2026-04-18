#!/usr/bin/env python3
"""
RIG Pipeline — Main Entry Point
================================
Regulatory Intent Graph: end-to-end pipeline for extracting, graphing,
aligning, and evaluating obligation structures from financial regulations.

Usage:
    python run_pipeline.py                    # Full run (download + annotate + pipeline)
    python run_pipeline.py --skip-annotation  # Skip annotation (reuse cached)
    python run_pipeline.py --skip-download    # Skip PDF download (use existing)
    python run_pipeline.py --dry-run          # Show what would run without executing
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_AVAILABLE,
    RAW_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, GRAPHS_DIR, OUTPUTS_DIR,
)


# ── ANSI Colors ──────────────────────────────────────────────────────────────

class C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CHECK = f"{GREEN}\u2714{RESET}"
    CROSS = f"{RED}\u2718{RESET}"
    WARN = f"{YELLOW}\u26A0{RESET}"


def print_banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ██╗ ██████╗     RIG Pipeline                      ║
║   ██╔══██╗██║██╔════╝     Regulatory Intent Graph            ║
║   ██████╔╝██║██║  ███╗    v1.0.0                             ║
║   ██╔══██╗██║██║   ██║                                       ║
║   ██║  ██║██║╚██████╔╝    Financial Regulation Analysis      ║
║   ╚═╝  ╚═╝╚═╝ ╚═════╝                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def check_env():
    """Check environment and print status."""
    print(f"{C.BOLD}Environment Check:{C.RESET}")
    issues = []

    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-openai-key-here":
        print(f"  {C.CHECK} OpenAI API key configured")
    else:
        print(f"  {C.CROSS} OpenAI API key not set")
        issues.append("OPENAI_API_KEY")

    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "sk-ant-your-anthropic-key-here":
        print(f"  {C.CHECK} Anthropic API key configured")
    else:
        print(f"  {C.CROSS} Anthropic API key not set")
        issues.append("ANTHROPIC_API_KEY")

    if OLLAMA_AVAILABLE:
        print(f"  {C.CHECK} Ollama/Llama3 available")
    else:
        print(f"  {C.DIM}  - Ollama/Llama3 not configured (optional){C.RESET}")

    if issues:
        print(f"\n  {C.WARN}  Missing keys: {', '.join(issues)}")
        print(f"     Pipeline will run with reduced functionality.")
    print()
    return issues


# ── Pipeline Stages ──────────────────────────────────────────────────────────

def stage_download():
    """Stage 1: Download regulatory PDFs."""
    from src.pdf_downloader import download_all_pdfs
    return download_all_pdfs()


def stage_extract():
    """Stage 2: Extract and segment PDFs."""
    from src.pdf_extractor import process_all_pdfs
    return process_all_pdfs()


def stage_annotate(processed_docs):
    """Stage 3: Auto-annotate obligations with Claude."""
    from src.auto_annotator import run_auto_annotation
    return run_auto_annotation(processed_docs)


def stage_parse(processed_docs):
    """Stage 4: Parse obligation tuples."""
    from src.obligation_parser import parse_document
    print("\n=== Obligation Tuple Parsing ===")
    parsed_docs = []
    for doc in processed_docs:
        parsed = parse_document(doc)
        parsed_docs.append(parsed)
    return parsed_docs


def stage_graph(parsed_docs):
    """Stage 5: Build knowledge graphs."""
    from src.graph_builder import build_and_save
    print("\n=== Knowledge Graph Construction ===")
    graph_stats = []
    for doc in parsed_docs:
        stats = build_and_save(doc)
        graph_stats.append(stats)
    return graph_stats


def stage_align():
    """Stage 6: Cross-document graph alignment."""
    from src.graph_aligner import run_alignment
    return run_alignment()


def stage_adversarial(parsed_docs):
    """Stage 7: Adversarial robustness testing."""
    from src.adversarial_tester import run_adversarial_tests
    return run_adversarial_tests(parsed_docs)


def stage_metrics(processed_docs, parsed_docs, alignment_result, adversarial_result):
    """Stage 8: Compute all RIG metrics."""
    from src.metrics import compute_all_metrics

    # Collect data for metrics
    all_extracted = []
    all_parsed = []
    all_original = []

    for doc in processed_docs:
        all_original.extend(doc.get("clauses", []))

    for doc in parsed_docs:
        tuples = doc.get("parsed_tuples", [])
        all_extracted.extend(tuples)
        all_parsed.extend(tuples)

    total_obligation_sentences = len(all_original)
    alignments = alignment_result.get("alignments", [])

    # Estimate total cross-doc pairs
    doc_clause_counts = [len(d.get("parsed_tuples", [])) for d in parsed_docs]
    total_cross_doc_pairs = 0
    for i in range(len(doc_clause_counts)):
        for j in range(i + 1, len(doc_clause_counts)):
            total_cross_doc_pairs += doc_clause_counts[i] * doc_clause_counts[j]

    # Load adversarial samples if available
    adv_samples_path = OUTPUTS_DIR / "reports" / "adversarial_samples.json"
    adversarial_samples = []
    if adv_samples_path.exists():
        with open(adv_samples_path) as f:
            adversarial_samples = json.load(f)

    # Load gold standard if available
    gold_path = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
    gold_standard = None
    if gold_path.exists():
        import csv
        with open(gold_path, encoding="utf-8") as f:
            gold_standard = list(csv.DictReader(f))

    return compute_all_metrics(
        extracted_clauses=all_extracted,
        parsed_tuples=all_parsed,
        original_clauses=all_original,
        alignments=alignments,
        total_cross_doc_pairs=total_cross_doc_pairs,
        adversarial_results=adversarial_result,
        adversarial_samples=adversarial_samples,
        total_obligation_sentences=total_obligation_sentences,
        gold_standard=gold_standard,
    )


# ── Final Checklist ──────────────────────────────────────────────────────────

def print_final_checklist(results: dict):
    """Print a final setup/run checklist with status indicators."""
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                   RIG Pipeline — Summary                     ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")

    steps = [
        ("PDF Download",        results.get("download")),
        ("Text Extraction",     results.get("extract")),
        ("Auto Annotation",     results.get("annotate")),
        ("Obligation Parsing",  results.get("parse")),
        ("Graph Construction",  results.get("graph")),
        ("Cross-Doc Alignment", results.get("align")),
        ("Adversarial Testing", results.get("adversarial")),
        ("Metric Computation",  results.get("metrics")),
    ]

    for name, result in steps:
        if result is None:
            print(f"  {C.DIM}  - {name}: skipped{C.RESET}")
        elif result:
            print(f"  {C.CHECK} {name}: completed")
        else:
            print(f"  {C.CROSS} {name}: failed")

    # Print metrics summary if available
    metrics = results.get("metrics")
    if metrics and isinstance(metrics, dict):
        print(f"\n{C.BOLD}  Metrics:{C.RESET}")
        for name, data in metrics.items():
            if isinstance(data, dict) and "value" in data:
                val = data["value"]
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"    {name}: {bar} {val:.4f}")

    # Key output files
    print(f"\n{C.BOLD}  Output Files:{C.RESET}")
    key_files = [
        ANNOTATIONS_DIR / "gold_standard_prefilled.csv",
        ANNOTATIONS_DIR / "cross_doc_pairs_prefilled.csv",
        OUTPUTS_DIR / "reports" / "rig_metrics.json",
        OUTPUTS_DIR / "reports" / "adversarial_results.json",
        OUTPUTS_DIR / "reports" / "cross_doc_alignments.json",
    ]
    for f in key_files:
        if f.exists():
            print(f"    {C.CHECK} {f.relative_to(Path(__file__).parent)}")
        else:
            print(f"    {C.DIM}  - {f.relative_to(Path(__file__).parent)}{C.RESET}")

    # Manual step highlight
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print(f"""
{C.YELLOW}{C.BOLD}┌──────────────────────────────────────────────────────────────┐
│  ACTION REQUIRED: Configure API keys                         │
│                                                              │
│  cp .env.template .env                                       │
│  # Then paste your API keys in .env                          │
│                                                              │
│  Required: OPENAI_API_KEY and/or ANTHROPIC_API_KEY           │
│  Optional: OLLAMA_AVAILABLE=true (if Ollama is installed)    │
└──────────────────────────────────────────────────────────────┘{C.RESET}
""")
    else:
        print(f"\n  {C.CHECK} .env file configured\n")

    print(f"{C.DIM}  Pipeline complete. Elapsed: {results.get('elapsed', '?')}s{C.RESET}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RIG Pipeline — Regulatory Intent Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-annotation", action="store_true",
                        help="Skip auto-annotation (reuse cached annotations)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PDF download (use existing files)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pipeline stages without executing")
    args = parser.parse_args()

    print_banner()
    start = time.time()

    if args.dry_run:
        print(f"{C.BOLD}Dry run — pipeline stages:{C.RESET}")
        stages = [
            "1. Download regulatory PDFs",
            "2. Extract & segment text",
            "3. Auto-annotate obligations (Claude)",
            "4. Parse obligation tuples (GPT-4o)",
            "5. Build knowledge graphs",
            "6. Cross-document alignment",
            "7. Adversarial robustness testing",
            "8. Compute RIG metrics (RCC, OAL, RIF, CAS, HRR)",
        ]
        for s in stages:
            print(f"  {C.CYAN}>{C.RESET} {s}")
        print(f"\nRun without --dry-run to execute.")
        return

    env_issues = check_env()
    results = {}

    # Stage 1: Download
    if not args.skip_download:
        results["download"] = stage_download()
    else:
        print(f"\n{C.DIM}[SKIP] PDF download (--skip-download){C.RESET}")
        results["download"] = True

    # Stage 2: Extract
    processed_docs = stage_extract()
    results["extract"] = bool(processed_docs)

    # Stage 3: Annotate
    cached_gold = ANNOTATIONS_DIR / "gold_standard_prefilled.csv"
    if not args.skip_annotation:
        annotation_result = stage_annotate(processed_docs)
        results["annotate"] = bool(annotation_result)
    elif cached_gold.exists():
        print(f"\n{C.DIM}[SKIP] Auto-annotation (--skip-annotation, using cached){C.RESET}")
        results["annotate"] = True
    else:
        print(f"\n{C.WARN} No cached annotations found — running annotation anyway")
        annotation_result = stage_annotate(processed_docs)
        results["annotate"] = bool(annotation_result)

    # Stage 4: Parse
    parsed_docs = stage_parse(processed_docs)
    results["parse"] = bool(parsed_docs)

    # Stage 5: Graph
    graph_stats = stage_graph(parsed_docs)
    results["graph"] = bool(graph_stats)

    # Stage 6: Align
    alignment_result = stage_align()
    results["align"] = bool(alignment_result)

    # Stage 7: Adversarial
    adversarial_result = stage_adversarial(parsed_docs)
    results["adversarial"] = bool(adversarial_result)

    # Stage 8: Metrics
    metrics = stage_metrics(processed_docs, parsed_docs, alignment_result, adversarial_result)
    results["metrics"] = metrics

    elapsed = round(time.time() - start, 1)
    results["elapsed"] = elapsed

    print_final_checklist(results)


if __name__ == "__main__":
    main()
