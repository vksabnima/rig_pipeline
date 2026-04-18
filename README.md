# RIG: Regulatory Intent Graph Pipeline

**LLM-Assisted Compliance Extraction from Financial & Data-Protection Regulations**

This repository accompanies the manuscript *"RIG: A Regulatory Intent Graph Methodology for LLM-Assisted Compliance Extraction from Financial Documents"* (under review). It provides the full reference implementation, evaluation harness, adversarial test sets, and validated annotations needed to reproduce every table and figure in the paper.

**Author:** Vikash Kumar — Independent Researcher, Austin, Texas, USA
**Correspondence:** vikash.singh261@gmail.com
**License:** MIT (see [`LICENSE`](LICENSE))

---

## 1. What RIG Does

RIG is a three-stage pipeline that turns raw regulatory PDFs into a queryable **Unified Compliance Graph (UCG)**:

```
PDF → Clause Segmentation → LLM Tuple Extraction → Knowledge Graph
                                                         ↓
                        Cross-Document Alignment → Metric Evaluation
```

Each obligation clause is compressed into a structured six-field tuple:

```
T = ⟨ subject, obligation, condition, threshold, deadline, exception ⟩
```

Applied to **Basel III (161 pp)**, **SEC Regulation Best Interest (770 pp)**, and the **EU GDPR (88 pp)**, the pipeline produces:

| Artifact | Count |
|---|---|
| Segmented obligation clauses | **2,167** |
| UCG nodes (obligations + conditions + exceptions) | **4,603** |
| UCG edges (Triggers / DependsOn / ExceptionOf) | **74,306** |
| Cross-document obligation alignments at τ = 0.50 | **2,390** |

---

## 2. Five Evaluation Metrics

The paper introduces five metrics tailored to regulatory extraction. All are implemented in [`src/metrics.py`](src/metrics.py) and re-computable via [`recompute_metrics.py`](recompute_metrics.py) / [`recompute_hrr.py`](recompute_hrr.py).

| Metric | Definition | Scope |
|---|---|---|
| **RCC** — Regulatory Clause Coverage | Fraction of clauses yielding a non-null tuple | Extraction completeness |
| **OAL** — Obligation Alignment Latency | Fraction of nodes with ≥3 non-null tuple fields | Compliance-chain complexity |
| **RIF** — Regulatory Intent Fidelity | Mean cosine similarity to validated gold standard | Semantic correctness |
| **CAS** — Cross-document Alignment Score | Precision of human-verified shared-obligation pairs | Cross-framework linkage |
| **HRR** — Hallucination Resistance Rate | Decomposed into `HRR-Detection` + `HRR-Stability` | Adversarial robustness |

The HRR decomposition is a key contribution: composite HRR alone can be inflated by **extraction-sparsity robustness**, where a model scores high simply because it emits near-null tuples (see §5.4 of the paper).

---

## 3. Headline Results (reproducible)

Benchmarked across **GPT-4o**, **Claude Sonnet**, and **Llama 3.2** on Basel III + Reg BI + GDPR:

- **Claude Sonnet > GPT-4o on RIF** with non-overlapping 95 % bootstrap CIs on both annotated documents (gap ≈ 0.053, stable across documents).
- **Cross-document alignment is model-agnostic** — inter-model CAS range = **0.011**.
- **GDPR is structurally a superior extraction target** — OAL ≈ 0.75 vs. 0.45 on Basel III / Reg BI.
- **90 %** of the 30 lowest-fidelity extractions fail on **cross-reference dependencies** (e.g. *"as specified in paragraph N"*) — a tractable architectural target, not a general LLM limitation.

Full benchmark tables: see Table 2 of the manuscript; replicate via §5 below.

---

## 4. Repository Layout

```
rig_pipeline/
├── run_pipeline.py          # End-to-end driver (download → extract → graph → align → score)
├── run_multimodel.py        # Multi-model benchmark (GPT-4o / Claude Sonnet / Llama 3.2)
├── gdpr_sonnet.py           # GDPR-specific extraction using Claude Sonnet
├── human_review.py          # Two-stage LLM-assisted annotation validator (Basel III gold)
├── bootstrap_reviewed.py    # Bootstrap 95 % CIs over the validated gold standard
├── recompute_metrics.py     # Recompute RCC / OAL / RIF / CAS from cached extractions
├── recompute_hrr.py         # Recompute HRR and its detection/stability decomposition
├── rerun_alignment.py       # Rerun cross-document alignment at alternate τ values
├── error_analysis.py        # Analyze the N lowest-fidelity extractions
│
├── task1_gdpr.py            # Task 1 — GDPR extraction run
├── task2_threshold.py       # Task 2 — τ sensitivity sweep (Figure 4)
├── task3_bootstrap.py       # Task 3 — Bootstrap CI generation
│
├── src/
│   ├── pdf_downloader.py    # Fetch Basel III / Reg BI / GDPR source PDFs
│   ├── pdf_extractor.py     # pdfplumber + header/footer/footnote stripping
│   ├── obligation_parser.py # Rule-based clause segmenter (Obligation / Condition / Exception / Declarative)
│   ├── auto_annotator.py    # Automated pre-annotation pass (Claude Sonnet)
│   ├── graph_builder.py     # NetworkX DiGraph construction + betweenness centrality
│   ├── graph_aligner.py     # sentence-transformer (all-MiniLM-L6-v2) cross-document alignment
│   ├── metrics.py           # RCC / OAL / RIF / CAS / HRR implementations
│   └── adversarial_tester.py# 250-clause adversarial set + 15 hallucination baits
│
├── config/config.py         # Paths, models, τ threshold, random seed
├── requirements.txt
├── .env.template            # Copy to .env and populate API keys
├── setup_ollama.sh          # Optional: install Ollama + Llama 3 locally
├── data/                    # raw/ processed/ annotations/ graphs/ (gitignored)
├── outputs/                 # Metric tables + figures (gitignored)
└── tests/
```

---

## 5. Reproducing the Paper

### 5.1 Prerequisites

- Python ≥ 3.10
- At least one API key: `OPENAI_API_KEY` (GPT-4o) **or** `ANTHROPIC_API_KEY` (Claude Sonnet)
- Optional: [Ollama](https://ollama.ai) with `llama3.2` pulled, for the open-source benchmark column

### 5.2 Setup

```bash
git clone https://github.com/vksabnima/rig_pipeline.git
cd rig_pipeline

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.template .env          # then edit .env with your API keys

# Optional: enable local Llama 3.2 benchmark column
bash setup_ollama.sh
```

### 5.3 End-to-end run

```bash
python run_pipeline.py                     # Full pipeline: download → extract → graph → align → score
python run_pipeline.py --skip-download     # Reuse cached PDFs in data/raw/
python run_pipeline.py --skip-annotation   # Reuse cached annotations in data/annotations/
python run_pipeline.py --dry-run           # Show planned stages without executing
```

### 5.4 Targeted re-runs (faster; no API calls if cached)

| Paper artifact | Command |
|---|---|
| Table 1 — corpus statistics | `python run_pipeline.py --skip-annotation` |
| Table 2 — multi-model benchmark | `python run_multimodel.py` |
| Table 2 — RIF 95 % bootstrap CIs | `python bootstrap_reviewed.py` |
| Table 3 — UCG cross-document alignment | `python rerun_alignment.py --threshold 0.50` |
| Figure 2 — metric heatmap | `python recompute_metrics.py --heatmap` |
| Figure 4 — τ sensitivity curve | `python task2_threshold.py` |
| Figure 5 — adversarial change rates | `python recompute_hrr.py --by-perturbation` |
| §5.8 — error analysis (top-30 failures) | `python error_analysis.py --bottom 30` |

### 5.5 Gold standard & validated annotations

The Basel III gold standard (758 clauses) was produced by a two-stage LLM process and validated per `human_review.py`:

- 449 / 758 (**59.2 %**) accepted as-is at mean confidence **0.909**
- 309 / 758 (**40.8 %**) corrected; `subject_gt` was the most frequently corrected field (**45.6 %** of corrections)

All validated annotations ship with the pipeline and are used as ground truth for RIF.

---

## 6. Data & Source Regulations

Source PDFs are **not redistributed** here; `src/pdf_downloader.py` fetches them from authoritative sources on first run:

| Framework | Source | Pages | Clauses |
|---|---|---|---|
| Basel III | Bank for International Settlements — [`bis.org/bcbs/publ/d424.pdf`](https://www.bis.org/bcbs/publ/d424.pdf) | 161 | 758 |
| SEC Reg BI | SEC — [`sec.gov/rules/final/2019/34-86031.pdf`](https://www.sec.gov/rules/final/2019/34-86031.pdf) | 770 | 911 |
| EU GDPR | EUR-Lex — Regulation (EU) 2016/679 | 88 | 498 |

---

## 7. Reviewer Quick-Check Guide

If you are reviewing the manuscript and want the fastest path to verifying a claim:

| Claim in paper | File to inspect | Expected artifact |
|---|---|---|
| Six-field tuple schema (Eq. 1) | `src/obligation_parser.py`, `gdpr_sonnet.py` | Prompt template + JSON schema |
| RCC / OAL / RIF / CAS formulas (Eq. 2–5) | `src/metrics.py` | One function per metric |
| HRR decomposition (Eq. 6–7) | `src/metrics.py`, `recompute_hrr.py` | `hrr_detection()`, `hrr_stability()` |
| 250-clause adversarial set + 15 baits | `src/adversarial_tester.py` | Perturbation generators + bait list |
| Bootstrap CI (n = 1000) | `bootstrap_reviewed.py`, `task3_bootstrap.py` | Seeded resampling; reproducible |
| τ = 0.50 justification (Fig. 4) | `task2_threshold.py` | Sweep over τ ∈ {0.40 … 0.70} |
| Cross-reference failure mode (§5.8) | `error_analysis.py` | Failure taxonomy on bottom-30 |

All randomness is seeded via `RANDOM_SEED = 42` in [`config/config.py`](config/config.py).

---

## 8. Limitations (as declared in the manuscript)

- Validated gold standard covers Basel III only; Reg BI and GDPR RIF rely on automated annotation.
- Claude Sonnet serves as both extractor and annotation validator, introducing potential self-referential bias in its own RIF score.
- Corpus is English-language; multilingual extension is future work.
- τ = 0.50 is empirically selected; production deployments should re-calibrate per domain.

---

## 9. Citation

If you use RIG, its metrics, or the UCG in your work, please cite:

```bibtex
@article{kumar2025rig,
  title   = {RIG: A Regulatory Intent Graph Methodology for LLM-Assisted Compliance Extraction from Financial Documents},
  author  = {Kumar, Vikash},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/vksabnima/rig_pipeline}
}
```

---

## 10. AI Tool Disclosure

Consistent with the manuscript's disclosure: Claude (Anthropic) and GPT-4o (OpenAI) are **components of the pipeline** (tuple extraction, annotation validation) and were also used as writing aids during manuscript preparation. All scientific claims, methodology, and conclusions are the sole responsibility of the author.
