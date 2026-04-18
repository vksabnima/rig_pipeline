"""
RIG Pipeline Configuration
--------------------------
Central configuration for the Regulatory Intent Graph pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
GRAPHS_DIR = DATA_DIR / "graphs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── PDF Sources ──────────────────────────────────────────────────────────────
PDF_SOURCES = {
    "basel3": {
        "url": "https://www.bis.org/bcbs/publ/d424.pdf",
        "filename": "basel3.pdf",
        "description": "Basel III: Finalising post-crisis reforms",
    },
    "regbi": {
        "url": "https://www.sec.gov/rules/final/2019/34-86031.pdf",
        "filename": "regbi.pdf",
        "description": "Regulation Best Interest (Reg BI)",
    },
}

MIN_PDF_SIZE_KB = 100  # Minimum acceptable PDF file size in KB

# ── Model Configuration ──────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_AVAILABLE = os.getenv("OLLAMA_AVAILABLE", "false").lower() == "true"

# Models used in the pipeline
MODELS = {
    "annotator": "claude-sonnet-4-20250514",       # Auto-annotation of obligations
    "extractor": "gpt-4o",                          # Obligation tuple extraction
    "local": "llama3",                               # Local model via Ollama (optional)
}

# ── Sentence Transformer ────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Graph Settings ───────────────────────────────────────────────────────────
GRAPH_FORMAT = "graphml"  # Export format for knowledge graphs
SIMILARITY_THRESHOLD = 0.50  # Minimum cosine similarity for cross-doc alignment

# ── Adversarial Testing ─────────────────────────────────────────────────────
ADVERSARIAL_PERTURBATION_TYPES = [
    "synonym_swap",
    "negation_injection",
    "clause_reorder",
    "threshold_mutation",
    "entity_substitution",
]
ADVERSARIAL_NUM_SAMPLES = 50  # Number of adversarial samples per perturbation type

# ── Metrics ──────────────────────────────────────────────────────────────────
# RCC: Regulatory Clause Coverage
# OAL: Obligation Alignment Level
# RIF: Regulatory Intent Fidelity
# CAS: Cross-document Alignment Score
# HRR: Hallucination Resistance Rate
METRIC_NAMES = ["RCC", "OAL", "RIF", "CAS", "HRR"]

# ── Pipeline Settings ────────────────────────────────────────────────────────
RANDOM_SEED = 42
LOG_LEVEL = "INFO"
