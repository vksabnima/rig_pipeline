"""
PDF Extractor
-------------
Extracts raw text from regulatory PDFs, segments into sections,
and identifies OBLIGATION clauses using linguistic markers.
"""

import re
import json
from pathlib import Path
from typing import Optional

import pdfplumber

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import RAW_DIR, PROCESSED_DIR

# Linguistic markers that signal regulatory obligations
OBLIGATION_MARKERS = [
    r"\bshall\b", r"\bmust\b", r"\bis required to\b", r"\bare required to\b",
    r"\bobligation\b", r"\bmandatory\b", r"\bprohibited\b", r"\bshall not\b",
    r"\bmust not\b", r"\bensure that\b", r"\bis obliged to\b",
]
OBLIGATION_PATTERN = re.compile("|".join(OBLIGATION_MARKERS), re.IGNORECASE)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page, text}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text.strip()})
    return pages


def segment_into_sections(pages: list[dict]) -> list[dict]:
    """Group pages into logical sections based on heading patterns."""
    sections = []
    current_section = {"title": "Preamble", "pages": [], "text": ""}

    # Common regulatory heading patterns
    heading_re = re.compile(
        r"^(?:PART|CHAPTER|SECTION|ARTICLE|ANNEX|APPENDIX)\s+[IVXLCDM0-9]+",
        re.IGNORECASE | re.MULTILINE,
    )

    for page in pages:
        headings = heading_re.findall(page["text"])
        if headings and current_section["text"]:
            sections.append(current_section)
            current_section = {
                "title": headings[0].strip(),
                "pages": [],
                "text": "",
            }
        current_section["pages"].append(page["page"])
        current_section["text"] += "\n" + page["text"]

    if current_section["text"]:
        sections.append(current_section)

    return sections


def extract_obligation_clauses(sections: list[dict]) -> list[dict]:
    """Identify sentences containing obligation language."""
    clauses = []
    clause_id = 0

    for section in sections:
        sentences = re.split(r"(?<=[.;])\s+", section["text"])
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            if OBLIGATION_PATTERN.search(sentence):
                clause_id += 1
                clauses.append({
                    "clause_id": f"OBL-{clause_id:04d}",
                    "section": section["title"],
                    "text": sentence,
                    "pages": section["pages"],
                    "markers": OBLIGATION_PATTERN.findall(sentence),
                })

    return clauses


def process_pdf(pdf_name: str, pdf_path: Optional[Path] = None) -> dict:
    """Full extraction pipeline for a single PDF."""
    if pdf_path is None:
        pdf_path = RAW_DIR / f"{pdf_name}.pdf"

    if not pdf_path.exists():
        print(f"  [SKIP] {pdf_path.name} not found")
        return {"doc_id": pdf_name, "pages": [], "sections": [], "clauses": []}

    print(f"  [EXTRACT] {pdf_path.name}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"    Pages extracted: {len(pages)}")

    sections = segment_into_sections(pages)
    print(f"    Sections found: {len(sections)}")

    clauses = extract_obligation_clauses(sections)
    print(f"    Obligation clauses: {len(clauses)}")

    result = {
        "doc_id": pdf_name,
        "source_file": str(pdf_path.name),
        "total_pages": len(pages),
        "sections": len(sections),
        "clauses": clauses,
    }

    # Save processed output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{pdf_name}_clauses.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Saved to {out_path.name}")

    return result


def process_all_pdfs() -> list[dict]:
    """Process all PDFs in the raw directory."""
    print("\n=== PDF Extraction & Segmentation ===")
    results = []
    for pdf_file in sorted(RAW_DIR.glob("*.pdf")):
        name = pdf_file.stem
        results.append(process_pdf(name, pdf_file))
    return results


if __name__ == "__main__":
    process_all_pdfs()
