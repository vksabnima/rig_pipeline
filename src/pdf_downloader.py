"""
PDF Downloader
--------------
Downloads regulatory PDFs with browser User-Agent, size verification,
and curl fallback instructions on failure.
"""

import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import PDF_SOURCES, RAW_DIR, MIN_PDF_SIZE_KB

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def download_pdf(name: str, source: dict, output_dir: Path) -> bool:
    """Download a single PDF. Returns True on success."""
    url = source["url"]
    dest = output_dir / source["filename"]

    if dest.exists() and dest.stat().st_size > MIN_PDF_SIZE_KB * 1024:
        print(f"  [SKIP] {source['filename']} already exists ({dest.stat().st_size // 1024} KB)")
        return True

    print(f"  [DOWNLOAD] {name}: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = dest.stat().st_size / 1024
        if size_kb < MIN_PDF_SIZE_KB:
            print(f"  [WARN] {source['filename']} is only {size_kb:.1f} KB — may be incomplete")
            dest.unlink()
            _print_curl_fallback(url, dest)
            return False

        print(f"  [OK] {source['filename']} — {size_kb:.0f} KB")
        return True

    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        _print_curl_fallback(url, dest)
        return False


def _print_curl_fallback(url: str, dest: Path):
    """Print a manual curl command for the user."""
    print(f"\n  Manual fallback — run this command:")
    print(f'  curl -L -o "{dest}" -H "User-Agent: {HEADERS["User-Agent"]}" "{url}"\n')


def download_all_pdfs() -> dict:
    """Download all configured PDFs. Returns dict of {name: success_bool}."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== PDF Download ===")
    results = {}
    for name, source in PDF_SOURCES.items():
        results[name] = download_pdf(name, source, RAW_DIR)
    return results


if __name__ == "__main__":
    download_all_pdfs()
