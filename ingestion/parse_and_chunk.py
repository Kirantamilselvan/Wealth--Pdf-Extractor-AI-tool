"""
Parse raw SEC filing HTML into clean text, split into section-aware chunks,
and write structured JSON documents ready for the model/RAG layers.

Usage:
    python ingestion/parse_and_chunk.py
"""
import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))

# Common 10-K/10-Q section headers we try to split on. Filings are messy HTML,
# so this is heuristic, not exhaustive -- good enough for a v1 pipeline.
SECTION_PATTERNS = [
    r"item\s+1a\.?\s*risk factors",
    r"item\s+7\.?\s*management.?s discussion and analysis",
    r"item\s+1\.?\s*business",
    r"item\s+7a\.?\s*quantitative and qualitative disclosures",
    r"item\s+8\.?\s*financial statements",
]

CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


def html_to_text(html_path: Path) -> str:
    soup = BeautifulSoup(html_path.read_text(errors="ignore"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_sections(text: str) -> dict:
    """Best-effort split into named sections; falls back to a single 'full_text' section."""
    lower = text.lower()
    matches = []
    for pattern in SECTION_PATTERNS:
        for m in re.finditer(pattern, lower):
            matches.append((m.start(), pattern))
    matches.sort()

    if not matches:
        return {"full_text": text}

    sections = {}
    for i, (start, pattern) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        name = re.sub(r"[^a-z0-9]+", "_", pattern).strip("_")
        sections[name] = text[start:end].strip()
    return sections


def chunk_text(text: str, size: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def process_filing(html_path: Path):
    meta_path = html_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    text = html_to_text(html_path)
    sections = split_into_sections(text)

    chunks = []
    for section_name, section_text in sections.items():
        for i, chunk in enumerate(chunk_text(section_text)):
            chunks.append(
                {
                    "chunk_id": f"{html_path.stem}::{section_name}::{i}",
                    "section": section_name,
                    "text": chunk,
                }
            )

    doc = {
        "source_file": str(html_path),
        "ticker": html_path.parent.name,
        "form": meta.get("form"),
        "filing_date": meta.get("filing_date"),
        "company_name": meta.get("company_name"),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    out_path = PROCESSED_DIR / f"{doc['ticker']}_{html_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    return out_path


def main():
    html_files = list(RAW_DIR.glob("*/*.html"))
    if not html_files:
        print(f"No raw HTML files found under {RAW_DIR}. Run ingestion/edgar_fetch.py first.")
        return

    print(f"Processing {len(html_files)} filings...")
    for html_path in html_files:
        out_path = process_filing(html_path)
        print(f"  wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
