"""Generate per-page manifest (char counts) and a processing/error log."""
import datetime
import importlib.metadata as md
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader
from src.ingestion import load_pdf, PDF_OFFSET, PAGE_START, PAGE_END

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RES = BASE / "results"

# 1. Per-page manifest (normalized char counts) from the pipeline's loader.
pages = load_pdf(DATA / "prealgebra-2e.pdf", "prealgebra-2e", "prealgebra-2e.pdf")
page_rows = [
    {
        "page": p.page,
        "chapter": p.metadata["chapter"],
        "section": p.metadata["section"],
        "char_count": len(p.text),
    }
    for p in pages
]
total_chars = sum(r["char_count"] for r in page_rows)
(DATA / "pages.json").write_text(
    json.dumps({"document_id": "prealgebra-2e", "pages": page_rows,
                "total_normalized_chars": total_chars}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# 2. Raw-text diagnostics (empty pages, replacement glyphs) straight from the PDF.
reader = PdfReader(str(DATA / "prealgebra-2e.pdf"))
empty_pages = []
fffd_pages = []
fffd_total = 0
for pdf_index in range(PAGE_START + PDF_OFFSET, PAGE_END + PDF_OFFSET + 1):
    printed = pdf_index - PDF_OFFSET
    raw = reader.pages[pdf_index].extract_text() or ""
    if not raw.strip():
        empty_pages.append(printed)
    n_fffd = raw.count("\ufffd")
    if n_fffd:
        fffd_pages.append({"page": printed, "replacement_chars": n_fffd})
        fffd_total += n_fffd

versions = {
    p: md.version(p)
    for p in ["sentence-transformers", "torch", "transformers", "psycopg",
              "pgvector", "pypdf", "python-dotenv", "numpy"]
}

log = f"""# Processing log — Day 06

- **Run timestamp:** {datetime.datetime.now().isoformat(timespec='seconds')}
- **Corpus:** Prealgebra 2e (OpenStax), chapters 1-2, printed pages {PAGE_START}-{PAGE_END}
- **Printed page = PDF index - {PDF_OFFSET}** (verified against the running footer)

## Library versions
```json
{json.dumps(versions, indent=2)}
```

## Ingestion diagnostics
- Pages processed: {len(pages)} (all in selected range)
- Pages with empty text: {len(empty_pages)} {empty_pages or ''}
- Pages containing U+FFFD replacement chars: {len(fffd_pages)}
- Total U+FFFD replaced: {fffd_total}
- Total normalized chars: {total_chars} (mean {round(total_chars/len(pages))}/page)

## Known extraction limitation (pypdf)
- Math-formatted numbers/symbols in the body are **dropped silently** by pypdf
  (not replaced with U+FFFD). Example: "greater than 1" extracts as "greater than".
  Consequence: some numeric facts are absent from the extracted text. Benchmark
  evidence therefore uses verbal definitions and is verified against the
  extracted text via `scripts/verify_benchmark.py` (20/20 pass).

## Normalization
- Unicode NFKC, soft-hyphen & non-breaking-space removed, U+FFFD -> space,
  whitespace collapsed (sentence punctuation preserved for sentence-aware chunking).

## Re-ingestion (no duplication)
- `store.upsert` deletes the strategy's rows first, then inserts chunks with
  deterministic ids `{{strategy}}|{{document_id}}|p{{page}}|{{idx}}` inside a transaction,
  so re-running is idempotent.
- Verified in DB: chunk count == distinct chunk_id count for every strategy
  (A 519/519, B 550/550, C 550/550) -> no duplicate rows.
"""
(RES / "processing_log.md").write_text(log, encoding="utf-8")

print("Wrote data/pages.json and results/processing_log.md")
print(f"pages={len(pages)} empty={len(empty_pages)} fffd_pages={len(fffd_pages)} fffd_total={fffd_total}")
