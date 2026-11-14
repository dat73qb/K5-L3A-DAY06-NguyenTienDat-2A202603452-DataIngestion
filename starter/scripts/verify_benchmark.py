"""Verify benchmark evidence actually exists on the gold page after normalization.

Also checks: corpus page count, question/split/filter counts.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_pdf
from src.evaluation import normalize_for_match

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


def main() -> None:
    pages = load_pdf(DATA / "prealgebra-2e.pdf", "prealgebra-2e", "prealgebra-2e.pdf")
    by_page = {p.page: p for p in pages}
    content_pages = [p for p in pages if p.text]

    questions = json.loads((BASE / "benchmark" / "questions.json").read_text(encoding="utf-8"))

    print(f"Corpus: {len(pages)} pages ({len(content_pages)} with text) >= 30: {len(content_pages) >= 30}")
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    filters = [q for q in questions if q.get("filter")]
    filters_test = [q for q in filters if q["split"] == "test"]
    print(f"Questions: {len(questions)} total, {len(dev)} dev, {len(test)} test")
    print(f"Filter questions: {len(filters)} total, {len(filters_test)} in test")

    errors = 0
    for q in questions:
        page = by_page.get(q["gold_page"])
        if page is None:
            print(f"  [MISSING PAGE] {q['id']}: page {q['gold_page']} not in corpus")
            errors += 1
            continue
        norm_text = normalize_for_match(page.text)
        norm_ev = normalize_for_match(q["evidence_quote"])
        if norm_ev not in norm_text:
            print(f"  [EVIDENCE NOT FOUND] {q['id']} page {q['gold_page']}: {q['evidence_quote']!r}")
            errors += 1
        else:
            print(f"  [OK] {q['id']} page {q['gold_page']} ({q['split']}, {q['type']})")

    if errors:
        print(f"\n{errors} evidence verification error(s).")
        sys.exit(1)
    print("\nAll evidence quotes verified on their gold pages.")


if __name__ == "__main__":
    main()
