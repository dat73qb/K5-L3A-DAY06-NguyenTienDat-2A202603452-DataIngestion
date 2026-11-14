"""Generate the manual-scoring worksheet: coherence positions and per-test
question top-3 chunks for grounding/metadata audit."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_pdf
from src.chunking import fixed_size, by_paragraph

BASE = Path(__file__).resolve().parents[1]
STRATS = {
    "A": lambda p: fixed_size(p, 500, 50),
    "B": lambda p: by_paragraph(p, 500),
    "C": lambda p: by_paragraph(p, 500),
}

pages = load_pdf(BASE / "data" / "prealgebra-2e.pdf", "prealgebra-2e", "x")
by_page = {p.page: p for p in pages}
questions = json.loads((BASE / "benchmark" / "questions.json").read_text(encoding="utf-8"))
test_qs = [q for q in questions if q["split"] == "test"]

out = {"coherence": [], "grounding": []}

# --- Coherence: 10 fixed positions = first sentence of pages 7..16 ---
position_pages = list(range(7, 17))
for page_no in position_pages:
    page = by_page[page_no]
    # marker = first ~120 chars (a self-contained opening span)
    marker = page.text[:120]
    row = {"page": page_no, "marker": marker, "strategies": {}}
    for s, fn in STRATS.items():
        chunks = fn(page)
        hit = next((c for c in chunks if marker[:80] in c.text), chunks[0] if chunks else None)
        row["strategies"][s] = {
            "chunk_id": hit.chunk_id if hit else None,
            "text": (hit.text[:600] if hit else ""),
        }
    out["coherence"].append(row)

# --- Grounding + metadata: test questions x strategies x top-3 ---
for s in ("A", "B", "C"):
    tr = json.loads((BASE / "results" / f"trace_{s}.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in tr}
    for q in test_qs:
        t = by_id[q["id"]]
        out["grounding"].append({
            "strategy": s,
            "question_id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold_page": q["gold_page"],
            "evidence_quote": q["evidence_quote"],
            "hits": [
                {"page": h["page"], "chapter": h["chapter"], "section": h["section"],
                 "score": h["score"], "text": h["text"][:500]}
                for h in t["hits"][:3]
            ],
        })

(BASE / "results" / "manual_worksheet.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Wrote results/manual_worksheet.json")
print("Coherence positions:", len(out["coherence"]))
print("Grounding rows:", len(out["grounding"]))
