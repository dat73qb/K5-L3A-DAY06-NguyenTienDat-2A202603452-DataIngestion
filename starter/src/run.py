"""Pipeline entry point: ingest -> chunk -> embed -> store -> benchmark.

Run from the ``starter/`` directory with ``python -m src.run``.
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk
from .store import PgVectorStore

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
BENCHMARK_DIR = BASE / "benchmark"
RESULTS_DIR = BASE / "results"

DOCUMENT_ID = "prealgebra-2e"
SOURCE = "prealgebra-2e.pdf"
DEFAULT_MODEL = "intfloat/multilingual-e5-base"

# Three controlled strategies:
#   A: fixed-size chunking, no metadata filter
#   B: sentence-aware chunking, no metadata filter
#   C: same chunking as B + metadata filter (only when a question carries one)
STRATEGIES = {
    "A": {"kind": "fixed", "size": 500, "overlap": 50, "filter": False},
    "B": {"kind": "paragraph", "max_size": 500, "filter": False},
    "C": {"kind": "paragraph", "max_size": 500, "filter": True},
}


def chunk_corpus(pages: list, strategy: str) -> list[Chunk]:
    cfg = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for page in pages:
        if cfg["kind"] == "fixed":
            chunks.extend(fixed_size(page, cfg["size"], cfg["overlap"]))
        else:
            chunks.extend(by_paragraph(page, cfg["max_size"]))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = sorted(len(c.text) for c in chunks)
    n = len(lengths)
    if n == 0:
        return {"count": 0}
    return {
        "count": n,
        "total_chars": sum(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(sum(lengths) / n, 1),
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
    }


def main() -> None:
    load_dotenv(BASE / ".env")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL missing; copy starter/.env.example to starter/.env")
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus (Chapters 1-2) and benchmark.
    pdf = DATA_DIR / "prealgebra-2e.pdf"
    pages = load_pdf(pdf, document_id=DOCUMENT_ID, source=SOURCE)
    content_pages = [p for p in pages if p.text]
    print(f"Corpus: {len(pages)} pages ({len(content_pages)} with text)")

    questions = json.loads(
        (BENCHMARK_DIR / "questions.json").read_text(encoding="utf-8")
    )
    dev = [q for q in questions if q.get("split") == "dev"]
    test = [q for q in questions if q.get("split") == "test"]
    print(f"Benchmark: {len(questions)} questions ({len(dev)} dev / {len(test)} test)")

    # 2. Embedder and vector store (same model/store/metric for all strategies).
    embedder = MyEmbedder(model_name)
    print(f"Embedder: {model_name} ({embedder.dimensions} dims)")
    store = PgVectorStore(db_url, embedder.dimensions)
    store.create_schema()

    results: dict = {"model": model_name, "dimensions": embedder.dimensions}
    try:
        for strategy in ("A", "B", "C"):
            t0 = time.time()
            chunks = chunk_corpus(pages, strategy)
            vectors = embedder.embed_documents([c.text for c in chunks])
            n_stored = store.upsert(strategy, chunks, vectors)
            ingest_s = round(time.time() - t0, 2)

            traces = []
            for q in questions:
                filters = q.get("filter") if STRATEGIES[strategy]["filter"] else None
                hits = store.search(
                    strategy,
                    embedder.embed_query(q["question"]),
                    top_k=3,
                    filters=filters,
                )
                ev = evaluate_query(q, hits)
                ev["hits"] = hits
                traces.append(ev)

            summary = summarize(traces)
            (RESULTS_DIR / f"trace_{strategy}.json").write_text(
                json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            results[strategy] = {
                "config": STRATEGIES[strategy],
                "chunk_stats": chunk_stats(chunks),
                "stored": n_stored,
                "ingest_seconds": ingest_s,
                "summary": summary,
            }
            print(
                f"  {strategy}: chunks={n_stored} ingest={ingest_s}s "
                f"dev Hit@1 {summary['dev']['hit1']}/{summary['dev']['n']} "
                f"Hit@3 {summary['dev']['hit3']}/{summary['dev']['n']} | "
                f"test Hit@1 {summary['test']['hit1']}/{summary['test']['n']} "
                f"Hit@3 {summary['test']['hit3']}/{summary['test']['n']}"
            )
    finally:
        store.close()

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
