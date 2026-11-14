# Processing log — Day 06

- **Run timestamp:** 2026-11-13T19:12:03
- **Corpus:** Prealgebra 2e (OpenStax), chapters 1-2, printed pages 7-169
- **Printed page = PDF index - 9** (verified against the running footer)

## Library versions
```json
{
  "sentence-transformers": "6.3.0",
  "torch": "2.15.0",
  "transformers": "5.22.2",
  "psycopg": "3.3.8",
  "pgvector": "0.5.1",
  "pypdf": "6.21.4",
  "python-dotenv": "1.2.4",
  "numpy": "2.5.4"
}
```

## Ingestion diagnostics
- Pages processed: 163 (all in selected range)
- Pages with empty text: 0 
- Pages containing U+FFFD replacement chars: 0
- Total U+FFFD replaced: 0
- Total normalized chars: 203235 (mean 1247/page)

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
  deterministic ids `{strategy}|{document_id}|p{page}|{idx}` inside a transaction,
  so re-running is idempotent.
- Verified in DB: chunk count == distinct chunk_id count for every strategy
  (A 519/519, B 550/550, C 550/550) -> no duplicate rows.
