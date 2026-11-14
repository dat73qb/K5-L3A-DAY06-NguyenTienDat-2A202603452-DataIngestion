"""Evaluate retrieved evidence and preserve per-query traces."""
import re
import unicodedata
from typing import Any


def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for evidence matching."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_hit(chunk: dict[str, Any], question: dict[str, Any]) -> bool:
    """A chunk is a hit if it belongs to a gold source/page and contains the
    documented evidence (or an accepted alternative)."""
    if chunk["document_id"] != question.get("gold_source_id"):
        return False
    gold_pages = {question["gold_page"]}
    gold_pages.update(question.get("alt_gold_pages", []))
    if chunk["page"] not in gold_pages:
        return False
    evidence = question.get("evidence_quote", "")
    if not evidence:
        return True
    return normalize_for_match(evidence) in normalize_for_match(chunk["text"])


def evaluate_query(question: dict[str, Any],
                   hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Return hit@1/hit@3 flags plus the top-k provenance for one query."""
    flags = [is_hit(h, question) for h in hits]
    return {
        "id": question["id"],
        "split": question.get("split"),
        "question": question["question"],
        "gold_source_id": question.get("gold_source_id"),
        "gold_page": question.get("gold_page"),
        "filter": question.get("filter"),
        "hit1": bool(flags[0]) if flags else False,
        "hit3": bool(any(flags[:3])),
        "top_pages": [h["page"] for h in hits[:3]],
        "top_chapters": [h["chapter"] for h in hits[:3]],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by split (dev/test)."""
    out: dict[str, Any] = {}
    for split in ("dev", "test"):
        subset = [t for t in traces if t.get("split") == split]
        n = len(subset)
        hit1 = sum(1 for t in subset if t["hit1"])
        hit3 = sum(1 for t in subset if t["hit3"])
        out[split] = {"n": n, "hit1": hit1, "hit3": hit3}
    return out
