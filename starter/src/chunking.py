"""Fixed-size and sentence-aware chunking strategies.

Strategy A uses fixed character windows (can cut mid-sentence).
Strategies B/C group whole sentences up to a size limit (never cut a sentence).
"""
import re

from .models import Chunk, DocumentPage

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split English prose on sentence boundaries only."""
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _make_chunk(page: DocumentPage, text: str, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{page.document_id}|p{page.page}|{idx}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=dict(page.metadata),
    )


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split page text into overlapping fixed-size windows (strategy A)."""
    text = page.text
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    idx = 0
    while start < n:
        end = min(start + size, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_make_chunk(page, chunk_text, idx))
            idx += 1
        if end >= n:
            break
        start = end - overlap
    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 500) -> list[Chunk]:
    """Group whole sentences up to max_size chars, never cutting mid-sentence.

    A single over-long sentence is split at word boundaries to respect max_size.
    Tiny fragments (stray words) are merged into the next sentence so no chunk
    is left without enough context.
    """
    sentences = _merge_short(_split_sentences(page.text), min_len=20)
    if not sentences:
        return []
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, buf_len, idx
        if buf:
            chunks.append(_make_chunk(page, " ".join(buf), idx))
            idx += 1
            buf = []
            buf_len = 0

    for sentence in sentences:
        if len(sentence) > max_size:
            # split the over-long sentence at word boundaries
            flush()
            for piece in _split_long_sentence(sentence, max_size):
                chunks.append(_make_chunk(page, piece, idx))
                idx += 1
            continue
        if buf and buf_len + len(sentence) + 1 > max_size:
            flush()
        buf.append(sentence)
        buf_len += len(sentence) + (1 if buf else 0)
    flush()
    return chunks


def _merge_short(sentences: list[str], min_len: int = 20) -> list[str]:
    """Merge sentence fragments shorter than min_len into the next sentence."""
    if not sentences:
        return []
    merged: list[str] = []
    carry = ""
    for s in sentences:
        s = (carry + " " + s).strip() if carry else s
        if len(s) < min_len:
            carry = s
        else:
            merged.append(s)
            carry = ""
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)
    return [s for s in merged if s.strip()]


def _split_long_sentence(sentence: str, max_size: int) -> list[str]:
    """Split an over-long sentence into word-aligned pieces <= max_size."""
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        if current and current_len + len(w) + 1 > max_size:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(w)
        current_len += len(w) + (1 if current_len else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces
