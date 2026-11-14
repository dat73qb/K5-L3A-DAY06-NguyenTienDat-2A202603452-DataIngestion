"""Read and normalize source data (OpenStax Prealgebra 2e, Chapters 1-2)."""
from pathlib import Path
import re
import unicodedata

from .models import DocumentPage

# Section map for Prealgebra 2e, Chapters 1-2.
# "start" = printed (physical) page number where the section begins.
# Source: book table of contents (PDF pages 6-8).
SECTIONS = [
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "1.1 Introduction to Whole Numbers", "start": 7},
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "1.2 Add Whole Numbers", "start": 22},
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "1.3 Subtract Whole Numbers", "start": 38},
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "1.4 Multiply Whole Numbers", "start": 51},
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "1.5 Divide Whole Numbers", "start": 68},
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "Chapter 1 Review", "start": 85},
    {"chapter": 1, "chapter_title": "Whole Numbers", "section": "Chapter 1 Exercises", "start": 87},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "2.1 Use the Language of Algebra", "start": 95},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "2.2 Evaluate, Simplify, and Translate Expressions", "start": 113},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "2.3 Solving Equations Using the Subtraction and Addition Properties of Equality", "start": 126},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "2.4 Find Multiples and Factors", "start": 139},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "2.5 Prime Factorization and the Least Common Multiple", "start": 152},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "Chapter 2 Review", "start": 162},
    {"chapter": 2, "chapter_title": "The Language of Algebra", "section": "Chapter 2 Exercises", "start": 165},
]

# Verified: printed page number == PDF page index - 9.
PDF_OFFSET = 9
PAGE_START = 7    # printed page 7  -> PDF index 16
PAGE_END = 169    # printed page 169 -> PDF index 178 (end of Chapter 2)


def section_for_page(page: int) -> dict:
    """Return the section dict that a printed page number belongs to."""
    current = SECTIONS[0]
    for s in SECTIONS:
        if s["start"] <= page:
            current = s
        else:
            break
    return current


def load_pdf(path: Path, document_id: str, source: str,
             page_start: int = PAGE_START, page_end: int = PAGE_END) -> list[DocumentPage]:
    """Extract one DocumentPage per printed page in [page_start, page_end].

    The printed page number (used as the traceable ``page`` identifier) is
    derived from the PDF page index via a fixed offset verified against the
    book's running footer.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for pdf_index in range(page_start + PDF_OFFSET, page_end + PDF_OFFSET + 1):
        printed = pdf_index - PDF_OFFSET
        raw = reader.pages[pdf_index].extract_text() or ""
        text = normalize_text(raw)
        sec = section_for_page(printed)
        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=printed,
            text=text,
            metadata={
                "chapter": sec["chapter_title"],
                "chapter_number": sec["chapter"],
                "section": sec["section"],
            },
        ))
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding sentence structure.

    Keeps sentence-ending punctuation so sentence-aware chunking can still
    find boundaries after whitespace is collapsed.
    """
    if not text:
        return ""
    text = text.replace("\u00ad", "")        # soft hyphen
    text = text.replace("\xa0", " ")         # non-breaking space
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufffd", " ")       # unmappable glyph -> space
    text = re.sub(r"\s+", " ", text)         # collapse all whitespace/newlines
    return text.strip()
