"""Extract text from the formats regulatory documents actually arrive in.

`ingest.py` chunks text, but nothing produced text from a PDF or a Word file, which is
how every SAMA and CBUAE circular arrives. This closes that end.

Three things matter more here than raw extraction quality:

Page numbers. `PolicyChunk.page_number` existed and nothing set it, so a citation could
name a clause but not where to find it in the source. A reviewer verifying "SAMA-CIRC-4
#7.2" against a 90-page PDF needs the page.

Repeated headers and footers. A running header on every page ("SAMA - Confidential - Page
4 of 90") otherwise lands inside clause text, where it pollutes both the embedding and the
BM25 term statistics. They are detected by repetition across pages rather than by position,
because margins vary between documents.

Tables. Extracting a table as flowing text interleaves the columns and produces sentences
that were never in the document - a provisioning matrix becomes a row of numbers with no
headings attached. Tables are extracted separately and rendered row-wise so each row keeps
its own headings.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# A line has to appear on at least this share of pages to count as a running header.
REPEAT_THRESHOLD = 0.6
# Below this many pages, repetition is not evidence of a header.
MIN_PAGES_FOR_REPEAT_DETECTION = 3
# Lines longer than this are body text even if they repeat.
MAX_HEADER_CHARS = 120


class ExtractionError(RuntimeError):
    """Raised when a document cannot be read as text."""


@dataclass
class Page:
    """One page of extracted text, keeping the number the reader will cite."""

    page_number: int
    text: str


def _normalise(line: str) -> str:
    """Strip the varying parts of a running header so repetition is detectable.

    "Page 4 of 90" and "Page 5 of 90" are the same header. Without this every page's
    header looks unique and none is ever removed.
    """
    collapsed = re.sub(r"\s+", " ", line).strip()
    return re.sub(r"\d+", "#", collapsed)


def find_repeated_lines(pages: list[Page]) -> set[str]:
    """Normalised lines that appear on most pages: headers, footers, watermarks."""
    if len(pages) < MIN_PAGES_FOR_REPEAT_DETECTION:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        seen = {
            _normalise(line) for line in page.text.splitlines()
            if line.strip() and len(line.strip()) <= MAX_HEADER_CHARS
        }
        counts.update(seen)
    threshold = max(2, int(len(pages) * REPEAT_THRESHOLD))
    return {line for line, count in counts.items() if count >= threshold}


def strip_repeated_lines(pages: list[Page], repeated: set[str]) -> list[Page]:
    stripped = []
    for page in pages:
        kept = [
            line for line in page.text.splitlines()
            if _normalise(line) not in repeated or not line.strip()
        ]
        stripped.append(Page(page.page_number, "\n".join(kept).strip()))
    return stripped


def _render_table(rows: list[list[str | None]]) -> str:
    """Render a table row-wise, each cell carrying its own column heading.

    Reading a provisioning matrix as flowing text gives "1 2 3 0.5% 3% 100%", which is
    not recoverable. Row-wise it stays "Stage: 2 | Coverage: 3%", which a model can cite
    and a reviewer can check.
    """
    if not rows:
        return ""
    header = [str(cell or "").strip() for cell in rows[0]]
    lines = []
    for row in rows[1:]:
        cells = [str(cell or "").strip() for cell in row]
        if not any(cells):
            continue
        pairs = [
            f"{header[i]}: {cells[i]}" if i < len(header) and header[i] else cells[i]
            for i in range(len(cells)) if cells[i]
        ]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def extract_pdf(path: str | Path, include_tables: bool = True) -> list[Page]:
    """Extract a PDF page by page, tables rendered separately from the prose."""
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on the extra being installed
        raise ExtractionError(
            "PDF extraction needs pymupdf; install the 'documents' extra") from exc

    pages: list[Page] = []
    with pymupdf.open(str(path)) as document:
        for index, page in enumerate(document, start=1):
            parts = [page.get_text("text").strip()]
            if include_tables:
                try:
                    for table in page.find_tables().tables:
                        rendered = _render_table(table.extract())
                        if rendered:
                            parts.append(rendered)
                except Exception:
                    # Table detection is best-effort and version-dependent. Losing a table
                    # is recoverable; losing the page because detection raised is not.
                    pass
            pages.append(Page(index, "\n\n".join(p for p in parts if p)))
    if not any(page.text for page in pages):
        raise ExtractionError(
            f"{path} yielded no text. A scanned PDF needs OCR before ingestion, and "
            "ingesting it empty would silently shrink the corpus."
        )
    return pages


def extract_docx(path: str | Path) -> list[Page]:
    """Extract a Word document.

    Word has no page concept until it is laid out, so everything is page 1 rather than a
    guessed number. A wrong page reference is worse than none: it sends a reviewer to the
    wrong part of the document and looks authoritative doing it.
    """
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on the extra being installed
        raise ExtractionError(
            "Word extraction needs python-docx; install the 'documents' extra") from exc

    document = docx.Document(str(path))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Word headings carry their level in the style name. Re-emitting them as markdown
        # headings is what lets split_sections see the clause structure.
        style = (paragraph.style.name or "") if paragraph.style else ""
        match = re.match(r"Heading (\d)", style)
        blocks.append(f"{'#' * int(match.group(1))} {text}" if match else text)

    for table in document.tables:
        rendered = _render_table([[cell.text for cell in row.cells] for row in table.rows])
        if rendered:
            blocks.append(rendered)

    text = "\n\n".join(blocks).strip()
    if not text:
        raise ExtractionError(f"{path} yielded no text")
    return [Page(1, text)]


def extract(path: str | Path, include_tables: bool = True) -> list[Page]:
    """Extract by suffix, with headers and footers removed."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        pages = extract_pdf(path, include_tables=include_tables)
    elif suffix in (".docx", ".doc"):
        pages = extract_docx(path)
    elif suffix in (".txt", ".md"):
        pages = [Page(1, Path(path).read_text(encoding="utf-8"))]
    else:
        raise ExtractionError(f"Unsupported document type: {suffix or path}")
    return strip_repeated_lines(pages, find_repeated_lines(pages))
