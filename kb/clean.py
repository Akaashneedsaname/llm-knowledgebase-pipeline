"""
Text normalization for PDF and OCR output.

Raw extracted text is noisy: running headers/footers on every page, page
numbers, words hyphenated across line breaks, and hard-wrapped lines. This
module turns it into clean paragraphs that chunk and embed well.
"""

import re
from collections import Counter

# Chapter / part headings, e.g. "CHAPTER IV", "Chapter 12. The Return", "Part II"
HEADING_RE = re.compile(
    r"^[ \t]*(?:Chapter|CHAPTER|Book|BOOK|Part|PART|Section|SECTION)[ \t]+(?:[0-9]+|[IVXLCDM]+)\b(?:[.:\- \t][^\n]{0,60})?$",
    re.M,
)


def _line_signature(line: str) -> str:
    """Normalize a line so 'Page 3' and 'Page 4' count as the same footer."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_lines(page: str, n: int = 2) -> list[tuple[int, str]]:
    """The first and last n non-empty lines of a page, with their line numbers."""
    lines = [(i, l) for i, l in enumerate(page.splitlines()) if l.strip()]
    return lines[:n] + lines[-n:] if len(lines) > 2 * n else lines


def remove_repeated_lines(pages: list[str], min_share: float = 0.5) -> list[str]:
    """
    Remove running headers and footers.

    Only lines at the top or bottom of a page are candidates. A short edge
    line that repeats on at least `min_share` of pages is treated as
    boilerplate. Chapter headings are never removed.
    """
    if len(pages) < 3:
        return pages

    def candidates(page):
        return {
            _line_signature(line)
            for _, line in _edge_lines(page)
            if len(line.strip()) <= 80 and not HEADING_RE.match(line)
        }

    counts = Counter()
    for page in pages:
        counts.update(candidates(page))

    threshold = max(2, int(len(pages) * min_share))
    boilerplate = {sig for sig, count in counts.items() if count >= threshold}

    cleaned = []
    for page in pages:
        drop = {i for i, line in _edge_lines(page) if _line_signature(line) in boilerplate}
        cleaned.append("\n".join(l for i, l in enumerate(page.splitlines()) if i not in drop))
    return cleaned


def clean_text(text: str) -> str:
    """
    Normalize a single page of text into paragraphs separated by blank lines.
    """
    text = text.replace("\r", "").replace("­", "")         # soft hyphens
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)              # re-join words hyphenated across lines
    text = re.sub(r"^\s*(page\s*)?\d+\s*$", "", text, flags=re.M | re.I)  # bare page numbers
    text = re.sub(r"[ \t]+", " ", text)

    # Keep headings on their own paragraph before unwrapping lines
    text = HEADING_RE.sub(lambda m: f"\n\n{m.group(0).strip()}\n\n", text)

    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)                 # unwrap hard-wrapped lines
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_pages(pages: list[str]) -> list[str]:
    """Full cleaning pass across a document's pages."""
    return [clean_text(p) for p in remove_repeated_lines(pages)]


def paragraphs(text: str) -> list[str]:
    """Split cleaned text into paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]
