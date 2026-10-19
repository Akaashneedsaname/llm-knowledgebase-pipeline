"""
Document-level metadata extraction and labeling.

Layers, cheapest first:
1. Embedded file metadata (PDF title/author fields)
2. Heuristics on the text (Project Gutenberg headers, title pages, "by <author>")
3. Optional LLM labeling (summary, topics, genre) when a Gemini key is configured
"""

import re
from collections import Counter
from typing import Optional

from kb.clean import HEADING_RE, paragraphs
from kb.config import settings
from kb import llm

STOPWORDS = set("""
a an and are as at be been but by for from had has have he her his i if in into is it its
me my no not of on or our she so than that the their them then there they this to too up
upon was we were what when which who will with would you your all any can could did do
does one out said very more some such only own same just now also about after again
before being both each few most other over under while him those these here where how
""".split())


def _from_gutenberg_header(text: str) -> dict:
    """Project Gutenberg texts start with 'Title:' / 'Author:' lines."""
    found = {}
    for field in ("title", "author", "language"):
        m = re.search(rf"^\s*{field}:\s*(.+)$", text, re.I | re.M)
        if m:
            found[field] = m.group(1).strip()
    return found


def _titlecase(line: str) -> str:
    return line.title() if line.isupper() else line


def _from_title_page(text: str) -> dict:
    """Guess title and author from the first lines of a title page."""
    lines = [l.strip() for l in text.splitlines() if l.strip()][:12]
    found = {}
    for i, line in enumerate(lines):
        m = re.match(r"^by\s+(.{3,60})$", line, re.I)
        if m:
            found["author"] = m.group(1).strip(" .")
            if i > 0:
                found["title"] = ": ".join(_titlecase(l) for l in lines[max(0, i - 2):i])
            break
    if "title" not in found and lines and len(lines[0]) <= 80:
        found["title"] = lines[0]
    return found


def basic_metadata(first_page_text: str, embedded: dict, filename: str) -> dict:
    """Combine embedded metadata and text heuristics into title/author/language."""
    meta = {"title": None, "author": None, "language": None}

    for source in (_from_title_page(first_page_text), _from_gutenberg_header(first_page_text)):
        meta.update({k: v for k, v in source.items() if v})

    # Embedded PDF fields win when present
    if embedded.get("title"):
        meta["title"] = embedded["title"]
    if embedded.get("author"):
        meta["author"] = embedded["author"]

    if meta["title"]:
        meta["title"] = meta["title"].title() if meta["title"].isupper() else meta["title"]
    else:
        meta["title"] = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    return meta


def detect_chapters(pages: list[str]) -> list[dict]:
    """Find chapter headings and the page each one starts on."""
    chapters = []
    for page_number, text in enumerate(pages, start=1):
        for para in paragraphs(text):
            if HEADING_RE.match(para):
                chapters.append({"heading": para.strip(), "page": page_number})
    return chapters


def keyword_labels(text: str, n: int = 8) -> list[str]:
    """Frequency-based keywords, used when no LLM is configured."""
    words = re.findall(r"[a-z][a-z'-]{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counts.most_common(n)]


def _label_excerpt(text: str) -> str:
    """
    A small, representative excerpt for labeling: slices from the start,
    middle and end of the document instead of the whole text, so labeling
    costs the same small amount for a 5-page or a 500-page document.
    """
    budget = settings.llm_label_chars
    if len(text) <= budget:
        return text
    part = budget // 3
    mid = len(text) // 2
    return "\n...\n".join([text[:part], text[mid - part // 2: mid + part // 2], text[-part:]])


def llm_labels(text_sample: str) -> Optional[dict]:
    """
    Ask the LLM for a short structured label set for the document.

    Returns None when no LLM is configured or the call fails.
    """
    prompt = (
        "You are labeling documents for an LLM knowledgebase. Read the excerpt and return JSON with keys: "
        '"summary" (2 sentences), "topics" (3-8 short lowercase tags), "genre" (one or two words), '
        '"language" (ISO 639-1 code).\n\nEXCERPT:\n' + _label_excerpt(text_sample)
    )
    result = llm.generate_json(prompt)
    if not isinstance(result, dict):
        return None
    return {k: result.get(k) for k in ("summary", "topics", "genre", "language")}


def build_metadata(pages: list[str], embedded: dict, filename: str, raw_first_page: str = None) -> dict:
    """
    Full metadata pass for one document.

    Args:
        pages: Cleaned page texts
        embedded: Embedded file metadata
        filename: Source file name (last-resort title)
        raw_first_page: First page before cleaning. Title pages need their original
            line breaks, and cleaning removes the running header that often holds the title.
    """
    first_page = raw_first_page if raw_first_page is not None else (pages[0] if pages else "")
    meta = basic_metadata(first_page, embedded, filename)
    meta["chapters"] = detect_chapters(pages)

    full_text = "\n\n".join(pages)
    labels = llm_labels(full_text) if llm.enabled() else None
    if labels:
        meta["labels"] = {**labels, "labeled_by": "llm"}
        meta["language"] = meta["language"] or labels.get("language")
    else:
        meta["labels"] = {"keywords": keyword_labels(full_text), "labeled_by": "keywords"}
    return meta
