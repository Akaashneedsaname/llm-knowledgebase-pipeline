"""
Paragraph-aware, token-based chunking.

Chunks are built from whole paragraphs up to a token budget, with a token
overlap carried into the next chunk. Paragraphs longer than the budget are
split on token boundaries. A chapter heading always starts a new chunk, and
every chunk keeps its page range and chapter so answers can cite exactly
where they came from.
"""

from dataclasses import dataclass
from typing import Optional

import tiktoken

from kb.clean import HEADING_RE, paragraphs
from kb.config import settings

_encoding = tiktoken.get_encoding("cl100k_base")

# Smaller leftovers before a chapter heading are merged into the next chunk
MIN_CHUNK_TOKENS = 40


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_encoding.encode(text))


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    chapter: Optional[str]
    token_count: int


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Split a single oversized paragraph on token boundaries."""
    tokens = _encoding.encode(text)
    pieces, start = [], 0
    while start < len(tokens):
        pieces.append(_encoding.decode(tokens[start:start + size]))
        if start + size >= len(tokens):
            break
        start += size - overlap
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` tokens of text, used to seed the next chunk."""
    if not overlap:
        return ""
    tokens = _encoding.encode(text)
    return _encoding.decode(tokens[-overlap:]).strip() if len(tokens) > overlap else ""


def chunk_pages(pages: list[str], chunk_size: int = None, chunk_overlap: int = None) -> list[Chunk]:
    """
    Chunk a document's cleaned pages.

    Args:
        pages: Cleaned page texts (paragraphs separated by blank lines)
        chunk_size: Target chunk size in tokens (defaults to settings)
        chunk_overlap: Overlap between consecutive chunks in tokens (defaults to settings)

    Returns:
        List of Chunk objects in document order
    """
    size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    # Paragraphs as [first_page, last_page, text]; a sentence that runs over a page
    # break (no closing punctuation, next page starts lowercase) is rejoined
    paras: list[list] = []
    for page_number, text in enumerate(pages, start=1):
        for i, para in enumerate(paragraphs(text)):
            prev = paras[-1] if paras else None
            if (i == 0 and prev and prev[1] == page_number - 1 and para[:1].islower()
                    and not prev[2].rstrip().endswith((".", "!", "?", '"', "'", ":"))):
                prev[1], prev[2] = page_number, f"{prev[2]} {para}"
            else:
                paras.append([page_number, page_number, para])

    # Flatten into a stream of (first_page, last_page, chapter, text)
    stream = []
    chapter = None
    piece_size = max(size - overlap - 4, size // 2)  # leave room for the overlap seed
    for first, last, para in paras:
        if HEADING_RE.match(para):
            chapter = para.strip()
        pieces = _split_long(para, piece_size, overlap) if count_tokens(para) > size else [para]
        stream.extend((first, last, chapter, piece) for piece in pieces)

    chunks: list[Chunk] = []
    buffer: list[tuple[int, int, str]] = []   # (first_page, last_page, text); may start with an overlap seed
    buffer_tokens = 0
    buffer_chapter = None
    has_new_content = False                   # True once the buffer holds more than the seed

    def emit():
        text = "\n\n".join(t for _, _, t in buffer)
        chunks.append(Chunk(
            index=len(chunks),
            text=text,
            page_start=buffer[0][0],
            page_end=buffer[-1][1],
            chapter=buffer_chapter,
            token_count=count_tokens(text),
        ))
        return text

    for first, last, para_chapter, para in stream:
        para_tokens = count_tokens(para)
        starts_chapter = HEADING_RE.match(para) is not None

        # A chapter heading starts a new chunk, unless the buffer only holds a
        # few stray tokens (e.g. a title line), which are folded into it instead
        chapter_break = starts_chapter and buffer_tokens >= MIN_CHUNK_TOKENS
        if has_new_content and (chapter_break or buffer_tokens + para_tokens > size):
            text = emit()
            seed = "" if starts_chapter else _tail(text, overlap)  # no overlap across chapters
            buffer = [(buffer[-1][1], buffer[-1][1], seed)] if seed else []
            buffer_tokens = count_tokens(seed) if seed else 0
            has_new_content = False

        if not has_new_content or starts_chapter:
            buffer_chapter = para_chapter
        buffer.append((first, last, para))
        buffer_tokens += para_tokens
        has_new_content = True

    if has_new_content:
        emit()

    return chunks
