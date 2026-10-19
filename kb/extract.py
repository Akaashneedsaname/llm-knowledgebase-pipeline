"""
Page-level text extraction.

Strategy per page:
- Digital PDF page with a usable text layer -> read the text layer (fast, exact)
- Scanned page (no/too little text) -> render to an image and OCR it with Tesseract
- Image files (PNG/JPG/TIFF scans) -> OCR
- Plain text files -> read as-is
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf
import pytesseract
from PIL import Image

from kb.config import settings

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_TYPES = {".pdf"} | SUPPORTED_IMAGES | SUPPORTED_TEXT


@dataclass
class Page:
    """One extracted page."""
    number: int
    text: str
    method: str                      # "text-layer" | "ocr" | "plain-text"
    ocr_confidence: Optional[float] = None


def ocr_image(image: Image.Image) -> tuple[str, float]:
    """
    OCR an image with Tesseract, keeping paragraph breaks.

    Tesseract's own paragraph grouping is unreliable on scans, so paragraphs
    are rebuilt from line geometry: a vertical gap clearly larger than normal
    line spacing starts a new paragraph.

    Returns:
        (text, mean word confidence 0-100)
    """
    data = pytesseract.image_to_data(
        image.convert("L"), lang=settings.ocr_lang, output_type=pytesseract.Output.DICT
    )

    # Group words into lines: key -> {"words", "top", "bottom"}
    lines: dict[tuple, dict] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not word.strip() or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        top, bottom = data["top"][i], data["top"][i] + data["height"][i]
        line = lines.setdefault(key, {"words": [], "top": top, "bottom": bottom})
        line["words"].append(word)
        line["top"], line["bottom"] = min(line["top"], top), max(line["bottom"], bottom)
        confidences.append(conf)

    ordered = list(lines.values())
    if not ordered:
        return "", 0.0

    heights = sorted(l["bottom"] - l["top"] for l in ordered)
    line_height = heights[len(heights) // 2] or 1

    out = [" ".join(ordered[0]["words"])]
    for prev, line in zip(ordered, ordered[1:]):
        gap = line["top"] - prev["bottom"]
        if gap > 0.8 * line_height or gap < -line_height:  # paragraph gap, or a new column
            out.append("")
        out.append(" ".join(line["words"]))

    mean_conf = sum(confidences) / len(confidences)
    return "\n".join(out).strip(), round(mean_conf, 1)


def render_page(page: pymupdf.Page, dpi: int = None) -> Image.Image:
    """Render a PDF page to a grayscale PIL image."""
    pix = page.get_pixmap(dpi=dpi or settings.ocr_dpi, colorspace=pymupdf.csGRAY)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _text_layer(page: pymupdf.Page) -> str:
    """Read a page's text layer, one block per paragraph."""
    blocks = page.get_text("blocks")
    return "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())


def extract_pages(path: Path) -> tuple[list[Page], dict]:
    """
    Extract text from a document, page by page.

    Args:
        path: Path to a PDF, image or text file

    Returns:
        (pages, embedded file metadata such as PDF title/author)
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = []
        with pymupdf.open(path) as doc:
            embedded = {k: v.strip() for k, v in (doc.metadata or {}).items() if v and v.strip()}
            for number, page in enumerate(doc, start=1):
                text = _text_layer(page)
                if len(text) >= settings.ocr_min_chars:
                    pages.append(Page(number, text, "text-layer"))
                else:
                    ocr_text, conf = ocr_image(render_page(page))
                    pages.append(Page(number, ocr_text, "ocr", conf))
        return pages, embedded

    if suffix in SUPPORTED_IMAGES:
        with Image.open(path) as img:
            text, conf = ocr_image(img)
        return [Page(1, text, "ocr", conf)], {}

    if suffix in SUPPORTED_TEXT:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Form feeds mark page breaks in many text exports
        parts = [p for p in text.split("\f") if p.strip()] or [text]
        return [Page(i, p, "plain-text") for i, p in enumerate(parts, start=1)], {}

    raise ValueError(f"Unsupported file type: {suffix}")
