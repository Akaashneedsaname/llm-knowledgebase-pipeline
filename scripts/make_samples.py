"""
Build a realistic sample corpus from a public-domain book.

Downloads H. G. Wells' "The Time Machine" from Project Gutenberg and produces:

- samples/time_machine_part1.pdf       digital PDF (text layer, running header,
                                        page numbers, embedded title/author)
- samples/time_machine_part2_scan.pdf  image-only "scanned" PDF (slight skew and
                                        noise, no text layer, no metadata)
- samples/time_machine_page_scan.jpg   a single scanned page as an image

Together they exercise every extraction path: text layer, OCR on scanned
PDF pages, and OCR on image files.

    python scripts/make_samples.py
"""

import io
import random
import re
import textwrap
import urllib.request
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter

SOURCE_URL = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
OUT_DIR = Path(__file__).resolve().parent.parent / "samples"

PAGE_W, PAGE_H, MARGIN = 612, 792, 72
FONT_SIZE, LINE_HEIGHT, WRAP = 11, 14.5, 88


def download_book() -> str:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as r:
        text = r.read().decode("utf-8")
    start = text.index("*** START OF")
    end = text.index("*** END OF")
    return text[text.index("\n", start):end]


def split_chapters(body: str) -> list[tuple[str, str, str]]:
    """Return (roman numeral, title, text) for each chapter."""
    body = (body.replace("\r", "").replace("—", " - ").replace("_", "")
            .replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"'))
    parts = re.split(r"\n\s*([IVX]+)\.\s*\n\s*([^\n]+)\n", body)
    chapters = []
    for i in range(1, len(parts) - 2, 3):
        numeral, title, text = parts[i], parts[i + 1].strip(), parts[i + 2]
        paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
        chapters.append((numeral, title, "\n\n".join(paras)))
    return chapters


def typeset(chapters, title_page: list[str], header: str, set_metadata: bool) -> pymupdf.Document:
    """Lay chapters out as a printed book: running header, page numbers, chapter breaks."""
    doc = pymupdf.open()

    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = 260
    for i, line in enumerate(title_page):
        size = 26 if i == 0 else 14
        width = pymupdf.get_text_length(line, fontname="tibo" if i == 0 else "tiro", fontsize=size)
        page.insert_text(((PAGE_W - width) / 2, y), line, fontname="tibo" if i == 0 else "tiro", fontsize=size)
        y += size * 2.2

    for numeral, title, text in chapters:
        page = None
        y = 0

        def new_page():
            nonlocal page, y
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.insert_text((MARGIN, 45), header, fontname="tiro", fontsize=9)
            page.insert_text((PAGE_W / 2 - 14, PAGE_H - 36), f"Page {len(doc)}", fontname="tiro", fontsize=9)
            y = MARGIN + 10

        new_page()
        page.insert_text((MARGIN, y + 8), f"Chapter {numeral}: {title}", fontname="tibo", fontsize=15)
        y += 40

        for para in text.split("\n\n"):
            for line in textwrap.wrap(para, WRAP):
                if y > PAGE_H - MARGIN:
                    new_page()
                page.insert_text((MARGIN, y), line, fontname="tiro", fontsize=FONT_SIZE)
                y += LINE_HEIGHT
            y += LINE_HEIGHT * 0.9  # paragraph gap

    if set_metadata:
        doc.set_metadata({"title": "The Time Machine", "author": "H. G. Wells", "subject": "Fiction"})
    return doc


def scan(page: pymupdf.Page, dpi: int = 200) -> Image.Image:
    """Simulate a flatbed scan: grayscale raster, slight skew, light noise and blur."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = img.rotate(random.uniform(-0.6, 0.6), expand=False, fillcolor=255)
    noise = Image.effect_noise(img.size, 18).convert("L")
    img = Image.blend(img, noise, 0.06).filter(ImageFilter.GaussianBlur(0.4))
    return img


def image_only_pdf(images: list[Image.Image]) -> pymupdf.Document:
    doc = pymupdf.open()
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)  # scanners usually produce JPEG pages
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, stream=buf.getvalue())
    return doc


def main():
    random.seed(7)
    OUT_DIR.mkdir(exist_ok=True)

    print("Downloading The Time Machine from Project Gutenberg...")
    chapters = split_chapters(download_book())
    print(f"Found {len(chapters)} chapters")

    title_page = ["THE TIME MACHINE", "An Invention", "by H. G. Wells"]

    part1 = typeset(chapters[:8], title_page, "THE TIME MACHINE", set_metadata=True)
    part1.save(OUT_DIR / "time_machine_part1.pdf")
    print(f"Wrote time_machine_part1.pdf ({len(part1)} pages, text layer)")

    printed = typeset(chapters[8:12], title_page + ["(Chapters IX - XII)"], "THE TIME MACHINE", set_metadata=False)
    scanned = image_only_pdf([scan(p) for p in printed])
    scanned.save(OUT_DIR / "time_machine_part2_scan.pdf", deflate=True)
    print(f"Wrote time_machine_part2_scan.pdf ({len(scanned)} pages, image only)")

    single = typeset(chapters[12:13], title_page, "THE TIME MACHINE", set_metadata=False)
    scan(single[1]).save(OUT_DIR / "time_machine_page_scan.jpg", quality=80)
    print("Wrote time_machine_page_scan.jpg (one scanned page)")


if __name__ == "__main__":
    main()
