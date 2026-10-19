# LLM Knowledgebase Pipeline

Turn piles of documents (digital PDFs, scanned books, photographed pages) into a clean, searchable knowledgebase for LLMs.

The pipeline OCRs scanned pages, strips print artifacts, extracts metadata (title, author, chapters, topic labels), chunks the text along paragraph and chapter boundaries, embeds it, and stores everything in PostgreSQL with pgvector. You can then search it semantically, ask questions with cited answers, or export it as a JSONL dataset for training and evaluation.

It is orchestrated as a **LangGraph** state graph with per-stage timing and data-quality warnings, it is idempotent (unchanged files are skipped), and it runs fully offline by default. A Gemini API key adds LLM labeling and generated answers.

> Adapted from the document-ingestion side of a production multi-source RAG system I built as an AI Engineer at US Farm Data, extended here with OCR, metadata extraction and dataset export.

## Pipeline

```mermaid
flowchart LR
    A[File] --> H[hash]
    H -- already ingested --> S[skip]
    H -- new / --force --> E[extract]
    E --> C[clean] --> M[enrich] --> K[chunk] --> V[embed] --> P[(store<br/>Postgres + pgvector)]
    P --> Q[search / ask]
    P --> X[export JSONL]
```

| Stage | What it does |
|---|---|
| **hash** | SHA-256 of the file bytes. Files already in the knowledgebase are skipped, so a folder can be re-run safely. |
| **extract** | Reads each PDF page's text layer. Pages with no usable text (scans) are rendered at 300 DPI and OCR'd with Tesseract. Image files are OCR'd directly. Paragraphs are rebuilt from OCR line geometry, since Tesseract's own paragraph grouping is unreliable on scans. Records OCR confidence per page. |
| **clean** | Removes running headers and footers (short lines repeated at the top or bottom of most pages), bare page numbers and soft hyphens; rejoins words hyphenated across lines; unwraps hard-wrapped lines into paragraphs. |
| **enrich** | Title and author from embedded PDF fields, Project Gutenberg headers or the title page. Chapter detection. Topic labels from the LLM (summary, topics, genre, language) or, offline, frequency-based keywords. |
| **chunk** | Paragraph-aware token chunking (default 400 tokens, 60 overlap). Chapter headings always start a new chunk, and sentences split across a page break are rejoined. Each chunk keeps its page range and chapter for citations. |
| **embed** | Pluggable: local `BAAI/bge-small-en-v1.5` via fastembed (384-dim, CPU, no key) or Gemini `gemini-embedding-001` (768-dim). |
| **store** | Document and chunks written in one transaction. HNSW cosine index. One chunk table per vector size, so both embedding providers can coexist. |

Every run prints a report per file: pages, OCR pages, mean OCR confidence, chapters, labels, chunks, tokens, seconds per stage and warnings (for example, low OCR confidence on specific pages).

## Quick start

Requires Docker.

```bash
# 1. Build and start Postgres + pgvector
docker compose up -d db
docker compose build

# 2. Create sample documents from a public-domain book (The Time Machine, H. G. Wells):
#    a digital PDF, an image-only "scanned" PDF with skew and noise, and a scanned JPG page
docker compose run --rm app python scripts/make_samples.py

# 3. Ingest
docker compose run --rm app python cli.py ingest samples/

# 4. Use it
docker compose run --rm app python cli.py stats
docker compose run --rm app python cli.py search "What are the Morlocks?"
docker compose run --rm app python cli.py ask "How does the Time Traveller explain the fourth dimension?"
docker compose run --rm app python cli.py export output/knowledgebase.jsonl

# Tests
docker compose run --rm --no-deps app python -m pytest -q
```

To enable LLM labeling and generated answers, copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Set `EMBEDDING_PROVIDER=gemini` to switch to Gemini embeddings.

## Example run

Three sample files: 55 pages, 20 of them scanned. Laptop CPU, local embeddings.

```
$ python cli.py stats
 id  file                             title                        pages  ocr  conf chunks  tokens
  5  time_machine_page_scan.jpg       The Time Machine                 1    1    96      2     709
  6  time_machine_part1.pdf           The Time Machine                35    0     -     86   26323
  7  time_machine_part2_scan.pdf      The Time Machine: An Inventi    19   19    95     50   14915
```

The scanned PDF has no text layer and no embedded metadata, yet title and author are still recovered from the OCR'd title page:

```json
{
  "file": "time_machine_part2_scan.pdf",
  "status": "ingested",
  "title": "The Time Machine: An Invention",
  "author": "H. G. Wells",
  "pages": 19,
  "ocr_pages": 19,
  "ocr_mean_confidence": 95.4,
  "chapters": 4,
  "chunks": 50,
  "seconds": {"hash": 0.12, "extract": 45.8, "clean": 0.01, "enrich": 0.0, "chunk": 0.02, "embed": 8.84, "store": 0.32},
  "warnings": []
}
```

OCR dominates the run time (about 2.4 s per page at 300 DPI), so the per-stage timings show exactly where to optimize, for example by parallelizing OCR across pages or lowering the DPI for clean scans.

Retrieval works across digital and scanned sources, with citations down to the chapter and page:

```
$ python cli.py ask "What happened to Weena?"
Sources (extractive mode):
  [1] The Time Machine, Chapter VIII: Explanation, p. 35  (similarity 0.652)
  [2] The Time Machine: An Invention, Chapter XII: In the Darkness, pp. 15-16  (similarity 0.635)
  [3] The Time Machine, Chapter VIII: Explanation, pp. 30-31  (similarity 0.634)
```

Each line of the JSONL export is a self-contained record:

```json
{"id": "6-7", "text": "...", "source": "time_machine_part1.pdf", "title": "The Time Machine",
 "author": "H. G. Wells", "chapter": "Chapter II: The Machine", "pages": [7, 7], "tokens": 320,
 "labels": {"keywords": ["time", "little", "machine", "..."], "labeled_by": "keywords"}}
```

## Design decisions

- **OCR only where needed.** Each PDF page is checked individually, so a mostly digital document with a few scanned inserts is OCR'd only on those pages. That's faster and more accurate than OCR'ing everything.
- **Paragraphs from geometry.** On scans, Tesseract often starts "new paragraphs" mid-sentence. Rebuilding paragraphs from vertical line gaps fixed broken chunks in the sample scans.
- **Paragraph-aware chunks with citations.** Splitting on raw token windows cuts sentences and mixes chapters. Chunks built from whole paragraphs, starting at chapter headings, retrieve better and cite precisely.
- **Idempotent by content hash.** Re-running on a growing archive only processes new or changed files. `--force` re-ingests on purpose.
- **Offline first, cloud optional.** The full pipeline runs with no API key (local embeddings, keyword labels, extractive answers). An LLM adds richer labels and generated answers when a key is set, and every LLM call degrades gracefully if it fails.
- **Observable.** Per-stage timings and quality warnings come back on every run, so slow stages and bad scans are visible without digging through logs.
- **HNSW over IVFFlat.** HNSW needs no training step, so search quality is good from the first document. It suits a knowledgebase that grows incrementally.

## Project layout

```
kb/
  config.py     settings from environment variables
  extract.py    text layer + Tesseract OCR, paragraph rebuilding
  clean.py      header/footer removal, de-hyphenation, unwrapping
  metadata.py   title/author/chapters, LLM or keyword labels
  chunk.py      paragraph- and chapter-aware token chunking
  embed.py      local (fastembed) and Gemini embedders
  llm.py        optional Gemini wrapper
  store.py      Postgres + pgvector schema, writes, similarity search
  pipeline.py   LangGraph orchestration, timing, run reports
  qa.py         retrieval and cited answers
  export.py     JSONL dataset export
cli.py          ingest / search / ask / export / stats
scripts/make_samples.py   builds the sample corpus
tests/          unit tests for cleaning, chunking and metadata
```

## Tech stack

Python 3.12 · LangGraph · PyMuPDF · Tesseract (pytesseract) · tiktoken · fastembed · Google Gemini API · PostgreSQL 16 + pgvector · Docker Compose · pytest
