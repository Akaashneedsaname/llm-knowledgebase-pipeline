"""
LangGraph orchestration of the ingestion pipeline.

Graph (one run per file):

    hash -> [already ingested?] --yes--> END (skipped)
                 |
                 no
                 v
    extract -> clean -> enrich -> chunk -> embed -> store -> END

Every node is timed, and the final state carries a run report (pages, OCR
pages, OCR confidence, chunks, tokens, seconds per stage) so slow or
low-quality documents are easy to spot.
"""

import hashlib
import operator
import time
from functools import wraps
from pathlib import Path
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from kb import store
from kb.chunk import Chunk, chunk_pages
from kb.clean import clean_pages
from kb.embed import get_embedder
from kb.extract import Page, extract_pages
from kb.metadata import build_metadata


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class PipelineState(TypedDict, total=False):
    """State passed between nodes."""
    # Input
    path: str
    force: bool

    # hash
    file_hash: str
    existing_id: Optional[int]
    skipped: bool

    # extract / clean / enrich / chunk / embed
    pages: list[Page]
    embedded_meta: dict
    cleaned: list[str]
    metadata: dict
    chunks: list[Chunk]
    vectors: list[list[float]]

    # store
    document_id: int

    # Observability
    timings: Annotated[dict, _merge]
    warnings: Annotated[list, operator.add]


def timed(name: str):
    """Record how long a node took in state['timings']."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: PipelineState) -> dict:
            start = time.perf_counter()
            update = fn(state)
            update["timings"] = {name: round(time.perf_counter() - start, 3)}
            return update
        return wrapper
    return decorator


# =============================================================================
# NODES
# =============================================================================

@timed("hash")
def hash_node(state: PipelineState) -> dict:
    """Fingerprint the file so unchanged files are never processed twice."""
    digest = hashlib.sha256(Path(state["path"]).read_bytes()).hexdigest()
    existing = store.find_document(digest)
    if existing and state.get("force"):
        store.delete_document(existing)
        existing = None
    return {"file_hash": digest, "existing_id": existing, "skipped": existing is not None}


@timed("extract")
def extract_node(state: PipelineState) -> dict:
    """Text layer for digital pages, OCR for scanned pages."""
    pages, embedded = extract_pages(Path(state["path"]))
    warnings = []
    low_conf = [p.number for p in pages if p.ocr_confidence is not None and p.ocr_confidence < 60]
    if low_conf:
        warnings.append(f"low OCR confidence on pages {low_conf}")
    empty = [p.number for p in pages if not p.text.strip()]
    if empty:
        warnings.append(f"no text recovered on pages {empty}")
    return {"pages": pages, "embedded_meta": embedded, "warnings": warnings}


@timed("clean")
def clean_node(state: PipelineState) -> dict:
    """Strip headers/footers/page numbers, fix hyphenation, unwrap lines."""
    return {"cleaned": clean_pages([p.text for p in state["pages"]])}


@timed("enrich")
def enrich_node(state: PipelineState) -> dict:
    """Title, author, chapters, and labels (LLM or keyword-based)."""
    pages = state["pages"]
    meta = build_metadata(
        state["cleaned"],
        state["embedded_meta"],
        Path(state["path"]).name,
        raw_first_page=pages[0].text if pages else None,
    )
    return {"metadata": meta}


@timed("chunk")
def chunk_node(state: PipelineState) -> dict:
    """Paragraph-aware token chunks with page and chapter tracking."""
    chunks = chunk_pages(state["cleaned"])
    return {"chunks": chunks, "warnings": [] if chunks else ["no chunks produced"]}


@timed("embed")
def embed_node(state: PipelineState) -> dict:
    """Embed every chunk with the configured provider."""
    texts = [c.text for c in state["chunks"]]
    return {"vectors": get_embedder().embed_documents(texts) if texts else []}


@timed("store")
def store_node(state: PipelineState) -> dict:
    """Write the document and its chunks to Postgres in one transaction."""
    pages = state["pages"]
    ocr_conf = [p.ocr_confidence for p in pages if p.ocr_confidence is not None]
    meta = state["metadata"]
    doc = {
        "file_hash": state["file_hash"],
        "source_file": Path(state["path"]).name,
        "title": meta.get("title"),
        "author": meta.get("author"),
        "language": meta.get("language"),
        "page_count": len(pages),
        "ocr_pages": sum(1 for p in pages if p.method == "ocr"),
        "ocr_mean_confidence": round(sum(ocr_conf) / len(ocr_conf), 1) if ocr_conf else None,
        "metadata": {**meta, "embedded": state["embedded_meta"]},
    }
    return {"document_id": store.save_document(doc, state["chunks"], state["vectors"])}


def _route_after_hash(state: PipelineState) -> str:
    return "skip" if state["skipped"] else "process"


# =============================================================================
# GRAPH
# =============================================================================

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("hash", hash_node)
    graph.add_node("extract", extract_node)
    graph.add_node("clean", clean_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)

    graph.add_edge(START, "hash")
    graph.add_conditional_edges("hash", _route_after_hash, {"skip": END, "process": "extract"})
    graph.add_edge("extract", "clean")
    graph.add_edge("clean", "enrich")
    graph.add_edge("enrich", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)

    return graph.compile()


_graph = None


def ingest_file(path: Path, force: bool = False, on_stage: Callable[[str, float], None] = None) -> dict:
    """
    Run the pipeline on one file.

    Args:
        path: File to ingest
        force: Re-ingest even if the file is already in the knowledgebase
        on_stage: Optional callback(stage_name, seconds) after each stage, for progress UIs

    Returns:
        A run report for the file
    """
    global _graph
    _graph = _graph or build_graph()

    initial = {"path": str(path), "force": force, "timings": {}, "warnings": []}
    state = initial
    for mode, chunk in _graph.stream(initial, stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
        elif on_stage:
            for node, update in chunk.items():
                on_stage(node, (update or {}).get("timings", {}).get(node, 0.0))

    if state.get("skipped"):
        return {"file": path.name, "status": "skipped (already ingested)", "document_id": state["existing_id"]}

    pages = state["pages"]
    chunks = state["chunks"]
    meta = state["metadata"]
    ocr_conf = [p.ocr_confidence for p in pages if p.ocr_confidence is not None]
    return {
        "file": path.name,
        "status": "ingested",
        "document_id": state["document_id"],
        "title": meta.get("title"),
        "author": meta.get("author"),
        "pages": len(pages),
        "ocr_pages": sum(1 for p in pages if p.method == "ocr"),
        "ocr_mean_confidence": round(sum(ocr_conf) / len(ocr_conf), 1) if ocr_conf else None,
        "chapters": len(meta.get("chapters", [])),
        "labels": meta.get("labels"),
        "chunks": len(chunks),
        "tokens": sum(c.token_count for c in chunks),
        "seconds": state["timings"],
        "warnings": state["warnings"],
    }
