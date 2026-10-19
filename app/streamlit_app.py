"""
Streamlit interface for the knowledgebase pipeline.

    streamlit run app/streamlit_app.py

Tabs:
- Ingest:       upload files (or use the sample corpus) and watch each pipeline stage run
- Library:      everything in the knowledgebase, with per-document details
- Search & Ask: semantic search and cited answers
- Export:       download the knowledgebase as a JSONL dataset
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from kb.config import settings
from kb.extract import SUPPORTED_TYPES

SAMPLES_DIR = ROOT / "samples"
UPLOADS_DIR = SAMPLES_DIR / "uploads"
STAGES = ["hash", "extract", "clean", "enrich", "chunk", "embed", "store"]
STAGE_LABELS = {
    "hash": "Fingerprinting file",
    "extract": "Extracting text (OCR on scanned pages)",
    "clean": "Cleaning text",
    "enrich": "Extracting metadata and labels",
    "chunk": "Chunking",
    "embed": "Creating embeddings",
    "store": "Saving to the database",
}

st.set_page_config(page_title="LLM Knowledgebase", page_icon="📚", layout="wide")


# =============================================================================
# SETUP
# =============================================================================

@st.cache_resource
def connect() -> str | None:
    """Create tables once per server start. Returns an error message on failure."""
    from kb.store import init_database
    try:
        init_database()
        return None
    except Exception as e:
        return str(e)


def sample_files() -> list[Path]:
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in SUPPORTED_TYPES)


def run_ingest(files: list[Path], force: bool):
    """Ingest files one by one with live stage progress."""
    from kb.pipeline import ingest_file

    reports = []
    for path in files:
        with st.status(f"**{path.name}**", expanded=True) as status:
            def on_stage(stage, seconds):
                st.write(f"✓ {STAGE_LABELS.get(stage, stage)} ({seconds:.1f}s)")

            try:
                report = ingest_file(path, force=force, on_stage=on_stage)
            except Exception as e:
                report = {"file": path.name, "status": f"failed: {e}"}

            if report["status"] == "ingested":
                status.update(label=f"**{path.name}**: ingested {report['chunks']} chunks", state="complete",
                              expanded=False)
            elif report["status"].startswith("skipped"):
                status.update(label=f"**{path.name}**: already in the knowledgebase, skipped", state="complete",
                              expanded=False)
            else:
                status.update(label=f"**{path.name}**: {report['status']}", state="error")
        reports.append(report)
    st.session_state["last_reports"] = reports


def show_report(report: dict):
    """One file's run report."""
    if report["status"] != "ingested":
        st.write(f"**{report['file']}**: {report['status']}")
        return

    with st.container(border=True):
        st.markdown(f"#### {report['title'] or report['file']}")
        st.caption(f"{report['file']} · by {report['author'] or 'unknown author'}")

        cols = st.columns(5)
        cols[0].metric("Pages", report["pages"])
        cols[1].metric("Scanned pages (OCR)", report["ocr_pages"])
        cols[2].metric("OCR confidence", f"{report['ocr_mean_confidence']}%" if report["ocr_mean_confidence"] else "n/a")
        cols[3].metric("Chunks", report["chunks"])
        cols[4].metric("Tokens", f"{report['tokens']:,}")

        timings = report["seconds"]
        st.write("**Seconds per stage**")
        st.dataframe(
            [{"stage": s, "seconds": timings.get(s, 0.0)} for s in STAGES],
            hide_index=True, width="content",
        )

        labels = report.get("labels") or {}
        if labels.get("labeled_by") == "llm":
            st.write(f"**Summary:** {labels.get('summary')}")
            st.write("**Topics:** " + ", ".join(labels.get("topics") or []))
        else:
            st.write("**Keywords:** " + ", ".join(labels.get("keywords") or []))

        for w in report.get("warnings") or []:
            st.warning(w)


# =============================================================================
# PAGE
# =============================================================================

st.title("📚 LLM Knowledgebase")
st.caption("Turn PDFs, scanned books and photos of pages into a searchable, cited knowledgebase.")

db_error = connect()

with st.sidebar:
    st.header("Status")
    if db_error:
        st.error(f"Database not reachable:\n\n{db_error}")
    else:
        st.success("Database connected")
    st.write(f"**Embeddings:** {settings.embedding_provider} ({settings.embedding_dim}-dim)")
    st.write(f"**AI labeling and answers:** {'on (Gemini)' if settings.llm_enabled else 'off (no Gemini key)'}")
    st.caption("Set GEMINI_API_KEY in .env and restart to turn AI features on.")

    st.divider()
    st.header("Sample documents")
    st.caption("Builds a digital PDF, a scanned PDF and a scanned photo from The Time Machine (public domain).")
    if st.button("Generate samples", width="stretch"):
        with st.spinner("Downloading the book and building samples..."):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_samples.py")],
                                    capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Samples ready")
        else:
            st.error(result.stderr[-1000:])

if db_error:
    st.stop()

tab_ingest, tab_library, tab_ask, tab_export = st.tabs(["⬆️ Ingest", "🗂️ Library", "🔎 Search & Ask", "⬇️ Export"])

# ---------------------------------------------------------------- Ingest
with tab_ingest:
    left, right = st.columns(2)

    with left:
        st.subheader("Upload your own files")
        uploads = st.file_uploader(
            "PDFs, scans (PNG, JPG, TIFF) or text files",
            type=[t.lstrip(".") for t in sorted(SUPPORTED_TYPES)],
            accept_multiple_files=True,
        )
        force_upload = st.checkbox("Re-process files that are already in the knowledgebase", key="force_upload")
        if st.button("Ingest uploaded files", type="primary", disabled=not uploads):
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            paths = []
            for f in uploads:
                path = UPLOADS_DIR / Path(f.name).name
                path.write_bytes(f.getbuffer())
                paths.append(path)
            run_ingest(paths, force_upload)

    with right:
        st.subheader("Or use the sample documents")
        samples = sample_files()
        if samples:
            st.write("\n".join(f"- {p.name}" for p in samples))
        else:
            st.info("No samples yet. Click **Generate samples** in the sidebar.")
        force_samples = st.checkbox("Re-process files that are already in the knowledgebase", key="force_samples")
        if st.button("Ingest sample documents", disabled=not samples):
            run_ingest(samples, force_samples)

    if st.session_state.get("last_reports"):
        st.divider()
        st.subheader("Last run")
        for report in st.session_state["last_reports"]:
            show_report(report)

# ---------------------------------------------------------------- Library
with tab_library:
    from kb.store import delete_document, get_document, get_stats

    rows = get_stats()
    if not rows:
        st.info("The knowledgebase is empty. Ingest some documents first.")
    else:
        total_pages = sum(r["page_count"] for r in rows)
        cols = st.columns(4)
        cols[0].metric("Documents", len(rows))
        cols[1].metric("Pages", total_pages)
        cols[2].metric("Scanned pages", sum(r["ocr_pages"] for r in rows))
        cols[3].metric("Chunks", sum(r["chunks"] for r in rows))

        st.dataframe(
            [{
                "id": r["id"],
                "file": r["source_file"],
                "title": r["title"],
                "author": r["author"],
                "pages": r["page_count"],
                "scanned pages": r["ocr_pages"],
                "OCR confidence": r["ocr_mean_confidence"],
                "chunks": r["chunks"],
                "tokens": r["tokens"],
            } for r in rows],
            hide_index=True, width="stretch",
        )

        names = {r["id"]: f"{r['title']} ({r['source_file']})" for r in rows}
        # Options are ids, so a document re-ingested under a new id never leaves a stale selection
        choice = st.selectbox("Document details", list(names), format_func=lambda i: names.get(i, f"document {i}"),
                              key="doc-" + "-".join(map(str, names)))
        doc = get_document(choice) if choice is not None else None
        if doc:
            meta = doc["metadata"] or {}
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Chapters**")
                chapters = meta.get("chapters") or []
                if chapters:
                    st.dataframe(chapters, hide_index=True, width="stretch")
                else:
                    st.write("None detected")
            with c2:
                labels = meta.get("labels") or {}
                st.write("**Labels**" + (" (AI)" if labels.get("labeled_by") == "llm" else " (keywords)"))
                st.json(labels)
            if st.button("Remove this document from the knowledgebase", key=f"del-{doc['id']}"):
                delete_document(doc["id"])
                st.rerun()

# ---------------------------------------------------------------- Search & Ask
with tab_ask:
    from kb.qa import answer, cite, search

    query = st.text_input("Ask a question or search the knowledgebase",
                          placeholder="e.g. What happened to Weena?")
    k = st.slider("Passages to retrieve", 1, 10, settings.top_k)
    b1, b2, _ = st.columns([1, 1.4, 9])
    ask_clicked = b1.button("Ask", type="primary", disabled=not query)
    search_clicked = b2.button("Search only", disabled=not query)

    if ask_clicked:
        with st.spinner("Searching and answering..."):
            result = answer(query, k)
        st.subheader("Answer")
        if result["mode"] == "extractive":
            st.caption("AI answers are off, so this shows the most relevant passage. Add a Gemini key to get written answers.")
        st.markdown(result["answer"])
        st.subheader("Sources")
        for s in result["sources"]:
            st.write(f"{s['ref']}  ·  similarity {s['similarity']}")

    if search_clicked:
        with st.spinner("Searching..."):
            results = search(query, k)
        if not results:
            st.info("Nothing found. Is the knowledgebase empty?")
        for i, c in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"**[{i}] {cite(c)}**  ·  similarity {float(c['similarity']):.3f}")
                st.write(c["chunk_text"])

# ---------------------------------------------------------------- Export
with tab_export:
    from kb.export import records, to_jsonl

    st.subheader("Export the knowledgebase as a dataset")
    st.write("One JSON record per chunk with its text, source, title, author, chapter, pages, "
             "token count and labels. Useful for fine-tuning data prep, evaluation sets, or loading into another system.")

    if st.button("Prepare export", type="primary"):
        rows = list(records())
        st.session_state["export"] = (to_jsonl(rows), rows[:3], len(rows))

    if st.session_state.get("export"):
        data, preview, count = st.session_state["export"]
        st.success(f"{count} chunks ready")
        st.download_button("Download knowledgebase.jsonl", data, file_name="knowledgebase.jsonl",
                           mime="application/jsonl")
        st.write("**Preview**")
        st.json(preview, expanded=False)
