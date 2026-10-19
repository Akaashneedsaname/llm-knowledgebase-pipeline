"""
Streamlit interface for the knowledgebase pipeline.

    streamlit run app/streamlit_app.py

Tabs:
- Add documents: upload files (or use the sample set) and watch each pipeline stage run
- Library:       everything in the knowledgebase, with per-document details
- Ask:           cited answers and matching passages
- Export:        download the knowledgebase as a JSONL dataset
"""

import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.theme import CSS, catalog_card, masthead, status_row, text_to_html
from kb.config import settings
from kb.extract import SUPPORTED_TYPES

SAMPLES_DIR = ROOT / "samples"
UPLOADS_DIR = SAMPLES_DIR / "uploads"
STAGE_LABELS = {
    "hash": "Fingerprinted the file",
    "extract": "Extracted the text (OCR on scanned pages)",
    "clean": "Cleaned the text",
    "enrich": "Found the title, author, chapters and labels",
    "chunk": "Split it into chunks",
    "embed": "Created embeddings",
    "store": "Saved to the library",
}
STAGE_SHORT = {
    "hash": "fingerprinting", "extract": "text extraction", "clean": "cleaning", "enrich": "metadata",
    "chunk": "chunking", "embed": "embeddings", "store": "saving",
}

st.set_page_config(page_title="LLM Knowledgebase", page_icon=":material/menu_book:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# HELPERS
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


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def scanned_phrase(pages: int, scanned: int) -> str:
    if scanned == 0:
        return "none scanned"
    if scanned == pages:
        return "all scanned"
    return f"{scanned} scanned"


def timing_sentence(seconds: dict) -> str:
    """'Took 55.2 s: text extraction 45.8 s, embeddings 8.8 s, the rest under a second.'"""
    total = sum(seconds.values())
    slow = sorted(((s, v) for s, v in seconds.items() if v >= 1), key=lambda x: -x[1])
    if not slow:
        return f"Took {total:.1f} s."
    parts = [f"{STAGE_SHORT.get(s, s)} {v:.1f} s" for s, v in slow]
    rest = ", the rest under a second" if len(slow) < len(seconds) else ""
    return f"Took {total:.1f} s: " + ", ".join(parts) + rest + "."


def run_ingest(files: list[Path], force: bool):
    """Add files one by one with live stage progress."""
    from kb.pipeline import ingest_file

    reports = []
    for path in files:
        with st.status(path.name, expanded=True) as status:
            def on_stage(stage, seconds):
                st.write(f"{STAGE_LABELS.get(stage, stage)} ({seconds:.1f} s)")

            try:
                report = ingest_file(path, force=force, on_stage=on_stage)
            except Exception as e:
                report = {"file": path.name, "status": f"failed: {e}"}

            if report["status"] == "ingested":
                status.update(label=f"{path.name}: added, {plural(report['chunks'], 'chunk')}",
                              state="complete", expanded=False)
            elif report["status"].startswith("skipped"):
                status.update(label=f"{path.name}: already in the library, skipped", state="complete",
                              expanded=False)
            else:
                status.update(label=f"{path.name}: {report['status']}", state="error")
        reports.append(report)
    st.session_state["last_reports"] = reports


def show_report(report: dict):
    """One file's run report, written as sentences."""
    if report["status"] != "ingested":
        html(f'<p class="kb-note"><b>{escape(report["file"])}</b>: {escape(report["status"])}</p>')
        return

    pages, scanned = report["pages"], report["ocr_pages"]
    facts = f"{plural(pages, 'page')}, {scanned_phrase(pages, scanned)}"
    if report["ocr_mean_confidence"] is not None:
        facts += f", read with {report['ocr_mean_confidence']:.0f}% OCR confidence"
    facts += f". {plural(report['chunks'], 'chunk')}, {plural(report['tokens'], 'token')}"
    if report["chapters"]:
        facts += f", {plural(report['chapters'], 'chapter')}"
    facts += "."

    labels = report.get("labels") or {}
    if labels.get("labeled_by") == "llm":
        label_html = (f'<b>Summary</b> {escape(labels.get("summary") or "")}<br>'
                      f'<b>Topics</b> {escape(", ".join(labels.get("topics") or []))}')
    else:
        label_html = f'<b>Keywords</b> {escape(", ".join(labels.get("keywords") or []))}'

    author = f"by {report['author']}" if report["author"] else "Author not found"
    # One line of HTML: indented or blank lines would be parsed as Markdown
    html(
        '<div class="kb-report">'
        f'<h4>{escape(report["title"] or report["file"])}</h4>'
        f'<p class="kb-byline">{escape(author)}, from {escape(report["file"])}</p>'
        f'<p class="kb-facts">{escape(facts)}</p>'
        f'<p class="kb-meta">{label_html}</p>'
        f'<p class="kb-meta">{escape(timing_sentence(report["seconds"]))}</p>'
        '</div>'
    )
    for w in report.get("warnings") or []:
        st.warning(f"{report['file']}: {w}")


# =============================================================================
# PAGE
# =============================================================================

html(masthead("LLM Knowledgebase",
              "Turn PDFs, scanned books and photos of pages into a knowledgebase you can search and cite."))

db_error = connect()

with st.sidebar:
    st.subheader("Status")
    if db_error:
        html(status_row("Database not reachable", "bad"))
        st.error(db_error)
    else:
        html(status_row("Database connected", "on"))
    html(status_row("Embeddings: local model" if settings.embedding_provider == "local"
                    else "Embeddings: Gemini", "on"))
    if settings.llm_enabled:
        html(status_row("AI labels and answers on (Gemini)", "on"))
        usage_box = st.empty()  # filled at the end of the run, after any AI calls
    else:
        html(status_row("AI labels and answers off", "off"))
        html('<p class="kb-side-note">Add GEMINI_API_KEY to .env and restart to turn them on.</p>')
        usage_box = None

    st.divider()
    st.subheader("Sample documents")
    html('<p class="kb-side-note">A digital PDF, a scanned PDF and a photo of a page, made from '
         'H. G. Wells\' The Time Machine (public domain).</p>')
    if st.button("Create sample documents", width="stretch"):
        with st.spinner("Downloading the book and building the samples..."):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_samples.py")],
                                    capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Sample documents created")
        else:
            st.error("Couldn't create the samples. The book is downloaded from Project Gutenberg, "
                     "so check the internet connection.\n\n" + result.stderr[-600:])

if db_error:
    st.stop()

tab_add, tab_library, tab_ask, tab_export = st.tabs(["Add documents", "Library", "Ask", "Export"])

# ---------------------------------------------------------------- Add documents
with tab_add:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Your files")
        uploads = st.file_uploader(
            "PDFs, scans and photos of pages (PNG, JPG, TIFF), or text files",
            type=[t.lstrip(".") for t in sorted(SUPPORTED_TYPES)],
            accept_multiple_files=True,
        )
        force_upload = st.checkbox("Process again even if already in the library", key="force_upload")
        if st.button("Add uploaded files", type="primary", disabled=not uploads):
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            paths = []
            for f in uploads:
                path = UPLOADS_DIR / Path(f.name).name
                path.write_bytes(f.getbuffer())
                paths.append(path)
            run_ingest(paths, force_upload)

    with right:
        st.subheader("Sample documents")
        samples = sample_files()
        if samples:
            html('<ul class="kb-files">' + "".join(f"<li>{escape(p.name)}</li>" for p in samples) + "</ul>")
        else:
            html('<p class="kb-note">No samples yet. Use <b>Create sample documents</b> in the sidebar.</p>')
        force_samples = st.checkbox("Process again even if already in the library", key="force_samples")
        if st.button("Add sample documents", disabled=not samples):
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
        html('<p class="kb-lede">Nothing here yet.</p>'
             '<p class="kb-note">Add files in the Add documents tab, or create the sample set from the sidebar.</p>')
    else:
        pages = sum(r["page_count"] for r in rows)
        scanned = sum(r["ocr_pages"] for r in rows)
        chunks = sum(r["chunks"] for r in rows)
        html(f'<p class="kb-lede">{plural(len(rows), "document")}, {plural(pages, "page")} '
             f'({scanned_phrase(pages, scanned)}), split into {plural(chunks, "searchable chunk")}.</p>')

        st.dataframe(
            [{
                "Title": r["title"],
                "Author": r["author"] or "",
                "File": r["source_file"],
                "Pages": r["page_count"],
                "Scanned": r["ocr_pages"],
                "OCR confidence": f"{r['ocr_mean_confidence']:.0f}%" if r["ocr_mean_confidence"] is not None else "",
                "Chunks": r["chunks"],
                "Tokens": r["tokens"],
            } for r in rows],
            hide_index=True, width="stretch",
        )

        names = {r["id"]: f"{r['title']} ({r['source_file']})" for r in rows}
        # Options are ids, so a document re-added under a new id never leaves a stale selection
        choice = st.selectbox("Document details", list(names), format_func=lambda i: names.get(i, f"document {i}"),
                              key="doc-" + "-".join(map(str, names)))
        doc = get_document(choice) if choice is not None else None
        if doc:
            meta = doc["metadata"] or {}
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Chapters**")
                chapters = meta.get("chapters") or []
                if chapters:
                    st.dataframe([{"Starts on page": c["page"], "Heading": c["heading"]} for c in chapters],
                                 hide_index=True, width="stretch")
                else:
                    html('<p class="kb-note">No chapter headings found.</p>')
            with c2:
                labels = meta.get("labels") or {}
                if labels.get("labeled_by") == "llm":
                    st.markdown("**Labels from AI**")
                    html(f'<p class="kb-note">{escape(labels.get("summary") or "")}</p>'
                         f'<p class="kb-note"><b>Topics</b> {escape(", ".join(labels.get("topics") or []))}</p>')
                else:
                    st.markdown("**Keywords**")
                    html(f'<p class="kb-note">{escape(", ".join(labels.get("keywords") or []))}</p>')
            if st.button("Remove from library", key=f"del-{doc['id']}"):
                delete_document(doc["id"])
                st.rerun()

# ---------------------------------------------------------------- Ask
with tab_ask:
    from kb.qa import answer, cite, search

    query = st.text_input("Ask a question about your documents", placeholder="e.g. What happened to Weena?")
    k = st.slider("Passages to use", 1, 10, settings.top_k)
    b1, b2, _ = st.columns([1, 2, 6])
    ask_clicked = b1.button("Ask", type="primary", disabled=not query, width="stretch")
    search_clicked = b2.button("Show matching passages", disabled=not query, width="stretch")

    if ask_clicked:
        with st.spinner("Finding passages and writing the answer..."):
            result = answer(query, k)
        st.subheader("Answer")
        if result.get("cached"):
            html('<p class="kb-note">Answered from cache: same question, same passages, no new AI call.</p>')
        if result["mode"] == "extractive":
            from kb import llm as llm_module
            if not settings.llm_enabled:
                html('<p class="kb-note">AI answers are off, so this is the most relevant passage. '
                     'Add a Gemini key to get written answers.</p>')
            else:
                st.warning("The AI answer didn't come through, so this is the most relevant passage instead. "
                           f"Reason: {llm_module.last_error or 'the AI call limit was reached'}")
        html(f'<div class="kb-answer">{text_to_html(result["answer"])}</div>')
        if result["sources"]:
            st.subheader("Sources")
            for i, s in enumerate(result["sources"], start=1):
                ref = s["ref"].split("] ", 1)[-1]
                html(catalog_card(ref, rank=i, similarity=s["similarity"]))

    if search_clicked:
        with st.spinner("Searching..."):
            results = search(query, k)
        if not results:
            html('<p class="kb-note">No matches. The library may be empty: add documents first.</p>')
        else:
            st.subheader("Matching passages")
            for i, c in enumerate(results, start=1):
                html(catalog_card(cite(c), c["chunk_text"], rank=i, similarity=float(c["similarity"])))

# ---------------------------------------------------------------- Export
with tab_export:
    from kb.export import records, to_jsonl

    st.subheader("Download the knowledgebase as a dataset")
    html('<p class="kb-note" style="max-width:70ch">One JSON record per chunk: its text, source file, title, '
         'author, chapter, pages, token count and labels. Ready for fine-tuning data prep, evaluation sets, '
         'or loading into another system.</p>')

    if st.button("Build dataset", type="primary"):
        rows = list(records())
        st.session_state["export"] = (to_jsonl(rows), rows[:3], len(rows))

    if st.session_state.get("export"):
        data, preview, count = st.session_state["export"]
        html(f'<p class="kb-lede">{plural(count, "record")} ready.</p>')
        st.download_button("Download knowledgebase.jsonl", data, file_name="knowledgebase.jsonl",
                           mime="application/jsonl")
        st.markdown("**First records**")
        st.json(preview, expanded=1)

# ---------------------------------------------------------------- AI usage (sidebar)
if usage_box is not None:
    from kb.llm import usage_summary
    u = usage_summary()
    note = (f"AI usage since start: {plural(u['calls'], 'call')}, {plural(u['total_tokens'], 'token')}. "
            f"Limit {settings.llm_max_calls} calls.")
    if u["blocked"]:
        note += f" {plural(u['blocked'], 'call')} blocked by the limit."
    usage_box.markdown(f'<p class="kb-side-note">{escape(note)}</p>', unsafe_allow_html=True)
