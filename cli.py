"""
Command-line interface.

    python cli.py init-db
    python cli.py ingest samples/ [--force]
    python cli.py stats
    python cli.py search "time travel and the fourth dimension"
    python cli.py ask "What did the Time Traveller find in the year 802,701?"
    python cli.py export output/knowledgebase.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from kb.config import settings
from kb.extract import SUPPORTED_TYPES


def collect_files(paths: list[str]) -> list[Path]:
    """Expand files and folders into a sorted list of supported documents."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED_TYPES)
        elif p.suffix.lower() in SUPPORTED_TYPES:
            files.append(p)
        else:
            print(f"Skipping unsupported file: {p}")
    return files


def print_llm_usage():
    if settings.llm_enabled:
        from kb.llm import usage_summary
        u = usage_summary()
        print(f"\nAI usage: {u['calls']} calls, {u['total_tokens']:,} tokens"
              + (f", {u['blocked']} blocked by LLM_MAX_CALLS" if u["blocked"] else ""))


def cmd_init_db(args):
    from kb.store import init_database
    init_database()
    print(f"Database ready (chunks_{settings.embedding_dim}, provider: {settings.embedding_provider})")


def cmd_ingest(args):
    from kb.pipeline import ingest_file
    from kb.store import init_database

    init_database()
    files = collect_files(args.paths)
    if not files:
        print("No supported files found.")
        return

    print(f"Ingesting {len(files)} file(s) with {settings.embedding_provider} embeddings"
          f"{' + LLM labeling' if settings.llm_enabled else ''}\n")
    start = time.perf_counter()
    reports = []
    for f in files:
        try:
            report = ingest_file(f, force=args.force)
        except Exception as e:  # one bad file shouldn't stop a batch
            report = {"file": f.name, "status": f"failed: {e}"}
        reports.append(report)
        print(json.dumps(report, indent=2, default=str))

    ingested = [r for r in reports if r["status"] == "ingested"]
    print(f"\nDone in {time.perf_counter() - start:.1f}s: {len(ingested)} ingested, "
          f"{sum(1 for r in reports if r['status'].startswith('skipped'))} skipped, "
          f"{sum(1 for r in reports if r['status'].startswith('failed'))} failed, "
          f"{sum(r['chunks'] for r in ingested)} chunks")
    print_llm_usage()


def cmd_stats(args):
    from kb.store import get_stats
    rows = get_stats()
    if not rows:
        print("Knowledgebase is empty.")
        return
    print(f"{'id':>3}  {'file':<32} {'title':<28} {'pages':>5} {'ocr':>4} {'conf':>5} {'chunks':>6} {'tokens':>7}")
    for r in rows:
        conf = f"{r['ocr_mean_confidence']:.0f}" if r["ocr_mean_confidence"] is not None else "-"
        print(f"{r['id']:>3}  {r['source_file'][:32]:<32} {(r['title'] or '')[:28]:<28} "
              f"{r['page_count']:>5} {r['ocr_pages']:>4} {conf:>5} {r['chunks']:>6} {r['tokens']:>7}")


def cmd_search(args):
    from kb.qa import cite, search
    for i, c in enumerate(search(args.query, args.k), start=1):
        print(f"\n[{i}] {cite(c)}  (similarity {float(c['similarity']):.3f})")
        print(c["chunk_text"][:400] + ("..." if len(c["chunk_text"]) > 400 else ""))


def cmd_ask(args):
    from kb.qa import answer
    result = answer(args.query, args.k)
    print(f"\n{result['answer']}\n\nSources ({result['mode']} mode):")
    for s in result["sources"]:
        print(f"  {s['ref']}  (similarity {s['similarity']})")
    print_llm_usage()


def cmd_export(args):
    from kb.export import export_jsonl
    count = export_jsonl(Path(args.out))
    print(f"Exported {count} chunks to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="LLM knowledgebase pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create tables and indexes").set_defaults(func=cmd_init_db)

    p = sub.add_parser("ingest", help="Ingest files or folders")
    p.add_argument("paths", nargs="+")
    p.add_argument("--force", action="store_true", help="Re-ingest files that were already ingested")
    p.set_defaults(func=cmd_ingest)

    sub.add_parser("stats", help="Show per-document statistics").set_defaults(func=cmd_stats)

    for name, func, help_text in (("search", cmd_search, "Semantic search"),
                                  ("ask", cmd_ask, "Answer a question with citations")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("query")
        p.add_argument("-k", type=int, default=None, help="Number of passages to retrieve")
        p.set_defaults(func=func)

    p = sub.add_parser("export", help="Export all chunks to JSONL")
    p.add_argument("out")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args()

    problems = settings.validate()
    if problems:
        sys.exit("Configuration error:\n  " + "\n  ".join(problems))

    args.func(args)


if __name__ == "__main__":
    main()
