"""
Export the knowledgebase as a JSONL dataset.

One line per chunk, with its source metadata, ready for fine-tuning data
prep, evaluation sets, or loading into another vector store.
"""

import json
from pathlib import Path

from kb.store import iter_chunks_for_export


def export_jsonl(out_path: Path) -> int:
    """Write every chunk to a JSONL file. Returns the number of records."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_chunks_for_export():
            record = {
                "id": f"{row['document_id']}-{row['chunk_index']}",
                "text": row["chunk_text"],
                "source": row["source_file"],
                "title": row["title"],
                "author": row["author"],
                "language": row["language"],
                "chapter": row["chapter"],
                "pages": [row["page_start"], row["page_end"]],
                "tokens": row["token_count"],
                "labels": row["labels"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
