"""
Retrieval and question answering over the knowledgebase.

With a Gemini key: retrieved passages are passed to the LLM, which answers
only from them and cites sources as [1], [2], ...
Without a key: returns the most relevant passages (extractive mode).
"""

import hashlib

from kb import llm
from kb.config import settings
from kb.embed import get_embedder
from kb.store import similarity_search

_answer_cache: dict[str, dict] = {}


def search(query: str, top_k: int = None) -> list[dict]:
    """Embed the query and return the most similar chunks."""
    return similarity_search(get_embedder().embed_query(query), top_k or settings.top_k)


def cite(chunk: dict) -> str:
    """Human-readable source reference for a chunk."""
    pages = (f"p. {chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
             else f"pp. {chunk['page_start']}-{chunk['page_end']}")
    parts = [chunk.get("title") or chunk["source_file"]]
    if chunk.get("chapter"):
        parts.append(chunk["chapter"])
    parts.append(pages)
    return ", ".join(parts)


def format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[{i}] ({cite(c)})\n{c['chunk_text']}" for i, c in enumerate(chunks, start=1))


def answer(query: str, top_k: int = None) -> dict:
    """
    Answer a question from the knowledgebase.

    Returns:
        {"answer": str, "mode": "llm" | "extractive", "sources": [{"ref", "similarity"}]}
    """
    chunks = search(query, top_k)
    sources = [{"ref": f"[{i}] {cite(c)}", "similarity": round(float(c["similarity"]), 3)}
               for i, c in enumerate(chunks, start=1)]

    if not chunks:
        return {"answer": "The knowledgebase is empty. Ingest some documents first.", "mode": "none", "sources": []}

    if llm.enabled():
        # Same question over the same passages -> reuse the earlier answer instead of paying again
        key = hashlib.sha256(
            (query.strip().lower() + "\x00" + "\x00".join(c["chunk_text"] for c in chunks)).encode()
        ).hexdigest()
        if key in _answer_cache:
            return {**_answer_cache[key], "cached": True}

        prompt = (
            "Answer the question using ONLY the numbered passages below. Cite passages inline like [1]. "
            "Be concise (under 150 words). If the passages do not contain the answer, say so.\n\n"
            f"PASSAGES:\n{format_context(chunks)}\n\nQUESTION: {query}\n\nANSWER:"
        )
        text = llm.generate(prompt, max_output_tokens=600)
        if text:
            result = {"answer": text.strip(), "mode": "llm", "sources": sources}
            if len(_answer_cache) >= 256:
                _answer_cache.pop(next(iter(_answer_cache)))  # drop the oldest
            _answer_cache[key] = result
            return result

    best = chunks[0]["chunk_text"]
    excerpt = best if len(best) <= 900 else best[:900].rsplit(" ", 1)[0] + " ..."
    return {"answer": f"Most relevant passage [1]:\n{excerpt}", "mode": "extractive", "sources": sources}
