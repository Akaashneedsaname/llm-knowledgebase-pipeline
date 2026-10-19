"""
PostgreSQL + pgvector storage.

Schema:
- documents:        one row per ingested file (hash, title, author, labels, OCR stats)
- chunks_<dim>:     one row per chunk with its embedding; a table per vector size
                    so local (384) and Gemini (768) embeddings can coexist

Ingestion is idempotent: files are identified by a SHA-256 of their bytes,
so re-running the pipeline on the same folder skips unchanged files.
"""

import json
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from kb.config import settings


def _chunks_table() -> str:
    return f"chunks_{settings.embedding_dim}"


@contextmanager
def get_connection():
    """Get a database connection, committing on success and rolling back on error."""
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Create the pgvector extension, tables and indexes if they don't exist."""
    table = _chunks_table()
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                file_hash TEXT NOT NULL,
                source_file TEXT NOT NULL,
                title TEXT,
                author TEXT,
                language TEXT,
                page_count INTEGER,
                ocr_pages INTEGER,
                ocr_mean_confidence REAL,
                metadata JSONB,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (file_hash, embedding_model)
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                chapter TEXT,
                token_count INTEGER,
                embedding vector({settings.embedding_dim}) NOT NULL,
                UNIQUE (document_id, chunk_index)
            );
        """)
        # HNSW works well from the first row (no training step, unlike IVFFlat)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_embedding
            ON {table} USING hnsw (embedding vector_cosine_ops);
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);")


def embedding_model_name() -> str:
    if settings.embedding_provider == "gemini":
        return f"gemini:{settings.gemini_embedding_model}"
    return f"local:{settings.local_embedding_model}"


def find_document(file_hash: str) -> Optional[int]:
    """Return the id of an already-ingested file, if any."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE file_hash = %s AND embedding_model = %s",
            (file_hash, embedding_model_name()),
        )
        row = cur.fetchone()
        return row[0] if row else None


def delete_document(document_id: int):
    """Delete a document and (via cascade) its chunks."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def save_document(doc: dict, chunks: list, vectors: list[list[float]]) -> int:
    """
    Store a document and its embedded chunks in a single transaction.

    Args:
        doc: Document fields (file_hash, source_file, title, author, language, page_count,
             ocr_pages, ocr_mean_confidence, metadata)
        chunks: List of Chunk objects
        vectors: One embedding per chunk

    Returns:
        The new document id
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"Got {len(vectors)} embeddings for {len(chunks)} chunks")

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (file_hash, source_file, title, author, language, page_count,
                                   ocr_pages, ocr_mean_confidence, metadata, embedding_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            doc["file_hash"], doc["source_file"], doc.get("title"), doc.get("author"),
            doc.get("language"), doc.get("page_count"), doc.get("ocr_pages"),
            doc.get("ocr_mean_confidence"), json.dumps(doc.get("metadata", {})),
            embedding_model_name(),
        ))
        document_id = cur.fetchone()[0]

        rows = [
            (document_id, c.index, c.text, c.page_start, c.page_end, c.chapter, c.token_count, v)
            for c, v in zip(chunks, vectors)
        ]
        execute_values(
            cur,
            f"""INSERT INTO {_chunks_table()}
                (document_id, chunk_index, chunk_text, page_start, page_end, chapter, token_count, embedding)
                VALUES %s""",
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s::vector)",
            page_size=200,
        )
        return document_id


def similarity_search(query_embedding: list[float], top_k: int = None, document_id: int = None) -> list[dict]:
    """
    Cosine similarity search over chunks.

    Returns:
        Chunks with document info and a similarity score (1 = identical)
    """
    top_k = top_k or settings.top_k
    where = "WHERE c.document_id = %s" if document_id else ""
    params = [query_embedding] + ([document_id] if document_id else []) + [query_embedding, top_k]

    with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT c.chunk_text, c.page_start, c.page_end, c.chapter, c.chunk_index,
                   d.source_file, d.title, d.author,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM {_chunks_table()} c
            JOIN documents d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, params)
        return [dict(r) for r in cur.fetchall()]


def iter_chunks_for_export():
    """Yield every chunk with its document metadata, in document order."""
    with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT d.id AS document_id, d.source_file, d.title, d.author, d.language,
                   d.metadata->'labels' AS labels,
                   c.chunk_index, c.chunk_text, c.page_start, c.page_end, c.chapter, c.token_count
            FROM {_chunks_table()} c
            JOIN documents d ON d.id = c.document_id
            WHERE d.embedding_model = %s
            ORDER BY d.id, c.chunk_index
        """, (embedding_model_name(),))
        for row in cur:
            yield dict(row)


def get_stats() -> list[dict]:
    """Per-document statistics."""
    with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT d.id, d.source_file, d.title, d.author, d.page_count, d.ocr_pages,
                   d.ocr_mean_confidence, COUNT(c.id) AS chunks, COALESCE(SUM(c.token_count), 0) AS tokens
            FROM documents d
            LEFT JOIN {_chunks_table()} c ON c.document_id = d.id
            WHERE d.embedding_model = %s
            GROUP BY d.id
            ORDER BY d.id
        """, (embedding_model_name(),))
        return [dict(r) for r in cur.fetchall()]
