"""Persistent vector store backed by PostgreSQL + pgvector (cosine metric)."""
import re
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from .models import Chunk

# Whitelist of columns that may be used as metadata filters (kept parameterized).
_FILTERABLE = {"chapter", "section", "document_id", "source"}


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        self.database_url = database_url
        self.dimensions = dimensions
        self._conn = psycopg.connect(database_url, connect_timeout=10)
        register_vector(self._conn)

    def create_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Recreate the table if the stored vector dimension no longer matches
            # the embedding model (e.g. switching from a 384-dim to a 768-dim model).
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' "
                "AND tablename='chunks')"
            )
            if cur.fetchone()[0]:
                cur.execute(
                    "SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_attribute a "
                    "WHERE a.attrelid='chunks'::regclass AND a.attname='embedding'"
                )
                row = cur.fetchone()
                m = re.match(r"vector(?:\((\d+)\))?", row[0]) if row else None
                existing_dim = int(m.group(1)) if m and m.group(1) else None
                if existing_dim is not None and existing_dim != self.dimensions:
                    cur.execute("DROP TABLE chunks")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    text PRIMARY KEY,
                    strategy    text NOT NULL,
                    document_id text NOT NULL,
                    source      text NOT NULL,
                    page        integer NOT NULL,
                    chapter     text,
                    section     text,
                    text        text NOT NULL,
                    embedding   vector({self.dimensions})
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks (strategy)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        self._conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk],
               vectors: list[list[float]]) -> int:
        """Replace all chunks of a strategy atomically (no duplicates on re-run)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if vectors and len(vectors[0]) != self.dimensions:
            raise ValueError(
                f"vector dimension {len(vectors[0])} != store dimension {self.dimensions}"
            )
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{strategy}|{chunk.chunk_id}"
                cur.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, strategy, document_id, source, page,
                         chapter, section, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk_id,
                        strategy,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.metadata.get("chapter"),
                        chunk.metadata.get("section"),
                        chunk.text,
                        Vector(vector),
                    ),
                )
        self._conn.commit()
        return len(chunks)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Cosine-similarity search with optional whitelisted metadata filters."""
        filters = filters or {}
        query_vector = Vector(query_vector)
        sql = (
            "SELECT chunk_id, document_id, source, page, chapter, section, text, "
            "1 - (embedding <=> %s) AS score "
            "FROM chunks WHERE strategy = %s"
        )
        params: list[Any] = [query_vector, strategy]
        for key, value in filters.items():
            if key not in _FILTERABLE:
                raise ValueError(f"unsupported filter field: {key}")
            sql += f" AND {key} = %s"
            params.append(value)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([query_vector, top_k])

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "chunk_id": r[0],
                "document_id": r[1],
                "source": r[2],
                "page": r[3],
                "chapter": r[4],
                "section": r[5],
                "text": r[6],
                "score": round(float(r[7]), 6),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
