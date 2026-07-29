"""Persistent SQLite store for chunk metadata used by keyword and vector indexes."""

from __future__ import annotations
import hashlib
import sqlite3
from pathlib import Path
from .chunking import Chunk


class ChunkStore:
    """Stores chunk metadata and content in SQLite.

    The vector embeddings live in Chroma; this table is the authoritative source
    of chunk existence and metadata for keyword search and lifecycle management.
    """

    def __init__(self, path: Path):
        self.path = path
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        if self._conn:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                repository TEXT NOT NULL DEFAULT '',
                namespace TEXT NOT NULL DEFAULT '',
                slug TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        return self._conn

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            content_hash=row["content_hash"],
            title=row["title"],
            repository=row["repository"],
            namespace=row["namespace"],
            slug=row["slug"],
            file_path=row["file_path"],
            source_url=row["source_url"],
            updated_at=row["updated_at"],
        )

    def save_document_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        """Replace all chunks for a document atomically."""
        conn = self._ensure_open()
        with conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks
                (chunk_id, document_id, chunk_index, content, content_hash,
                 title, repository, namespace, slug, file_path, source_url, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.content,
                        c.content_hash,
                        c.title,
                        c.repository,
                        c.namespace,
                        c.slug,
                        c.file_path,
                        c.source_url,
                        c.updated_at,
                    )
                    for c in chunks
                ],
            )

    def delete_document(self, document_id: str) -> int:
        conn = self._ensure_open()
        with conn:
            cur = conn.execute(
                "DELETE FROM chunks WHERE document_id=?", (document_id,)
            )
            return cur.rowcount

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        conn = self._ensure_open()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        conn = self._ensure_open()
        rows = conn.execute(
            "SELECT * FROM chunks ORDER BY document_id, chunk_index"
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def chunk_count(self) -> int:
        conn = self._ensure_open()
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def content_signature(self) -> str:
        """Return a cheap fingerprint of the current chunk contents.

        Used to invalidate in-memory keyword indexes without loading all content.
        """
        conn = self._ensure_open()
        rows = conn.execute(
            "SELECT content_hash FROM chunks ORDER BY chunk_id"
        ).fetchall()
        combined = "".join(row[0] for row in rows)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def document_chunk_count(self, document_id: str) -> int:
        conn = self._ensure_open()
        row = conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id=?", (document_id,)
        ).fetchone()
        return row[0] if row else 0

    def documents_with_chunks(self) -> int:
        conn = self._ensure_open()
        row = conn.execute(
            "SELECT count(DISTINCT document_id) FROM chunks"
        ).fetchone()
        return row[0] if row else 0

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        conn = self._ensure_open()
        row = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def clear(self) -> None:
        conn = self._ensure_open()
        with conn:
            conn.execute("DELETE FROM chunks")

    def update_document_metadata(
        self,
        document_id: str,
        *,
        title: str | None = None,
        repository: str | None = None,
        namespace: str | None = None,
        slug: str | None = None,
        file_path: str | None = None,
        source_url: str | None = None,
        updated_at: str | None = None,
    ) -> int:
        """Update metadata columns for every chunk of a document."""
        conn = self._ensure_open()
        updates: dict[str, str] = {}
        for key, value in {
            "title": title,
            "repository": repository,
            "namespace": namespace,
            "slug": slug,
            "file_path": file_path,
            "source_url": source_url,
            "updated_at": updated_at,
        }.items():
            if value is not None:
                updates[key] = value
        if not updates:
            return 0
        with conn:
            cur = conn.execute(
                f"UPDATE chunks SET {', '.join(f'{k}=?' for k in updates)} WHERE document_id=?",
                (*updates.values(), document_id),
            )
            return cur.rowcount

    def has_document(self, document_id: str) -> bool:
        conn = self._ensure_open()
        row = conn.execute(
            "SELECT 1 FROM chunks WHERE document_id=? LIMIT 1", (document_id,)
        ).fetchone()
        return bool(row)

    def vacuum(self) -> None:
        conn = self._ensure_open()
        conn.execute("VACUUM")
