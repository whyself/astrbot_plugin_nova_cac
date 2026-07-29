"""Helper to rebuild the Chroma vector index from a SQLite document list."""

from __future__ import annotations
from datetime import datetime, timezone
from .chunking import split_markdown


class ChunkIndexer:
    def __init__(
        self,
        chunk_store,
        vector_index,
        embed,
        chunk_size: int = 1200,
        overlap: int = 180,
    ):
        self.chunk_store = chunk_store
        self.vector_index = vector_index
        self.embed = embed
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _make_chunks(self, row) -> list:
        return split_markdown(
            str(row["yuque_id"]),
            row["body"],
            title=row["title"],
            repository=row["repository"],
            namespace=row["namespace"],
            slug=row["slug"],
            file_path=row["path"] or "",
            source_url=row["url"],
            updated_at=row["updated_at"],
            size=self.chunk_size,
            overlap=self.overlap,
        )

    async def index_document(self, row) -> dict:
        document_id = str(row["yuque_id"])
        self.chunk_store.delete_document(document_id)
        self.vector_index.delete_document(document_id)
        chunks = self._make_chunks(row)
        if not chunks:
            return {"document_id": document_id, "chunks": 0, "error": None}
        try:
            vectors = []
            for chunk in chunks:
                vector = await self.embed(chunk.embedding_text)
                if vector is None:
                    raise RuntimeError("embedding unavailable")
                vectors.append(vector)
        except Exception as exc:
            return {
                "document_id": document_id,
                "chunks": 0,
                "error": f"embedding failed: {type(exc).__name__}",
            }
        self.chunk_store.save_document_chunks(document_id, chunks)
        result = self.vector_index.upsert(chunks, vectors)
        return {
            "document_id": document_id,
            "chunks": result["succeeded"],
            "error": result["error"],
            "failed_ids": result["failed_ids"],
        }

    async def rebuild(self, rows) -> dict:
        self.chunk_store.clear()
        self.vector_index.clear()
        total = failed_documents = 0
        errors: list[str] = []
        for row in rows:
            res = await self.index_document(row)
            total += res["chunks"]
            if res["error"]:
                failed_documents += 1
                errors.append(f"{res['document_id']}: {res['error']}")
        return {
            "chunks": total,
            "failed_documents": failed_documents,
            "errors": errors,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
