"""Persistent Chroma chunk vector index with model/dimension identity and safe batching."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any


class ChunkVectorIndex:
    """Adapter around Chroma persistent storage.

    The collection name encodes the embedding model so that incompatible vectors
    are never queried together.  Collection metadata stores the model and vector
    dimension; a mismatch causes the old collection to be dropped and rebuilt.
    """

    DEFAULT_SPACE = "cosine"
    EMBEDDING_VERSION = "v1"

    def __init__(
        self,
        directory: Path,
        model: str,
        embedding_dimension: int | None = None,
        batch_size: int = 64,
    ):
        self.directory = directory
        self.model = model
        self._embedding_dimension = embedding_dimension
        self.batch_size = max(1, batch_size)
        self._client: Any | None = None
        self._collection_name = self._collection_name_for(model)
        self._last_error: str | None = None
        self._last_failed_ids: list[str] = []

    @classmethod
    def _collection_name_for(cls, model: str) -> str:
        model_slug = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
        return f"nju_qa_chunks_{cls.EMBEDDING_VERSION}_{model_slug}"

    def _ensure_client(self):
        import chromadb

        if self._client is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.directory))
        return self._client

    def _collection(self):
        client = self._ensure_client()
        try:
            collection = client.get_collection(self._collection_name)
        except Exception:
            collection = None
        if collection is None:
            meta = {"embedding_model": self.model, "hnsw:space": self.DEFAULT_SPACE}
            if self._embedding_dimension:
                meta["embedding_dimension"] = str(self._embedding_dimension)
            collection = client.create_collection(
                self._collection_name,
                metadata=meta,
            )
            return collection

        stored = collection.metadata or {}
        stored_model = stored.get("embedding_model", "")
        if stored_model != self.model:
            client.delete_collection(self._collection_name)
            return self._collection()
        stored_dim = stored.get("embedding_dimension")
        if (
            self._embedding_dimension
            and stored_dim
            and int(stored_dim) != self._embedding_dimension
        ):
            client.delete_collection(self._collection_name)
            return self._collection()
        return collection

    def count(self) -> int:
        try:
            return self._collection().count()
        except Exception:
            return 0

    def delete_document(self, document_id: str) -> int:
        try:
            collection = self._collection()
            before = collection.count()
            collection.delete(where={"document_id": document_id})
            after = collection.count()
            return max(0, before - after)
        except Exception as exc:
            self._last_error = f"delete_document failed: {type(exc).__name__}"
            return 0

    def _safe_metadata(self, chunk) -> dict[str, Any]:
        """Store metadata without API keys or secrets."""
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title,
            "book_name": chunk.repository,
            "namespace": chunk.namespace,
            "slug": chunk.slug,
            "file_path": chunk.file_path,
            "source_url": chunk.source_url,
            "updated_at": chunk.updated_at,
            "content_hash": chunk.content_hash,
        }

    def upsert(
        self,
        chunks,
        embeddings: list[list[float]],
    ) -> dict[str, Any]:
        """Batch upsert chunks with per-batch error visibility.

        Returns a dict with ``succeeded``, ``failed_ids`` and ``error``.
        """
        self._last_error = None
        self._last_failed_ids = []
        if not chunks:
            return {"succeeded": 0, "failed_ids": [], "error": None}
        collection = self._collection()
        succeeded = 0
        failed_ids: list[str] = []
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            batch_embeddings = embeddings[start : start + self.batch_size]
            ids = [c.chunk_id for c in batch_chunks]
            try:
                collection.upsert(
                    ids=ids,
                    documents=[c.content for c in batch_chunks],
                    embeddings=batch_embeddings,
                    metadatas=[self._safe_metadata(c) for c in batch_chunks],
                )
                succeeded += len(batch_chunks)
            except Exception as exc:
                failed_ids.extend(ids)
                self._last_error = f"upsert batch failed: {type(exc).__name__}"
        self._last_failed_ids = failed_ids
        return {
            "succeeded": succeeded,
            "failed_ids": failed_ids,
            "error": self._last_error,
        }

    def query(
        self,
        embedding: list[float],
        n: int = 20,
        document_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        collection = self._collection()
        where: dict[str, Any] | None = None
        if document_filter:
            where = {"document_id": {"$in": document_filter}}
        result = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        return result

    def clear(self) -> int:
        """Drop the collection entirely and return the previous count."""
        try:
            client = self._ensure_client()
            collection = client.get_collection(self._collection_name)
            count = collection.count()
            client.delete_collection(self._collection_name)
            self._client = None
            return count
        except Exception:
            self._client = None
            return 0

    def close(self) -> None:
        self._client = None

    def last_error(self) -> str | None:
        return self._last_error

    def last_failed_ids(self) -> list[str]:
        return list(self._last_failed_ids)

    def embedding_dimension(self) -> int | None:
        """Return stored dimension if known."""
        try:
            meta = self._collection().metadata or {}
            dim = meta.get("embedding_dimension")
            return int(dim) if dim else None
        except Exception:
            return None

    def embedding_model(self) -> str:
        try:
            meta = self._collection().metadata or {}
            return str(meta.get("embedding_model", self.model))
        except Exception:
            return self.model
