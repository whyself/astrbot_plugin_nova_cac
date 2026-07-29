"""Synchronize the embedded Markdown pack into persistent local indexes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from .chunk_store import ChunkStore
from .chunking import Chunk, split_markdown
from .models import LocalDocument

EmbedMany = Callable[[list[str]], Awaitable[list[list[float]]]]

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class LocalCorpus:
    """Read-only Markdown source with persistent derived indexes."""

    def __init__(
        self,
        knowledge_root: Path,
        chunk_store: ChunkStore,
        *,
        vector_index=None,
        embed_many: EmbedMany | None = None,
        embedding_key: str = "",
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
    ) -> None:
        self.knowledge_root = Path(knowledge_root).resolve()
        self.chunk_store = chunk_store
        self.vector_index = vector_index
        self.embed_many = embed_many
        self.embedding_key = embedding_key
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.documents: dict[str, LocalDocument] = {}
        self._state_path = self.chunk_store.path.parent / "corpus_state.json"
        self.last_vector_error: str | None = None
        self._last_attempt_key: tuple[str, str, int, int] | None = None
        self._refresh_lock = asyncio.Lock()

    async def refresh(self, force: bool = False) -> dict[str, object]:
        async with self._refresh_lock:
            return await self._refresh_unlocked(force)

    async def _refresh_unlocked(self, force: bool = False) -> dict[str, object]:
        documents, content_signature = self._scan()
        self.documents = {doc.document_id: doc for doc in documents}
        state = self._read_state()
        vector_requested = self.vector_index is not None and self.embed_many is not None
        attempt_key = (
            content_signature,
            self.embedding_key,
            self.chunk_size,
            self.chunk_overlap,
        )
        needs_rebuild = (
            force
            or state.get("content_signature") != content_signature
            or state.get("embedding_key", "") != self.embedding_key
            or state.get("chunk_size") != self.chunk_size
            or state.get("chunk_overlap") != self.chunk_overlap
            or (
                vector_requested
                and not state.get("vector_ready", False)
                and self._last_attempt_key != attempt_key
            )
            or self.chunk_store.chunk_count() == 0
        )
        if not needs_rebuild:
            return {
                "rebuilt": False,
                "documents": len(documents),
                "chunks": self.chunk_store.chunk_count(),
                "vector_ready": bool(state.get("vector_ready", False)),
            }

        all_chunks: list[Chunk] = []
        self.chunk_store.clear()
        for document in documents:
            chunks = split_markdown(
                document.document_id,
                document.body,
                title=document.title,
                repository=document.category,
                namespace="nova",
                slug=document.absolute_path.stem,
                file_path=document.relative_path,
                source_url=document.source_url,
                updated_at=document.updated_at,
                size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            self.chunk_store.save_document_chunks(document.document_id, chunks)
            all_chunks.extend(chunks)

        vector_ready = False
        self.last_vector_error = None
        if self.vector_index is not None:
            self.vector_index.clear()
        if vector_requested and all_chunks:
            try:
                vectors = await self.embed_many(
                    [chunk.embedding_text[:8000] for chunk in all_chunks]
                )
                if len(vectors) != len(all_chunks):
                    raise RuntimeError("embedding result count mismatch")
                result = self.vector_index.upsert(all_chunks, vectors)
                vector_ready = (
                    result.get("succeeded") == len(all_chunks)
                    and not result.get("failed_ids")
                )
                if not vector_ready:
                    self.last_vector_error = str(result.get("error") or "vector upsert failed")
            except Exception as exc:  # noqa: BLE001
                self.last_vector_error = f"{type(exc).__name__}: {exc}"

        self._write_state(
            {
                "content_signature": content_signature,
                "embedding_key": self.embedding_key,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "vector_ready": vector_ready,
            }
        )
        self._last_attempt_key = attempt_key
        return {
            "rebuilt": True,
            "documents": len(documents),
            "chunks": len(all_chunks),
            "vector_ready": vector_ready,
            "vector_error": self.last_vector_error,
        }

    def get_document(
        self,
        *,
        document_id: str = "",
        file_path: str = "",
    ) -> LocalDocument | None:
        if document_id:
            return self.documents.get(document_id)
        normalized = Path(file_path).as_posix().lstrip("/")
        return next(
            (doc for doc in self.documents.values() if doc.relative_path == normalized),
            None,
        )

    def close(self) -> None:
        self.chunk_store.close()
        if self.vector_index is not None:
            self.vector_index.close()

    def _scan(self) -> tuple[list[LocalDocument], str]:
        documents: list[LocalDocument] = []
        signature = hashlib.sha256()
        for path in sorted(self.knowledge_root.rglob("*.md")):
            raw = path.read_text(encoding="utf-8")
            relative_path = path.relative_to(self.knowledge_root).as_posix()
            signature.update(relative_path.encode("utf-8"))
            signature.update(raw.encode("utf-8"))
            metadata = _parse_front_matter(raw)
            heading = _TITLE.search(_strip_front_matter(raw))
            title = metadata.get("title") or (heading.group(1).strip() if heading else path.stem)
            document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
            documents.append(
                LocalDocument(
                    document_id=document_id,
                    title=title,
                    relative_path=relative_path,
                    absolute_path=path,
                    category=relative_path.split("/", 1)[0],
                    source_url=metadata.get("source_url", ""),
                    created_at=metadata.get("created_at", ""),
                    updated_at=metadata.get("updated_at", ""),
                    body=raw,
                )
            )
        return documents, signature.hexdigest()

    def _read_state(self) -> dict[str, object]:
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write_state(self, value: dict[str, object]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _parse_front_matter(raw: str) -> dict[str, str]:
    matched = _FRONT_MATTER.match(raw)
    if matched is None:
        return {}
    metadata: dict[str, str] = {}
    for line in matched.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata


def _strip_front_matter(raw: str) -> str:
    matched = _FRONT_MATTER.match(raw)
    return raw[matched.end() :] if matched else raw
