"""Mirror the embedded NOVA Markdown pack into the reference document indexes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from .chunk_indexer import ChunkIndexer
from .chunking import split_markdown
from .models import Document

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class LocalPackSync:
    def __init__(
        self,
        knowledge_root: Path,
        store,
        index,
        chunk_store,
        vector_index,
        config,
        *,
        state_path: Path,
    ) -> None:
        self.knowledge_root = Path(knowledge_root).resolve()
        self.store = store
        self.index = index
        self.chunk_store = chunk_store
        self.vector_index = vector_index
        self.config = config
        self.state_path = state_path
        self._lock = asyncio.Lock()
        self._last_vector_attempt: str | None = None

    async def refresh(self, force: bool = False) -> dict[str, object]:
        async with self._lock:
            return await self._refresh_unlocked(force)

    async def _refresh_unlocked(self, force: bool = False) -> dict[str, object]:
        documents, signature = self._scan()
        identity = {
            "content_signature": signature,
            "embedding_base_url": self.config.embedding_base_url.rstrip("/"),
            "embedding_model": self.config.embedding_model,
            "enable_vector_search": self.config.enable_vector_search,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }
        vector_requested = bool(
            self.config.enable_vector_search
            and self.config.embedding_api_key
            and self.config.embedding_base_url
        )
        state = self._read_state()
        state_matches = all(state.get(key) == value for key, value in identity.items())
        attempt_key = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        needs_vector_retry = bool(
            vector_requested
            and not state.get("vector_ready", False)
            and self._last_vector_attempt != attempt_key
        )
        if (
            not force
            and state_matches
            and self.index.document_count()
            and self.chunk_store.chunk_count()
            and not needs_vector_retry
        ):
            return {
                "rebuilt": False,
                "documents": self.index.document_count(),
                "chunks": self.chunk_store.chunk_count(),
                "vector_ready": bool(state.get("vector_ready", False)),
            }

        seen: set[str] = set()
        for document in documents:
            seen.add(document.yuque_id)
            self.store.write(document)
            self.index.upsert(document)
        for row in self.index.delete_missing("nova", seen):
            self.store.remove(Path(row["path"]))
            self.chunk_store.delete_document(str(row["yuque_id"]))
            if self.config.enable_vector_search:
                self.vector_index.delete_document(str(row["yuque_id"]))

        vector_ready = vector_requested
        errors: list[str] = []
        if vector_ready:
            self._last_vector_attempt = attempt_key
            from .retriever import HybridRetriever

            retriever = HybridRetriever(
                self.index,
                self.config,
                chunk_store=self.chunk_store,
                vector_index=self.vector_index,
            )
            indexer = ChunkIndexer(
                self.chunk_store,
                self.vector_index,
                retriever.embed_text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            result = await indexer.rebuild(self.index.all_documents())
            errors = list(result.get("errors", []))
            vector_ready = not errors
            if errors:
                self.vector_index.clear()
                self._rebuild_keyword_chunks()
        else:
            self._rebuild_keyword_chunks()

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {**identity, "vector_ready": vector_ready},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "rebuilt": True,
            "documents": len(documents),
            "chunks": self.chunk_store.chunk_count(),
            "vector_ready": vector_ready,
            "vector_error": "; ".join(errors[:5]) or None,
        }

    def _rebuild_keyword_chunks(self) -> None:
        self.chunk_store.clear()
        for row in self.index.all_documents():
            chunks = split_markdown(
                str(row["yuque_id"]),
                row["body"],
                title=row["title"],
                repository=row["repository"],
                namespace=row["namespace"],
                slug=row["slug"],
                file_path=row["path"] or "",
                source_url=row["url"],
                updated_at=row["updated_at"],
                size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            self.chunk_store.save_document_chunks(str(row["yuque_id"]), chunks)

    def _scan(self) -> tuple[list[Document], str]:
        signature = hashlib.sha256()
        documents: list[Document] = []
        for path in sorted(self.knowledge_root.rglob("*.md")):
            raw = path.read_text(encoding="utf-8")
            relative = path.relative_to(self.knowledge_root).as_posix()
            signature.update(relative.encode("utf-8"))
            signature.update(raw.encode("utf-8"))
            metadata = _parse_front_matter(raw)
            body = _strip_front_matter(raw)
            heading = _TITLE.search(body)
            title = metadata.get("title") or (heading.group(1).strip() if heading else path.stem)
            document_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
            stored_path = Path("nova") / Path(relative)
            documents.append(
                Document(
                    yuque_id=document_id,
                    title=title,
                    repository=relative.split("/", 1)[0],
                    namespace="nova",
                    slug=path.stem,
                    url=metadata.get("source_url", ""),
                    created_at=metadata.get("created_at", ""),
                    updated_at=metadata.get("updated_at", ""),
                    body=body,
                    path=stored_path,
                )
            )
        return documents, signature.hexdigest()

    def _read_state(self) -> dict[str, object]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}


def _parse_front_matter(raw: str) -> dict[str, str]:
    matched = _FRONT_MATTER.match(raw)
    if matched is None:
        return {}
    output: dict[str, str] = {}
    for line in matched.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            output[key.strip()] = value.strip().strip("\"'")
    return output


def _strip_front_matter(raw: str) -> str:
    matched = _FRONT_MATTER.match(raw)
    return raw[matched.end() :] if matched else raw
