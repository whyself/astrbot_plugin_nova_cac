"""Chunk-level hybrid retriever combining vector and keyword signals."""

from __future__ import annotations
import asyncio
import httpx
from pathlib import Path
from typing import Any

from .config import PluginConfig
from .document_index import DocumentIndex
from .keyword_index import ChunkKeywordIndex
from .knowledge_structure import (
    _is_archived_path,
    _matches_prefix,
    _normalize_path,
    _path_segments,
)
from .models import ChunkResult, Document, SearchResult


class HybridRetriever:
    def __init__(
        self,
        index: DocumentIndex,
        config: PluginConfig,
        chunk_store=None,
        vector_index=None,
        keyword_index=None,
        embed=None,
    ):
        self.index = index
        self.config = config
        self.chunk_store = chunk_store
        self.vector_index = vector_index
        self.keyword_index = keyword_index or ChunkKeywordIndex()
        self._custom_embed = embed
        self._last_error: str | None = None

    async def _embed(self, text: str) -> list[float] | None:
        if self._custom_embed is not None:
            try:
                return await self._custom_embed(text)
            except Exception as exc:
                self._last_error = f"embedding failed: {type(exc).__name__}"
                return None
        if not self.config.embedding_api_key or not self.config.embedding_base_url:
            return None
        url = self.config.embedding_base_url.rstrip("/") + "/embeddings"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.config.embedding_api_key}"
                    },
                    json={
                        "model": self.config.embedding_model,
                        "input": text[:8000],
                    },
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
        except Exception as exc:
            self._last_error = f"embedding failed: {type(exc).__name__}"
            return None

    async def embed_text(self, text: str) -> list[float] | None:
        """Public alias used by SyncService."""
        return await self._embed(text)

    @staticmethod
    def _doc(row) -> Document:
        return Document(
            yuque_id=row["yuque_id"],
            title=row["title"],
            repository=row["repository"],
            namespace=row["namespace"],
            slug=row["slug"],
            url=row["url"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            body=row["body"],
            path=Path(row["path"]) if row["path"] else None,
        )

    def _chunk_in_scope(
        self,
        chunk: ChunkResult,
        scope: dict[str, Any],
        lookup: dict[str, Any],
    ) -> bool:
        """Return True when a chunk belongs to the requested retrieval scope."""
        namespace = scope.get("namespace", "")
        repository = scope.get("repository", "")
        path_prefix = scope.get("path_prefix", "")
        document_ids = scope.get("document_ids")
        include_archived = scope.get("include_archived", True)

        if document_ids is not None and chunk.document_id not in document_ids:
            return False

        if namespace:
            if chunk.namespace != namespace:
                row = lookup.get(chunk.document_id)
                if not row or row["namespace"] != namespace:
                    return False

        if repository:
            repo = chunk.repository or ""
            if not repo:
                row = lookup.get(chunk.document_id)
                repo = (row["repository"] if row else "") or ""
            if repository.casefold() not in repo.casefold():
                return False

        if path_prefix:
            path = chunk.file_path or ""
            if not path:
                row = lookup.get(chunk.document_id)
                path = row["path"] if row else ""
            segments = _path_segments(_normalize_path(path))
            prefix_segments = _path_segments(path_prefix)
            anchor = 1 if namespace else 0
            if not _matches_prefix(segments[anchor:], prefix_segments):
                return False

        if not include_archived:
            path = chunk.file_path or ""
            if not path:
                row = lookup.get(chunk.document_id)
                path = row["path"] if row else ""
            if _is_archived_path(_path_segments(_normalize_path(path))):
                return False

        return True

    def _rebuild_keyword_index(self) -> None:
        if self.chunk_store is None:
            return
        signature = self.chunk_store.content_signature()
        if getattr(self, "_keyword_index_signature", None) == signature:
            return
        self.keyword_index.build(self.chunk_store.all_chunks())
        self._keyword_index_signature = signature

    async def _vector_candidates(
        self, query: str, top_k: int
    ) -> list[tuple[ChunkResult, float]]:
        if self.vector_index is None or not self.config.enable_vector_search:
            return []
        try:
            if self.vector_index.count() <= 0:
                return []
        except Exception as exc:
            self._last_error = f"vector count failed: {type(exc).__name__}"
            return []
        vector = await self._embed(query)
        if vector is None:
            return []
        try:
            raw = self.vector_index.query(vector, n=top_k * 4)
        except Exception as exc:
            self._last_error = f"vector query failed: {type(exc).__name__}"
            return []
        docs = raw.get("documents", [[]])[0]
        meta = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results = []
        for m, content, distance in zip(meta, docs, distances):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            cr = ChunkResult(
                chunk_id=m.get("chunk_id", ""),
                document_id=m.get("document_id", ""),
                title=m.get("title", ""),
                content_snippet=content or "",
                source_url=m.get("source_url", ""),
                vector_raw_score=float(distance),
                vector_score_type="cosine_distance",
                vector_relevance=relevance,
                chunk_index=int(m.get("chunk_index", 0) or 0),
                file_path=m.get("file_path", ""),
                slug=m.get("slug", ""),
                namespace=m.get("namespace", ""),
            )
            results.append((cr, relevance))
        return results

    async def _keyword_candidates(
        self, query: str, top_k: int
    ) -> list[tuple[ChunkResult, float]]:
        if self.chunk_store is None:
            return []
        # Rebuild keyword index lazily from the store.
        self._rebuild_keyword_index()
        hits = self.keyword_index.search(query, top_k=top_k * 4)
        results = []
        for hit in hits:
            cr = ChunkResult(
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                title=hit.chunk.title,
                content_snippet=hit.chunk.content,
                source_url=hit.chunk.source_url,
                keyword_score=hit.score,
                chunk_index=hit.chunk.chunk_index,
                file_path=hit.chunk.file_path,
                slug=hit.chunk.slug,
                namespace=hit.chunk.namespace,
                repository=hit.chunk.repository,
            )
            results.append((cr, hit.score))
        return results

    def _merge_candidates(
        self,
        vector_hits: list[tuple[ChunkResult, float]],
        keyword_hits: list[tuple[ChunkResult, float]],
    ) -> dict[str, ChunkResult]:
        """Merge by chunk_id keeping both scores and retrieval methods."""
        merged: dict[str, ChunkResult] = {}
        for cr, rel in vector_hits:
            existing = merged.get(cr.chunk_id)
            if existing is None:
                cr = ChunkResult(
                    **{
                        **cr.__dict__,
                        "vector_relevance": rel,
                        "retrieval_methods": ("vector",),
                    }
                )
                merged[cr.chunk_id] = cr
            else:
                merged[cr.chunk_id] = ChunkResult(
                    **{
                        **existing.__dict__,
                        "vector_raw_score": cr.vector_raw_score,
                        "vector_relevance": rel,
                        "retrieval_methods": tuple(
                            dict.fromkeys([*existing.retrieval_methods, "vector"])
                        ),
                    }
                )
        for cr, score in keyword_hits:
            existing = merged.get(cr.chunk_id)
            if existing is None:
                cr = ChunkResult(
                    **{
                        **cr.__dict__,
                        "keyword_score": score,
                        "retrieval_methods": ("keyword",),
                    }
                )
                merged[cr.chunk_id] = cr
            else:
                merged[cr.chunk_id] = ChunkResult(
                    **{
                        **existing.__dict__,
                        "keyword_score": score,
                        "retrieval_methods": tuple(
                            dict.fromkeys([*existing.retrieval_methods, "keyword"])
                        ),
                    }
                )
        return merged

    def _normalize(
        self, merged: dict[str, ChunkResult]
    ) -> dict[str, ChunkResult]:
        """Normalize keyword scores to [0,1]; vector relevance is already 0-1 from cosine distance."""
        if not merged:
            return {}
        max_keyword = max((c.keyword_score for c in merged.values()), default=0.0)
        scale = max(max_keyword, 1.0)
        result = {}
        for chunk_id, cr in merged.items():
            v = cr.vector_relevance  # already 0-1
            k = cr.keyword_score / scale if scale > 0 else 0.0
            has_v = "vector" in cr.retrieval_methods
            has_k = "keyword" in cr.retrieval_methods
            if has_v and has_k:
                final = 0.5 * v + 0.5 * k
            elif has_v:
                final = v
            else:
                final = k
            result[chunk_id] = ChunkResult(
                **{
                    **cr.__dict__,
                    "keyword_score": k,
                    "final_score": final,
                }
            )
        return result

    def _limit_same_document(
        self, items: list[ChunkResult], max_per_doc: int
    ) -> list[ChunkResult]:
        counts: dict[str, int] = {}
        output = []
        for item in items:
            if counts.get(item.document_id, 0) >= max_per_doc:
                continue
            counts[item.document_id] = counts.get(item.document_id, 0) + 1
            output.append(item)
        return output

    def _merge_adjacent_chunks(
        self, items: list[ChunkResult]
    ) -> list[ChunkResult]:
        if not items or self.chunk_store is None:
            return items
        by_doc: dict[str, list[ChunkResult]] = {}
        for item in items:
            by_doc.setdefault(item.document_id, []).append(item)
        merged: list[ChunkResult] = []
        for doc_id, doc_items in by_doc.items():
            doc_items.sort(key=lambda x: x.chunk_index)
            i = 0
            while i < len(doc_items):
                current = doc_items[i]
                # Merge with next if indices are adjacent and both are keyword-only or both vector-only.
                # Conservative: merge adjacent chunks from the same document.
                while (
                    i + 1 < len(doc_items)
                    and doc_items[i + 1].chunk_index == doc_items[i].chunk_index + 1
                ):
                    next_item = doc_items[i + 1]
                    combined = ChunkResult(
                        chunk_id=current.chunk_id,
                        document_id=current.document_id,
                        title=current.title,
                        content_snippet=(current.content_snippet + "\n\n" + next_item.content_snippet)[:2400],
                        source_url=current.source_url,
                        vector_raw_score=max(current.vector_raw_score, next_item.vector_raw_score),
                        vector_score_type=current.vector_score_type,
                        vector_relevance=max(current.vector_relevance, next_item.vector_relevance),
                        keyword_score=max(current.keyword_score, next_item.keyword_score),
                        final_score=max(current.final_score, next_item.final_score),
                        retrieval_methods=tuple(
                            dict.fromkeys([*current.retrieval_methods, *next_item.retrieval_methods])
                        ),
                        reliable=current.reliable or next_item.reliable,
                        chunk_index=current.chunk_index,
                        file_path=current.file_path,
                        slug=current.slug or next_item.slug,
                        namespace=current.namespace or next_item.namespace,
                    )
                    current = combined
                    i += 1
                merged.append(current)
                i += 1
        # Re-sort by final score.
        merged.sort(key=lambda x: x.final_score, reverse=True)
        return merged

    async def debug_search(self, query: str, **scope) -> dict:
        self._last_error = None
        if self.chunk_store is not None:
            self._rebuild_keyword_index()
        vector_hits, keyword_hits = await asyncio.gather(
            self._vector_candidates(query, self.config.retrieval_top_k),
            self._keyword_candidates(query, self.config.retrieval_top_k),
        )
        merged = self._merge_candidates(vector_hits, keyword_hits)
        normalized = self._normalize(merged)
        lookup = {r["yuque_id"]: r for r in self.index.all_documents()}

        # Mark reliability based on threshold and supporting evidence.
        threshold = self.config.score_threshold
        strong_vector = 0.7
        for chunk_id, cr in normalized.items():
            has_keyword = cr.keyword_score > 0
            has_strong_vector = cr.vector_relevance >= strong_vector
            normalized[chunk_id] = ChunkResult(
                **{
                    **cr.__dict__,
                    "reliable": cr.final_score >= threshold
                    and (has_keyword or has_strong_vector),
                }
            )

        scoped = {
            chunk_id: cr
            for chunk_id, cr in normalized.items()
            if self._chunk_in_scope(cr, scope, lookup)
        }
        selected = self._select_from_normalized(scoped, threshold)

        # Fallback: if the hybrid pipeline drops everything, trust pure keyword hits.
        # Keyword hits already have title/path/phrase boosts applied.
        if not selected and keyword_hits:
            filtered_keyword = [
                (cr, score)
                for cr, score in keyword_hits
                if self._chunk_in_scope(cr, scope, lookup)
            ]
            if filtered_keyword:
                fallback: dict[str, ChunkResult] = {}
                fallback_threshold = threshold * 0.5
                for cr, score in filtered_keyword:
                    if cr.chunk_id in fallback:
                        continue
                    fallback[cr.chunk_id] = ChunkResult(
                        **{
                            **cr.__dict__,
                            "keyword_score": score,
                            "final_score": score,
                            "retrieval_methods": tuple(
                                dict.fromkeys([*cr.retrieval_methods, "keyword"])
                            ),
                            "reliable": score >= fallback_threshold,
                        }
                    )
                selected = self._select_from_normalized(fallback, fallback_threshold)

        search_results = []
        for i, item in enumerate(selected):
            row = lookup.get(item.document_id)
            doc = self._doc(row) if row else None
            if doc is None:
                continue
            search_results.append(
                SearchResult(
                    source_id=f"S{i + 1}",
                    document=doc,
                    score=item.final_score,
                    chunk=item,
                    vector_score=item.vector_relevance,
                    keyword_score=item.keyword_score,
                    retrieval_methods=item.retrieval_methods,
                    reliable=item.reliable,
                )
            )

        return {
            "query": query,
            "mode": "hybrid"
            if vector_hits and keyword_hits
            else ("vector" if vector_hits else "keyword"),
            "embedding_available": bool(vector_hits),
            "query_terms": self.keyword_index.extract_terms(query),
            "keyword_candidates": keyword_hits,
            "vector_candidates": vector_hits,
            "merged": list(normalized.values()),
            "selected": search_results,
            "threshold": threshold,
            "chunk_count": self.chunk_store.chunk_count() if self.chunk_store else 0,
            "vector_count": self.vector_index.count() if self.vector_index else 0,
            "last_error": self._last_error,
            "scope": scope,
        }

    def _select_from_normalized(
        self, normalized: dict[str, ChunkResult], threshold: float
    ) -> list[ChunkResult]:
        sorted_items = sorted(
            normalized.values(), key=lambda x: x.final_score, reverse=True
        )
        max_per_doc = max(2, self.config.retrieval_top_k // 2)
        limited = self._limit_same_document(sorted_items, max_per_doc)
        merged_adjacent = self._merge_adjacent_chunks(limited)
        return [
            item for item in merged_adjacent if item.final_score >= threshold
        ][: self.config.retrieval_top_k]

    async def search(self, query: str, **scope) -> list[SearchResult]:
        report = await self.debug_search(query, **scope)
        return report["selected"]

    def debug_text(self, report: dict) -> str:
        lines = [
            f"检索模式：{report['mode']}",
            f"Embedding 可用：{report['embedding_available']}",
            f"查询拆分：{', '.join(report.get('query_terms', []))}",
            f"Chunk 总数：{report['chunk_count']}",
            f"向量 chunk 数：{report['vector_count']}",
            f"阈值：{report['threshold']:.3f}",
        ]
        if report.get("last_error"):
            lines.append(f"最近错误：{report['last_error']}")
        lines.append(f"关键词候选：{len(report['keyword_candidates'])}")
        for cr, score in report["keyword_candidates"][:10]:
            snippet = cr.content_snippet.replace(chr(10), " ")[:60]
            lines.append(f"  关键词 《{cr.title}》#{cr.chunk_index} {snippet}...: {score:.3f}")
        lines.append(f"向量候选：{len(report['vector_candidates'])}")
        for cr, rel in report["vector_candidates"][:10]:
            snippet = cr.content_snippet.replace(chr(10), " ")[:60]
            lines.append(
                f"  向量 《{cr.title}》#{cr.chunk_index} {snippet}...: rel={rel:.3f} dist={cr.vector_raw_score:.3f}"
            )
        lines.append("合并与过滤：")
        for item in report.get("merged", [])[:10]:
            reason = "低于阈值" if item.final_score < report["threshold"] else "保留"
            lines.append(
                f"  《{item.title}》#{item.chunk_index} final={item.final_score:.3f} "
                f"v={item.vector_relevance:.3f} k={item.keyword_score:.3f} "
                f"methods={','.join(item.retrieval_methods)} {reason}"
            )
        lines.append("最终选择：")
        for r in report["selected"]:
            lines.append(
                f"  《{r.document.title}》#{r.chunk.chunk_index if r.chunk else '-'}: {r.score:.3f}"
            )
        return "\n".join(lines)

    async def rebuild_embeddings(self) -> int:
        from .chunk_indexer import ChunkIndexer

        indexer = ChunkIndexer(
            self.chunk_store,
            self.vector_index,
            self._embed,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        result = await indexer.rebuild(self.index.all_documents())
        self._rebuild_keyword_index()
        if result.get("errors"):
            self._last_error = "; ".join(result["errors"])
        return result["chunks"]

    def last_error(self) -> str | None:
        return self._last_error
