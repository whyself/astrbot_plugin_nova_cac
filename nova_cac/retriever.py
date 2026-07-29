"""Chunk-level hybrid retrieval adapted from astrbot_plugin_nju_qa."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from .chunk_store import ChunkStore
from .keyword_index import ChunkKeywordIndex
from .local_corpus import LocalCorpus
from .models import ChunkResult, SearchResult

EmbedOne = Callable[[str], Awaitable[list[float] | None]]


class HybridRetriever:
    def __init__(
        self,
        corpus: LocalCorpus,
        chunk_store: ChunkStore,
        *,
        vector_index=None,
        embed_one: EmbedOne | None = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> None:
        self.corpus = corpus
        self.chunk_store = chunk_store
        self.vector_index = vector_index
        self.embed_one = embed_one
        self.top_k = max(1, int(top_k))
        self.score_threshold = max(0.0, min(1.0, float(score_threshold)))
        self.keyword_index = ChunkKeywordIndex()
        self._keyword_signature = ""
        self.last_error: str | None = None

    async def search(self, query: str) -> list[SearchResult]:
        await self.corpus.refresh()
        vector_hits = await self._vector_candidates(query)
        keyword_hits = self._keyword_candidates(query)
        merged = self._merge(vector_hits, keyword_hits)
        selected = sorted(
            merged.values(),
            key=lambda item: item.final_score,
            reverse=True,
        )

        output: list[SearchResult] = []
        per_document: dict[str, int] = {}
        for item in selected:
            if item.final_score < self.score_threshold:
                continue
            if per_document.get(item.document_id, 0) >= 2:
                continue
            document = self.corpus.documents.get(item.document_id)
            if document is None:
                continue
            per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
            reliable = item.final_score >= self.score_threshold
            item = ChunkResult(
                **{
                    **item.__dict__,
                    "reliable": reliable,
                }
            )
            output.append(
                SearchResult(
                    source_id=f"S{len(output) + 1}",
                    document=document,
                    score=item.final_score,
                    chunk=item,
                    vector_score=item.vector_relevance,
                    keyword_score=item.keyword_score,
                    retrieval_methods=item.retrieval_methods,
                    reliable=reliable,
                )
            )
            if len(output) >= self.top_k:
                break
        return output

    async def _vector_candidates(self, query: str) -> list[tuple[ChunkResult, float]]:
        if self.vector_index is None or self.embed_one is None:
            return []
        try:
            vector = await self.embed_one(query)
            if not vector:
                return []
            raw = self.vector_index.query(vector, n=self.top_k * 4)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"vector query failed: {type(exc).__name__}"
            return []

        results: list[tuple[ChunkResult, float]] = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for metadata, content, distance in zip(metadatas, documents, distances):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            results.append(
                (
                    ChunkResult(
                        chunk_id=metadata.get("chunk_id", ""),
                        document_id=metadata.get("document_id", ""),
                        title=metadata.get("title", ""),
                        content_snippet=content or "",
                        source_url=metadata.get("source_url", ""),
                        vector_raw_score=float(distance),
                        vector_relevance=relevance,
                        chunk_index=int(metadata.get("chunk_index", 0) or 0),
                        file_path=metadata.get("file_path", ""),
                    ),
                    relevance,
                )
            )
        return results

    def _keyword_candidates(self, query: str) -> list[tuple[ChunkResult, float]]:
        signature = self.chunk_store.content_signature()
        if signature != self._keyword_signature:
            self.keyword_index.build(self.chunk_store.all_chunks())
            self._keyword_signature = signature
        results: list[tuple[ChunkResult, float]] = []
        for hit in self.keyword_index.search(query, top_k=self.top_k * 4):
            results.append(
                (
                    ChunkResult(
                        chunk_id=hit.chunk.chunk_id,
                        document_id=hit.chunk.document_id,
                        title=hit.chunk.title,
                        content_snippet=hit.chunk.content,
                        source_url=hit.chunk.source_url,
                        keyword_score=hit.score,
                        chunk_index=hit.chunk.chunk_index,
                        file_path=hit.chunk.file_path,
                    ),
                    hit.score,
                )
            )
        return results

    @staticmethod
    def _merge(
        vector_hits: list[tuple[ChunkResult, float]],
        keyword_hits: list[tuple[ChunkResult, float]],
    ) -> dict[str, ChunkResult]:
        raw: dict[str, dict[str, object]] = {}
        for chunk, score in vector_hits:
            raw[chunk.chunk_id] = {
                "chunk": chunk,
                "vector": score,
                "keyword": 0.0,
                "methods": ["vector"],
            }
        for chunk, score in keyword_hits:
            entry = raw.setdefault(
                chunk.chunk_id,
                {
                    "chunk": chunk,
                    "vector": 0.0,
                    "keyword": 0.0,
                    "methods": [],
                },
            )
            if not entry["chunk"].content_snippet:
                entry["chunk"] = chunk
            entry["keyword"] = score
            entry["methods"].append("keyword")

        merged: dict[str, ChunkResult] = {}
        for chunk_id, entry in raw.items():
            chunk = entry["chunk"]
            vector = float(entry["vector"])
            keyword = float(entry["keyword"])
            methods = tuple(dict.fromkeys(entry["methods"]))
            if len(methods) == 2:
                final = 0.5 * vector + 0.5 * keyword
            else:
                final = vector or keyword
            merged[chunk_id] = ChunkResult(
                **{
                    **chunk.__dict__,
                    "vector_relevance": vector,
                    "keyword_score": keyword,
                    "final_score": final,
                    "retrieval_methods": methods,
                }
            )
        return merged

