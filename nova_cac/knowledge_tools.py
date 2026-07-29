"""Framework-independent implementation behind the Agent knowledge tool."""

from __future__ import annotations

from astrbot.api import logger

from .agent import SourceTracker
from .doc_utils import clean_document_body
from .models import ChunkResult
from .retriever import HybridRetriever


async def search_knowledge_base(
    retriever: HybridRetriever, tracker: SourceTracker, query: str, **scope
) -> dict:
    """Search and record candidate sources returned to the Agent."""

    if not retriever.index.all_documents():
        return {
            "reliable": False,
            "reason": "本地知识包尚未建立索引。",
            "candidates": [],
        }
    results = await retriever.search(query, **scope)
    if not results:
        return {
            "reliable": False,
            "reason": "知识库中暂未找到可靠答案。",
            "candidates": [],
        }
    # Search/grep hits are candidates only; the model must read the actual
    # document to turn them into grounded evidence.
    tracker.add_candidates(results)
    if getattr(tracker, "diagnostics", False):
        logger.info("NOVA search_knowledge_base: query=%r results=%d", query, len(results))
        for i, r in enumerate(results[:10], 1):
            logger.info(
                "NOVA search candidate %d: title=%s path=%s score=%s reliable=%s",
                i,
                r.document.title,
                r.document.path,
                r.score,
                r.reliable,
            )
    return {
        "reliable": all(r.reliable for r in results),
        "candidates": [_result_to_dict(r.chunk, r) for r in results],
    }


def _clean_snippet(text: str) -> str:
    """Remove Yuque HTML/markdown noise from short snippets."""
    return clean_document_body(text[:2600])


def _result_to_dict(chunk: ChunkResult | None, result) -> dict:
    if chunk is None:
        return {
            "source_id": result.source_id,
            "title": result.document.title,
            "author": "",
            "book_name": result.document.repository,
            "content_snippet": _clean_snippet(result.document.body),
            "file_path": str(result.document.path or ""),
            "yuque_id": result.document.yuque_id,
            "slug": result.document.slug,
            "source_url": result.document.url,
            "score": result.score,
            "score_type": "combined_similarity",
            "retrieval_method": "hybrid",
            "reliable": result.reliable,
        }
    return {
        "source_id": result.source_id,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "author": "",
        "book_name": result.document.repository,
        "content_snippet": _clean_snippet(chunk.content_snippet),
        "file_path": chunk.file_path or str(result.document.path or ""),
        "yuque_id": result.document.yuque_id,
        "slug": chunk.slug or result.document.slug,
        "source_url": chunk.source_url or result.document.url,
        "score": chunk.final_score,
        "vector_raw_score": chunk.vector_raw_score,
        "vector_score_type": chunk.vector_score_type,
        "vector_relevance": chunk.vector_relevance,
        "keyword_score": chunk.keyword_score,
        "final_score": chunk.final_score,
        "retrieval_methods": list(chunk.retrieval_methods),
        "reliable": chunk.reliable,
    }
