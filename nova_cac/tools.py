"""Agent tools for searching and reading the embedded local Markdown pack."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from astrbot.api import FunctionTool
except ImportError:  # pragma: no cover - local unit-test fallback
    class FunctionTool:  # type: ignore[no-redef]
        pass

from .evidence import SourceTracker
from .local_corpus import LocalCorpus
from .retriever import HybridRetriever


@dataclass
class SearchKnowledgeBaseTool(FunctionTool):
    name: str = "search_knowledge_base"
    description: str = (
        "使用向量语义与 BM25 关键词混合检索 NOVA 本地知识包，返回候选文档和片段。"
        "适合口语化、概括性、理念性或不确定原文措辞的问题；候选结果必须再用 read_doc 精读。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "完整问题或语义检索词"}
            },
            "required": ["query"],
        }
    )
    retriever: HybridRetriever | None = None
    tracker: SourceTracker | None = None

    async def call(self, context, **kwargs):
        return await self._run(query=str(kwargs.get("query", "")))

    async def run(self, event, query: str):
        return await self._run(query=query)

    async def _run(self, query: str) -> dict[str, object]:
        if self.retriever is None or self.tracker is None:
            return {"error": "检索工具尚未初始化"}
        results = await self.retriever.search(query)
        self.tracker.add_candidates(results)
        return {
            "mode": "hybrid" if any("vector" in r.retrieval_methods for r in results) else "keyword",
            "candidates": [
                {
                    "document_id": result.document.document_id,
                    "title": result.document.title,
                    "file_path": result.document.relative_path,
                    "source_url": result.document.source_url,
                    "chunk_index": result.chunk.chunk_index,
                    "content_snippet": result.chunk.content_snippet[:1800],
                    "score": round(result.score, 4),
                    "retrieval_methods": list(result.retrieval_methods),
                }
                for result in results
            ],
        }


@dataclass
class GrepLocalDocsTool(FunctionTool):
    name: str = "grep_local_docs"
    description: str = (
        "按关键词逐行搜索 NOVA 本地 Markdown，返回文件路径、行号和上下文。"
        "适合制度条款、日期、名称和原文定位；找到后必须用 read_doc 精读。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "空格分隔的 1—4 个核心关键词",
                },
                "context_lines": {
                    "type": "integer",
                    "default": 2,
                    "description": "匹配行前后的上下文行数",
                },
            },
            "required": ["keywords"],
        }
    )
    corpus: LocalCorpus | None = None

    async def call(self, context, **kwargs):
        return await self._run(
            keywords=str(kwargs.get("keywords", "")),
            context_lines=int(kwargs.get("context_lines", 2)),
        )

    async def run(self, event, keywords: str, context_lines: int = 2):
        return await self._run(keywords=keywords, context_lines=context_lines)

    async def _run(self, keywords: str, context_lines: int = 2) -> dict[str, object]:
        if self.corpus is None:
            return {"error": "文档工具尚未初始化"}
        await self.corpus.refresh()
        terms = [term.casefold() for term in re.split(r"\s+", keywords.strip()) if term][:4]
        if not terms:
            return {"results": []}
        context_lines = max(0, min(8, context_lines))
        results: list[dict[str, object]] = []
        for document in self.corpus.documents.values():
            lines = document.body.splitlines()
            matched = [
                index
                for index, line in enumerate(lines)
                if any(term in line.casefold() for term in terms)
            ]
            if not matched:
                continue
            start = max(0, min(matched) - context_lines)
            end = min(len(lines), max(matched) + context_lines + 1)
            window = "\n".join(lines[start:end])
            coverage = sum(term in window.casefold() for term in terms)
            title_hits = sum(term in document.title.casefold() for term in terms)
            results.append(
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "file_path": document.relative_path,
                    "source_url": document.source_url,
                    "line_start": start + 1,
                    "line_end": end,
                    "content": window[:3000],
                    "score": coverage + title_hits * 2,
                }
            )
        results.sort(key=lambda item: float(item["score"]), reverse=True)
        return {"results": results[:10]}


@dataclass
class ReadDocTool(FunctionTool):
    name: str = "read_doc"
    description: str = (
        "按 document_id 或 file_path 读取本地 Markdown 的精确行范围。"
        "只有本工具读到的正文才会成为最终回答证据。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "候选文档 ID"},
                "file_path": {"type": "string", "description": "knowledge 下的相对路径"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "default": 40},
            },
        }
    )
    corpus: LocalCorpus | None = None
    tracker: SourceTracker | None = None

    async def call(self, context, **kwargs):
        return await self._run(**kwargs)

    async def run(self, event, **kwargs):
        return await self._run(**kwargs)

    async def _run(
        self,
        document_id: str = "",
        file_path: str = "",
        start_line: int = 1,
        end_line: int = 40,
        **_,
    ) -> dict[str, object]:
        if self.corpus is None or self.tracker is None:
            return {"error": "文档工具尚未初始化"}
        await self.corpus.refresh()
        document = self.corpus.get_document(
            document_id=str(document_id),
            file_path=str(file_path),
        )
        if document is None:
            return {"error": "未找到文档"}
        lines = document.body.splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), max(start, int(end_line)), start + 79)
        content = "\n".join(lines[start - 1 : end])
        excerpt = self.tracker.add_read(
            document,
            content,
            line_start=start,
            line_end=end,
        )
        return {
            "evidence_id": excerpt.evidence_id,
            "document_id": document.document_id,
            "title": document.title,
            "file_path": document.relative_path,
            "source_url": document.source_url,
            "line_start": start,
            "line_end": end,
            "content": content[:8000],
        }


@dataclass
class ListDocsTool(FunctionTool):
    name: str = "list_docs"
    description: str = "列出 NOVA 本地知识包中的文章标题、分类和路径，用于导航，不能替代正文读取。"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "可选的分类目录过滤"}
            },
        }
    )
    corpus: LocalCorpus | None = None

    async def call(self, context, **kwargs):
        return await self._run(category=str(kwargs.get("category", "")))

    async def run(self, event, category: str = ""):
        return await self._run(category=category)

    async def _run(self, category: str = "") -> dict[str, object]:
        if self.corpus is None:
            return {"error": "文档工具尚未初始化"}
        await self.corpus.refresh()
        docs = [
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "category": doc.category,
                "file_path": doc.relative_path,
            }
            for doc in self.corpus.documents.values()
            if not category or category.casefold() in doc.category.casefold()
        ]
        return {"documents": docs}


@dataclass
class GetDocOutlineTool(FunctionTool):
    name: str = "get_doc_outline"
    description: str = "读取文档 Markdown 标题及行号，用于确定随后 read_doc 的精确范围。"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "file_path": {"type": "string"},
            },
        }
    )
    corpus: LocalCorpus | None = None

    async def call(self, context, **kwargs):
        return await self._run(**kwargs)

    async def run(self, event, **kwargs):
        return await self._run(**kwargs)

    async def _run(self, document_id: str = "", file_path: str = "", **_) -> dict[str, object]:
        if self.corpus is None:
            return {"error": "文档工具尚未初始化"}
        await self.corpus.refresh()
        document = self.corpus.get_document(document_id=document_id, file_path=file_path)
        if document is None:
            return {"error": "未找到文档"}
        headings = [
            {"line": index, "heading": line.strip()}
            for index, line in enumerate(document.body.splitlines(), 1)
            if re.match(r"^#{1,6}\s+", line.strip())
        ]
        return {
            "document_id": document.document_id,
            "title": document.title,
            "file_path": document.relative_path,
            "headings": headings,
        }
