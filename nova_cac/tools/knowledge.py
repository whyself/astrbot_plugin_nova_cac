from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from ..agent import SourceTracker
from ..knowledge_tools import search_knowledge_base
from ..retriever import HybridRetriever


@dataclass
class SearchKnowledgeBaseTool(FunctionTool):
    """Search synchronized NJU documents and return only traceable source material."""

    name: str = "search_knowledge_base"
    description: str = (
        "检索NOVA知识库。涉及NOVA具体事实、政策、流程、时间、地点、"
        "联系方式或课程要求时必须先调用此工具。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的完整问题或关键词"},
                "namespace": {
                    "type": "string",
                    "description": "限制在指定知识库命名空间（path 第一段）",
                },
                "repository": {
                    "type": "string",
                    "description": "限制在指定 repository",
                },
                "path_prefix": {
                    "type": "string",
                    "description": "限制在指定路径前缀（相对 namespace）",
                },
                "document_ids": {
                    "type": "string",
                    "description": "空格或逗号分隔的文档 ID 集合",
                },
                "include_archived": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否包含路径中出现“归档”的文档",
                },
            },
            "required": ["query"],
        }
    )
    retriever: HybridRetriever | None = None
    tracker: SourceTracker | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> str:
        """Current AstrBot tool-loop entry point."""
        query = str(kwargs.get("query", ""))
        scope = {k: v for k, v in kwargs.items() if k != "query"}
        if "document_ids" in scope and isinstance(scope["document_ids"], str):
            ids = [v.strip() for v in re.split(r"[,\s]+", scope["document_ids"]) if v.strip()]
            scope["document_ids"] = set(ids) if ids else None
        return await self._search(query, **scope)

    async def run(self, event: AstrMessageEvent, query: str, **scope) -> str:
        """Compatibility entry point for AstrBot versions using ``run``."""
        return await self._search(query, **scope)

    async def _search(self, query: str, **scope) -> str:
        if self.retriever is None or self.tracker is None:
            return "知识库工具未初始化。"
        return await search_knowledge_base(self.retriever, self.tracker, query, **scope)
