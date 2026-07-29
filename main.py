"""AstrBot adapter for the NOVA CAC knowledge-pack assistant."""

from __future__ import annotations

import asyncio
import weakref
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .nova_cac.agent import (
    AGENT_ERROR,
    NO_EVIDENCE,
    NO_PROVIDER,
    SAFE_FAILURE,
    NovaCacAgent,
)
from .nova_cac.chunk_store import ChunkStore
from .nova_cac.core import ConversationMemory, PackLoader
from .nova_cac.local_corpus import LocalCorpus
from .nova_cac.retriever import HybridRetriever
from .nova_cac.routing import extract_cac_query
from .nova_cac.tools import (
    GetDocOutlineTool,
    GrepLocalDocsTool,
    ListDocsTool,
    ReadDocTool,
    SearchKnowledgeBaseTool,
)
from .nova_cac.vector_index import ChunkVectorIndex


@register(
    "astrbot_plugin_nova_cac",
    "whyself",
    "基于本地 NOVA 知识包与近期上下文的 CAC 风格问答",
    "0.2.0",
)
class NovaCacPlugin(Star):
    """Explicitly triggered `/cac` knowledge Q&A."""

    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context = context
        self.config = config or {}

        pack_root = Path(__file__).resolve().parent / "knowledge_pack"
        self.pack_loader = PackLoader(pack_root)
        self.memory = ConversationMemory(
            history_turns=self._config_int("history_turns", 6, minimum=0, maximum=20),
            max_sessions=self._config_int(
                "max_sessions",
                256,
                minimum=1,
                maximum=5000,
            ),
            max_chars=self._config_int(
                "history_max_chars",
                12000,
                minimum=1000,
                maximum=50000,
            ),
        )
        retrieval_top_k = self._config_int(
            "retrieval_top_k",
            5,
            minimum=1,
            maximum=12,
        )
        score_threshold = self._config_float(
            "score_threshold",
            0.2,
            minimum=0.0,
            maximum=1.0,
        )

        data_dir = StarTools.get_data_dir("astrbot_plugin_nova_cac")
        self._data_dir = data_dir
        self.chunk_store = ChunkStore(data_dir / "chunks.sqlite3")
        embedding_provider = self._resolve_embedding_provider()
        embedding_key = self._embedding_key(embedding_provider)
        vector_enabled = self._config_bool("enable_vector_search", True)
        vector_index = None
        embed_one = None
        embed_many = None
        if vector_enabled and embedding_provider is not None:
            dimension = self._embedding_dimension(embedding_provider)
            vector_index = ChunkVectorIndex(
                data_dir / "vectors",
                model=embedding_key,
                embedding_dimension=dimension,
            )
            embed_one = embedding_provider.get_embedding
            embed_many = self._batch_embedder(embedding_provider)
        elif vector_enabled:
            logger.warning(
                "NOVA CAC vector search disabled at runtime: "
                "no AstrBot Embedding Provider is available"
            )

        chunk_size = self._config_int(
            "chunk_size",
            1200,
            minimum=200,
            maximum=8000,
        )
        configured_overlap = self._config_int(
            "chunk_overlap",
            180,
            minimum=0,
            maximum=1000,
        )
        chunk_overlap = min(configured_overlap, max(0, chunk_size // 2 - 1))
        self.corpus = LocalCorpus(
            pack_root / "knowledge",
            self.chunk_store,
            vector_index=vector_index,
            embed_many=embed_many,
            embedding_key=embedding_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = HybridRetriever(
            self.corpus,
            self.chunk_store,
            vector_index=vector_index,
            embed_one=embed_one,
            top_k=retrieval_top_k,
            score_threshold=score_threshold,
        )
        diagnostics = self._config_bool("retrieval_diagnostics", False)
        self.agent = NovaCacAgent(
            self.context,
            lambda tracker: [
                SearchKnowledgeBaseTool(
                    retriever=self.retriever,
                    tracker=tracker,
                ),
                GrepLocalDocsTool(corpus=self.corpus),
                ReadDocTool(corpus=self.corpus, tracker=tracker),
                ListDocsTool(corpus=self.corpus),
                GetDocOutlineTool(corpus=self.corpus),
            ],
            diagnostics=diagnostics,
        )
        self._session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def initialize(self) -> None:
        result = await self.corpus.refresh()
        logger.info(
            "NOVA CAC local corpus: documents=%s chunks=%s rebuilt=%s vector_ready=%s",
            result.get("documents"),
            result.get("chunks"),
            result.get("rebuilt"),
            result.get("vector_ready"),
        )
        if result.get("vector_error"):
            logger.warning("NOVA CAC vector index fallback: %s", result["vector_error"])

    async def terminate(self) -> None:
        self.corpus.close()

    @filter.regex(r"^/?cac(?:\s+.*)?$")
    async def cac(self, event: AstrMessageEvent):
        """Answer a `/cac` NOVA question in either private or group chat."""

        raw_text = self._raw_plain_text(event)
        query = extract_cac_query(raw_text)
        if query is None:
            # The regex also sees AstrBot's wake-prefix-stripped `cac ...` form.
            # If the original message had no literal slash, leave it to the
            # normal AstrBot pipeline and never touch this plugin's history.
            return

        self._stop_event(event)
        command = query.casefold()
        session_key = self._session_key(event)

        if not query or command in {"help", "帮助", "用法"}:
            yield event.plain_result(self._help_text())
            return
        if command in {"reset", "clear", "清空", "重置"}:
            async with self._session_lock(session_key):
                cleared = self.memory.clear(session_key)
            message = "这段对话的 /cac 上下文已清空。" if cleared else "这段对话目前没有已保存的 /cac 上下文。"
            yield event.plain_result(message)
            return

        async with self._session_lock(session_key):
            try:
                system_prompt = await asyncio.to_thread(
                    self.pack_loader.build_system_prompt
                )
                answer = await self.agent.answer(
                    event,
                    query,
                    base_system_prompt=system_prompt,
                    contexts=self.memory.contexts(session_key),
                )
                if not answer:
                    yield event.plain_result(AGENT_ERROR)
                    return

                if answer not in {
                    AGENT_ERROR,
                    NO_EVIDENCE,
                    NO_PROVIDER,
                    SAFE_FAILURE,
                }:
                    self.memory.append_exchange(session_key, query, answer)
                yield event.plain_result(answer)
            except FileNotFoundError:
                logger.exception("NOVA CAC mandatory knowledge-pack file is missing")
                yield event.plain_result("知识包不完整，暂时无法回答。请联系管理员检查插件文件。")
            except Exception:
                logger.exception("NOVA CAC answer generation failed")
                yield event.plain_result("这次回答没有生成成功，请稍后再试。")

    def _config_int(
        self,
        key: str,
        default: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        getter = getattr(self.config, "get", None)
        raw: Any = getter(key, default) if callable(getter) else default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
        return min(maximum, max(minimum, value))

    def _config_float(
        self,
        key: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        getter = getattr(self.config, "get", None)
        raw: Any = getter(key, default) if callable(getter) else default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
        return min(maximum, max(minimum, value))

    def _config_bool(self, key: str, default: bool) -> bool:
        getter = getattr(self.config, "get", None)
        raw: Any = getter(key, default) if callable(getter) else default
        if isinstance(raw, str):
            return raw.casefold() in {"1", "true", "yes", "on"}
        return bool(raw)

    def _config_str(self, key: str, default: str = "") -> str:
        getter = getattr(self.config, "get", None)
        raw: Any = getter(key, default) if callable(getter) else default
        return str(raw or default).strip()

    def _resolve_embedding_provider(self):
        provider_id = self._config_str("embedding_provider_id")
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider is not None and hasattr(provider, "get_embedding"):
                return provider
            logger.warning(
                "NOVA CAC configured Embedding Provider was not found: %s",
                provider_id,
            )
            return None
        providers = self.context.get_all_embedding_providers()
        return providers[0] if providers else None

    def _embedding_key(self, provider) -> str:
        if provider is None:
            return ""
        configured = self._config_str("embedding_provider_id")
        provider_config = getattr(provider, "provider_config", {}) or {}
        provider_id = (
            configured
            or str(provider_config.get("id", ""))
            or type(provider).__name__
        )
        model = str(
            provider_config.get("model", "")
            or provider_config.get("model_name", "")
            or type(provider).__name__
        )
        dimension = self._embedding_dimension(provider) or "unknown"
        return f"provider={provider_id}|model={model}|dimension={dimension}"

    @staticmethod
    def _embedding_dimension(provider) -> int | None:
        getter = getattr(provider, "get_dim", None)
        if not callable(getter):
            return None
        try:
            dimension = int(getter())
            return dimension if dimension > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _batch_embedder(provider):
        async def embed_many(texts: list[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), 32):
                vectors.extend(await provider.get_embeddings(texts[start : start + 32]))
            return vectors

        return embed_many

    def _session_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_key] = lock
        return lock

    @staticmethod
    def _raw_plain_text(event: AstrMessageEvent) -> str:
        try:
            from astrbot.api import message_components as comp

            text = "".join(
                component.text
                for component in event.message_obj.message
                if isinstance(component, comp.Plain)
            ).strip()
            if text:
                return text
        except (AttributeError, ImportError):
            pass
        return str(getattr(event.message_obj, "message_str", "") or "").strip()

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        origin = getattr(event, "unified_msg_origin", None)
        if origin:
            return str(origin)
        if event.is_private_chat():
            return f"private:{event.get_sender_id()}"
        return f"group:{event.get_group_id()}"

    @staticmethod
    def _stop_event(event: AstrMessageEvent) -> None:
        setattr(event, "_nova_cac_command_handled", True)
        stop = getattr(event, "stop_event", None)
        if callable(stop):
            stop()

    @staticmethod
    def _help_text() -> str:
        return (
            "用法：\n"
            "群聊或私聊：/cac <问题>\n"
            "/cac reset：清空当前会话的近期上下文\n"
            "/cac help：查看用法"
        )
