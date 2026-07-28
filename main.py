"""AstrBot adapter for the NOVA CAC knowledge-pack assistant."""

from __future__ import annotations

import asyncio
import weakref
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .nova_cac.core import ConversationMemory, KnowledgeIndex, PackLoader
from .nova_cac.routing import command_allowed, extract_cac_query


@register(
    "astrbot_plugin_nova_cac",
    "whyself",
    "基于本地 NOVA 知识包与近期上下文的 CAC 风格问答",
    "0.1.0",
)
class NovaCacPlugin(Star):
    """Explicitly triggered `/cac` knowledge Q&A."""

    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context = context
        self.config = config or {}

        pack_root = Path(__file__).resolve().parent / "knowledge_pack"
        self.pack_loader = PackLoader(pack_root)
        self.knowledge_index = KnowledgeIndex(pack_root / "knowledge")
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
        self.retrieval_top_k = self._config_int(
            "retrieval_top_k",
            5,
            minimum=1,
            maximum=12,
        )
        self.max_context_chars = self._config_int(
            "max_context_chars",
            9000,
            minimum=1000,
            maximum=30000,
        )
        self._session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    @filter.regex(r"^/?cac(?:\s+.*)?$")
    async def cac(self, event: AstrMessageEvent):
        """Answer a NOVA question in private chat or an explicitly mentioned group."""

        raw_text = self._raw_plain_text(event)
        query = extract_cac_query(raw_text)
        is_private = bool(event.is_private_chat())
        mentioned = self._is_bot_mentioned(event)
        if query is None or not command_allowed(
            is_private=is_private,
            bot_mentioned=mentioned,
        ):
            self._stop_event(event)
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
                provider_id = await self.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
                if not provider_id:
                    yield event.plain_result("当前没有可用的聊天模型，请先在 AstrBot 中配置提供商。")
                    return

                system_prompt, user_prompt = await asyncio.to_thread(
                    self._prepare_prompts,
                    query,
                )
                response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    contexts=self.memory.contexts(session_key),
                )
                answer = str(getattr(response, "completion_text", "") or "").strip()
                if not answer:
                    logger.warning("NOVA CAC provider returned an empty response")
                    yield event.plain_result("这次模型没有生成有效回答，可以换个说法再问一次。")
                    return

                self.memory.append_exchange(session_key, query, answer)
                yield event.plain_result(answer)
            except FileNotFoundError:
                logger.exception("NOVA CAC mandatory knowledge-pack file is missing")
                yield event.plain_result("知识包不完整，暂时无法回答。请联系管理员检查插件文件。")
            except Exception:
                logger.exception("NOVA CAC answer generation failed")
                yield event.plain_result("这次回答没有生成成功，请稍后再试。")

    def _prepare_prompts(self, question: str) -> tuple[str, str]:
        # build_system_prompt intentionally rereads all four files on every call.
        system_prompt = self.pack_loader.build_system_prompt()
        chunks = self.knowledge_index.search(
            question,
            top_k=self.retrieval_top_k,
            max_chars=self.max_context_chars,
        )
        logger.info(
            "NOVA CAC retrieval selected %d chunks for query length %d",
            len(chunks),
            len(question),
        )
        return system_prompt, self.pack_loader.build_user_prompt(question, chunks)

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
    def _is_bot_mentioned(event: AstrMessageEvent) -> bool:
        try:
            from astrbot.api import message_components as comp

            return any(
                isinstance(component, comp.At)
                and str(component.qq) == str(event.get_self_id())
                for component in event.message_obj.message
            )
        except (AttributeError, ImportError):
            return False

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
            "私聊：/cac <问题>\n"
            "群聊：@机器人 /cac <问题>\n"
            "/cac reset：清空当前会话的近期上下文\n"
            "/cac help：查看用法"
        )
