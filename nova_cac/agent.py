"""Two-stage evidence-first Agent adapted from astrbot_plugin_nju_qa."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover - local unit-test fallback
    logger = logging.getLogger("nova_cac")

from .evidence import EvidenceExcerpt, SourceTracker
from .prompts import ANSWER_PROMPT, RESEARCH_PROMPT, SMALL_TALK_PROMPT

NO_PROVIDER = "当前没有可用的聊天模型，请先在 AstrBot 中配置提供商。"
NO_EVIDENCE = "目前资料里没有找到足够明确的内容。"
AGENT_ERROR = "这次回答没有生成成功，请稍后再试。"
SAFE_FAILURE = "现有证据还不足以形成可靠回答。"

ToolFactory = Callable[[SourceTracker], list[object]]
ToolLoop = Callable[..., Awaitable[object]]


class NovaCacAgent:
    """Research with tools, then answer only from selected evidence."""

    def __init__(
        self,
        context: object,
        tools: ToolFactory,
        *,
        tool_loop: ToolLoop | None = None,
        diagnostics: bool = False,
    ) -> None:
        self.context = context
        self.tools = tools
        self._tool_loop = tool_loop
        self.diagnostics = diagnostics

    async def answer(
        self,
        event: object,
        prompt: str,
        *,
        base_system_prompt: str,
        contexts: list[dict[str, str]] | None = None,
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        if not provider_id:
            return NO_PROVIDER
        contexts = contexts or []

        if _is_small_talk(prompt):
            return await self._small_talk(
                event,
                provider_id,
                prompt,
                base_system_prompt,
                contexts,
            )

        tracker = SourceTracker(diagnostics=self.diagnostics)
        await self._research(
            event,
            provider_id,
            prompt,
            base_system_prompt,
            contexts,
            tracker,
        )
        if self.diagnostics:
            logger.info(
                "NOVA CAC evidence: candidates=%s reads=%s files=%s",
                len(tracker.candidate_sources),
                len(tracker.evidence_excerpts),
                sorted({item.file_path for item in tracker.evidence_excerpts}),
            )
        if not tracker.evidence_excerpts:
            return NO_EVIDENCE
        return await self._answer_from_evidence(
            event,
            provider_id,
            prompt,
            base_system_prompt,
            contexts,
            tracker,
        )

    async def _small_talk(
        self,
        event,
        provider_id: str,
        prompt: str,
        base_system_prompt: str,
        contexts: list[dict[str, str]],
    ) -> str:
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=f"{base_system_prompt}\n\n{SMALL_TALK_PROMPT}",
                contexts=contexts,
                tracker=SourceTracker(),
                tools=[],
                max_steps=2,
            )
            return str(getattr(response, "completion_text", "") or "").strip() or AGENT_ERROR
        except Exception:  # noqa: BLE001
            logger.exception("NOVA CAC small-talk phase failed")
            return AGENT_ERROR

    async def _research(
        self,
        event,
        provider_id: str,
        prompt: str,
        base_system_prompt: str,
        contexts: list[dict[str, str]],
        tracker: SourceTracker,
    ) -> None:
        try:
            await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=f"{base_system_prompt}\n\n{RESEARCH_PROMPT}",
                contexts=contexts,
                tracker=tracker,
                max_steps=8,
            )
        except Exception:  # noqa: BLE001
            logger.exception("NOVA CAC research phase failed")

    async def _answer_from_evidence(
        self,
        event,
        provider_id: str,
        question: str,
        base_system_prompt: str,
        contexts: list[dict[str, str]],
        tracker: SourceTracker,
    ) -> str:
        excerpts = sorted(
            tracker.evidence_excerpts,
            key=lambda excerpt: (-excerpt.score, -len(excerpt.content)),
        )[:7]
        tracker.selected_excerpts = excerpts
        grounded_prompt = _build_grounded_prompt(question, excerpts)
        text = ""
        for attempt in range(2):
            try:
                response = await self._run_tool_loop(
                    event=event,
                    chat_provider_id=provider_id,
                    prompt=grounded_prompt,
                    system_prompt=f"{base_system_prompt}\n\n{ANSWER_PROMPT}",
                    contexts=contexts,
                    tracker=tracker,
                    tools=[],
                    max_steps=3,
                )
            except Exception:  # noqa: BLE001
                logger.exception("NOVA CAC answer phase failed")
                return AGENT_ERROR
            text = str(getattr(response, "completion_text", "") or "").strip()
            if _is_no_evidence(text) or _all_claims_grounded(text, excerpts):
                break
            grounded_prompt += (
                "\n\n上一版存在未逐句标注或无效的 [E#] 证据，"
                "请让每个事实句都紧跟有效证据标记后重答。"
            )
        else:
            return SAFE_FAILURE
        return _finalize(text, excerpts, tracker.verified_urls, question)

    async def _run_tool_loop(self, *, tracker: SourceTracker, **kwargs):
        tools = kwargs.pop("tools", None)
        if tools is None:
            tools = self.tools(tracker)
        max_steps = int(kwargs.pop("max_steps", 8))
        if self._tool_loop is not None:
            return await self._tool_loop(
                tools=tools,
                max_steps=max_steps,
                tracker=tracker,
                **kwargs,
            )
        from astrbot.core.agent.tool import ToolSet

        return await self.context.tool_loop_agent(
            tools=ToolSet(tools),
            max_steps=max_steps,
            tool_call_timeout=60,
            **kwargs,
        )


def _build_grounded_prompt(
    question: str,
    excerpts: list[EvidenceExcerpt],
) -> str:
    parts = [f"请回答原问题：{question}"]
    for excerpt in excerpts:
        location = ""
        if excerpt.line_start is not None:
            location = f"第 {excerpt.line_start}—{excerpt.line_end or excerpt.line_start} 行"
        parts.append(
            f"[{excerpt.evidence_id}]\n"
            f"标题：{excerpt.title}\n"
            f"文件：{excerpt.file_path} {location}\n"
            f"原文：{excerpt.url or 'n/a'}\n"
            f"版本状态：{excerpt.version_status}\n"
            f"内容：\n{excerpt.content}"
        )
    return "\n\n".join(parts)


def _finalize(
    text: str,
    excerpts: list[EvidenceExcerpt],
    verified_urls: set[str],
    question: str,
) -> str:
    used_ids = _extract_evidence_ids(text)
    if not used_ids:
        return NO_EVIDENCE if _is_no_evidence(text) else SAFE_FAILURE
    known_ids = {excerpt.evidence_id for excerpt in excerpts}
    if any(evidence_id not in known_ids for evidence_id in used_ids):
        return SAFE_FAILURE
    if not _all_claims_grounded(text, excerpts):
        return SAFE_FAILURE
    used = [excerpt for excerpt in excerpts if excerpt.evidence_id in used_ids]
    if not used and not _is_no_evidence(text):
        return SAFE_FAILURE
    visible = re.sub(r"\[E\d+]", "", text).strip()
    allowed_urls = set(verified_urls)
    allowed_urls.update(excerpt.url for excerpt in used if excerpt.url)
    visible = _strip_unverified_urls(visible, allowed_urls)
    if not _asks_for_sources(question) or not used:
        return visible
    citations = []
    seen: set[str] = set()
    for excerpt in used:
        key = excerpt.document_id or excerpt.file_path
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            f"{len(citations) + 1}. 《{excerpt.title}》：{excerpt.url or excerpt.file_path}"
        )
    return visible + ("\n\n来源：\n" + "\n".join(citations) if citations else "")


def _extract_evidence_ids(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[(E\d+)]", text)))


def _all_claims_grounded(text: str, excerpts: list[EvidenceExcerpt]) -> bool:
    known_ids = {excerpt.evidence_id for excerpt in excerpts}
    clauses = re.findall(r"[^。！？!?\n]+[。！？!?]?(?:\s*\[E\d+])?", text)
    substantive = [clause for clause in clauses if re.search(r"[\w一-鿿]{2}", clause)]
    return bool(substantive) and all(
        (ids := _extract_evidence_ids(clause))
        and all(evidence_id in known_ids for evidence_id in ids)
        for clause in substantive
    )


def _is_no_evidence(text: str) -> bool:
    return any(phrase in text for phrase in ("没有明确", "没有找到", "证据不足", "资料不足"))


def _asks_for_sources(question: str) -> bool:
    return any(word in question.casefold() for word in ("来源", "出处", "原文", "链接"))


def _strip_unverified_urls(text: str, allowed_urls: set[str]) -> str:
    return re.sub(
        r"https?://[^\s<>()，。；：\"']+",
        lambda match: match.group(0) if match.group(0) in allowed_urls else "",
        text,
    )


def _is_small_talk(prompt: str) -> bool:
    normalized = re.sub(r"[^\w一-鿿]", "", prompt).casefold()
    return normalized in {
        "你好",
        "在吗",
        "你是谁",
        "谢谢",
        "谢谢你",
        "hello",
        "hi",
    }
