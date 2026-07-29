"""Evidence-first two-stage Agent for NJU QA.

The Agent splits every factual question into two phases:

1. Research phase: the model may use search/navigation/reading tools to locate
   and read concrete evidence.  The natural-language text produced by this
   phase is ignored; only the evidence that was actually read is retained.

2. Answer phase: the model receives the original question and the collected
   evidence excerpts, and must answer using only those excerpts.  Every factual
   claim must be marked with an internal evidence id ``[E#]``.  Citations are
   rendered only for the excerpts the model actually used.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .doc_utils import read_document_content
from .evidence import (
    EvidenceExcerpt,
    classify_version_status,
    evidence_excerpts_from_read,
)
from .models import SearchResult
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT,
)


NO_PROVIDER = "当前未配置 LLM 服务。请联系管理员配置后再试。"
AGENT_ERROR = "当前无法调用 LLM 服务，请稍后重试。"
NO_EVIDENCE = "知识库中暂未找到可靠资料，我不能据此给出 NOVA 的具体结论。"
SAFE_FAILURE = "当前无法根据知识库给出可靠回答，请稍后重试或换个方式提问。"

_MAX_EVIDENCE = 7
_MAX_RESEARCH_STEPS = 12
_MAX_ANSWER_STEPS = 3
_NO_EVIDENCE_MARKERS = ("知识库中暂未找到可靠资料", "知识库中暂未找到可靠答案")

_SMALL_TALK_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(你好|您好|嗨|哈喽|hello|hi|hey|在吗|在不在|早上好|下午好|晚上好)([！!。，,？?\s]|$)", re.IGNORECASE),
    re.compile(r"^(谢谢|多谢|感谢|拜拜|再见|bye|goodbye)([！!。，,？?\s]|$)", re.IGNORECASE),
    re.compile(r"^(你?是谁|你叫什么|你叫什么名字|介绍一下你|自我介绍一下)"),
    re.compile(r"^(你?能做什么|你?会什么|你?有什么功能|帮助|help|怎么用)"),
    re.compile(r"^(好的|行|可以|ok|okay|知道了|明白)([！!。，,？?\s]|$)", re.IGNORECASE),
]


def _is_small_talk(prompt: str) -> bool:
    """Return True only when the whole message is a pure conversational opener."""
    text = prompt.strip()
    if not text:
        return True
    for pattern in _SMALL_TALK_PATTERNS:
        match = pattern.search(text)
        if match:
            # Anything beyond the greeting (except trailing punctuation/space)
            # means the message also contains a factual question.
            tail = text[match.end():].strip(" \t\n\r！!。，,、；：:？?\"'")
            if not tail:
                return True
    return False


def _strip_unverified_urls(text: str, allowed_urls: set[str]) -> str:
    """Remove URLs not present in ``allowed_urls`` to prevent hallucinated links."""

    def replace(match: re.Match) -> str:
        url = match.group(0)
        # Allow exact matches or URLs that share a prefix with an allowed URL.
        if url in allowed_urls:
            return url
        for allowed in allowed_urls:
            if url.startswith(allowed) or allowed.startswith(url):
                return url
        return ""

    return re.sub(r"https?://[^\s<>()，。；：）]+", replace, text)


def _strip_source_appendix(text: str) -> str:
    """Remove a trailing source section from the user-visible direct answer."""
    return re.sub(
        r"(?ms)\n+(?:#{1,6}\s*)?(?:参考来源|来源|出处|参考资料)\s*[：:].*$",
        "",
        text,
    ).strip()


def _is_simple_query(prompt: str) -> bool:
    compact = re.sub(r"\s+", "", prompt)
    detail_markers = (
        "详细",
        "完整",
        "系统",
        "深入",
        "全面",
        "逐条",
        "对比",
        "分析",
        "展开",
    )
    return len(compact) <= 48 and not any(marker in compact for marker in detail_markers)


def _query_wants_structure(prompt: str) -> bool:
    markers = ("列出", "列表", "清单", "有哪些", "分别", "步骤", "条件", "要求")
    return any(marker in prompt for marker in markers)


def _needs_style_rewrite(prompt: str, text: str) -> bool:
    """Detect report-like formatting that violates the conversational contract."""
    heading_count = len(
        re.findall(
            r"(?m)^\s*(?:#{1,6}\s+|\*\*[^*\n]{2,24}\*\*\s*$)",
            text,
        )
    )
    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)、])\s+", text))
    report_phrases = (
        "直接回答",
        "核心定位",
        "几个关键特点",
        "容易误解的地方",
        "组织方式上",
        "一个关键张力",
        "从几个层面说",
    )
    report_like = any(phrase in text for phrase in report_phrases)
    if _is_simple_query(prompt):
        unnecessary_list = bullet_count >= 2 and not _query_wants_structure(prompt)
        return heading_count > 0 or unnecessary_list or len(text) > 520 or report_like
    return heading_count >= 2 or (heading_count > 0 and bullet_count >= 3) or report_like


def _enforce_conversational_shape(prompt: str, text: str) -> str:
    """Last-resort formatting guard when a style-only rewrite still ignores rules."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        cleaned = block.strip()
        if not cleaned:
            continue
        if re.fullmatch(r"(?:#{1,6}\s+)?\*\*[^*\n]{2,24}\*\*", cleaned):
            continue
        if re.fullmatch(r"#{1,6}\s+.+", cleaned):
            continue
        lines = []
        for line in cleaned.splitlines():
            if not _query_wants_structure(prompt):
                line = re.sub(r"^\s*(?:[-*+]|\d+[.)、])\s+", "", line)
            line = line.strip()
            line = re.sub(r"^\*\*([^*]+)\*\*[：:]\s*", r"\1：", line)
            if line:
                lines.append(line)
        cleaned = " ".join(lines).strip()
        if not cleaned or cleaned.startswith(("如果你还想", "如果你想继续", "如果你在想")):
            continue
        paragraphs.append(cleaned)

    if _is_simple_query(prompt):
        paragraphs = paragraphs[:3]
        result = "\n\n".join(paragraphs)
        if len(result) > 420:
            shortened = result[:420]
            boundary = max(shortened.rfind(mark) for mark in "。！？")
            result = shortened[: boundary + 1] if boundary >= 120 else shortened.rstrip()
        return result.strip()
    return "\n\n".join(paragraphs).strip()


def _evidence_content_key(content: str) -> str:
    """Return a normalized form of evidence content for deduplication."""
    lines = content.splitlines()
    stripped = [re.sub(r"^\s*\d+[:.\s]+", "", line).strip() for line in lines]
    collapsed = re.sub(r"\s+", " ", " ".join(stripped)).strip()
    return collapsed.casefold()


def _evidence_overlap(a: EvidenceExcerpt, b: EvidenceExcerpt) -> bool:
    """Return True when two excerpts cover an overlapping line range in the same source."""
    if a.line_start is None or b.line_start is None:
        return False
    a_end = a.line_end if a.line_end is not None else a.line_start
    b_end = b.line_end if b.line_end is not None else b.line_start
    return (
        (a.document_id or a.file_path or a.url)
        == (b.document_id or b.file_path or b.url)
        and max(a.line_start, b.line_start) <= min(a_end, b_end)
    )


def _merge_excerpt_contents(a: EvidenceExcerpt, b: EvidenceExcerpt) -> str:
    """Merge two overlapping excerpts by uniting their full lines.

    Lines prefixed with ``N: `` are sorted by line number; other lines keep
    their original order and are deduplicated.
    """
    def _iter(content: str):
        for line in content.splitlines():
            match = re.match(r"^(\d+):\s?(.*)$", line)
            if match:
                yield int(match.group(1)), line
            else:
                yield None, line

    a_items = list(_iter(a.content))
    b_items = list(_iter(b.content))
    all_have_numbers = all(n is not None for n, _ in a_items + b_items)

    seen: set[Any] = set()
    combined: list[tuple[int | None, str]] = []
    for num, line in a_items + b_items:
        key = (num, _evidence_content_key(line)) if num is not None else _evidence_content_key(line)
        if key in seen:
            continue
        seen.add(key)
        combined.append((num, line))

    if all_have_numbers:
        combined.sort(key=lambda item: (item[0], 0))
    return "\n".join(line for _, line in combined)


def _copy_excerpt_metadata(target: EvidenceExcerpt, source: EvidenceExcerpt) -> None:
    """Copy content and metadata from ``source`` to ``target`` while keeping id."""
    target.document_id = source.document_id
    target.title = source.title
    target.url = source.url
    target.file_path = source.file_path
    target.content = source.content
    target.evidence_type = source.evidence_type
    target.qa_status = source.qa_status
    target.historical = source.historical
    target.score = source.score
    target.applicable_years = source.applicable_years
    target.applicable_cohorts = source.applicable_cohorts
    target.document_year = source.document_year
    target.version_status = source.version_status
    target.historical_reason = source.historical_reason


def _merge_excerpts(existing: EvidenceExcerpt, new: EvidenceExcerpt) -> EvidenceExcerpt | None:
    """Attempt to merge ``new`` into ``existing``.

    Returns the merged ``existing`` when a reliable merge is possible, otherwise
    ``None`` to signal that both excerpts should be kept.
    """
    existing_norm = _evidence_content_key(existing.content)
    new_norm = _evidence_content_key(new.content)

    if existing_norm in new_norm:
        # ``new`` truly contains ``existing``: keep the container but reuse the
        # existing evidence id so citations stay stable.
        existing.line_start = (
            new.line_start
            if new.line_start is not None
            else existing.line_start
        )
        existing.line_end = (
            new.line_end
            if new.line_end is not None
            else existing.line_end
        )
        _copy_excerpt_metadata(existing, new)
        return existing

    if new_norm in existing_norm:
        # ``existing`` already contains ``new``.
        return existing

    # Partial overlap: merge full lines.  If the line ranges do not actually
    # share any lines, do not force a merge.
    if existing.line_start is None or new.line_start is None:
        return None
    existing_end = existing.line_end if existing.line_end is not None else existing.line_start
    new_end = new.line_end if new.line_end is not None else new.line_start
    if max(existing.line_start, new.line_start) > min(existing_end, new_end):
        return None

    merged_content = _merge_excerpt_contents(existing, new)
    existing.content = merged_content
    existing.line_start = min(existing.line_start, new.line_start)
    existing.line_end = max(existing_end, new_end)
    existing.applicable_years = sorted(
        set((existing.applicable_years or []) + (new.applicable_years or []))
    ) or None
    existing.applicable_cohorts = sorted(
        set((existing.applicable_cohorts or []) + (new.applicable_cohorts or []))
    ) or None
    if new.document_year is not None:
        existing.document_year = new.document_year
    existing.version_status, existing.historical_reason = classify_version_status(
        existing.title,
        existing.file_path or None,
        existing.content,
        existing.applicable_years,
        existing.document_year,
    )
    existing.historical = existing.version_status in {"historical", "archived"}
    return existing


def _extract_used_evidence_ids(text: str) -> list[str]:
    """Return ordered evidence ids used by the model, e.g. ['E1','E3']."""
    return list(dict.fromkeys(f"E{m}" for m in re.findall(r"\[E(\d+)]", text)))


def _is_pure_no_evidence(text: str) -> bool:
    """Return True when the answer contains no substantive information."""
    cleaned = text
    for marker in _NO_EVIDENCE_MARKERS:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"[^\w一-鿿]", "", cleaned).strip()
    return not cleaned


@dataclass
class SourceTracker:
    """Tracks candidate sources and concrete evidence excerpts.

    * candidate_sources  -- search/grep results used to locate documents.
    * evidence_excerpts  -- actual text read by the model (the only thing that
                            may ground a factual answer).
    * selected_excerpts  -- excerpts chosen for the final answer prompt.
    """

    candidate_sources: list[SearchResult] = field(default_factory=list)
    evidence_excerpts: list[EvidenceExcerpt] = field(default_factory=list)
    selected_excerpts: list[EvidenceExcerpt] = field(default_factory=list)
    read_sources: set[str] = field(default_factory=set)
    verified_urls: set[str] = field(default_factory=set)
    diagnostics: bool = False

    def reset(self) -> None:
        self.candidate_sources.clear()
        self.evidence_excerpts.clear()
        self.selected_excerpts.clear()
        self.read_sources.clear()
        self.verified_urls.clear()

    def record_urls(self, content: str) -> None:
        self.verified_urls.update(re.findall(r"https?://[^\s<>()，。；：]+", content))

    def add_candidates(self, results: list[SearchResult]) -> None:
        """Register search results as candidates, not as final evidence."""
        by_id: dict[str, int] = {
            item.document.yuque_id: i
            for i, item in enumerate(self.candidate_sources)
            if item.document.yuque_id
        }
        for item in results:
            content = (
                item.chunk.content_snippet if item.chunk else item.document.body
            )
            if content:
                self.record_urls(content)
            if not item.document.yuque_id:
                self.candidate_sources.append(item)
                continue
            idx = by_id.get(item.document.yuque_id)
            if idx is None:
                self.candidate_sources.append(item)
                by_id[item.document.yuque_id] = len(self.candidate_sources) - 1
            else:
                existing = self.candidate_sources[idx]
                if item.score > existing.score:
                    self.candidate_sources[idx] = item
                    by_id[item.document.yuque_id] = idx

    def add_grep_hits(self, hits: list[dict], query_terms: list[str]) -> None:
        """Compatibility shim: grep hits are candidate evidence only."""
        from .evidence import grep_hits_to_search_results

        self.add_candidates(grep_hits_to_search_results(hits, query_terms))

    def add_read_document(
        self,
        document,
        content: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> None:
        """Record a document read as concrete evidence."""
        self.record_urls(content)
        for excerpt in evidence_excerpts_from_read(
            document,
            content[:2400],
            line_start=line_start,
            line_end=line_end,
        ):
            self.add_evidence(excerpt)

    def add_evidence(self, excerpt: EvidenceExcerpt) -> EvidenceExcerpt:
        """Append an evidence excerpt, deduplicating by source + lines + content.

        Exact duplicates are returned without creating a new excerpt.  Overlapping
        excerpts for the same source are merged so that nearby reads do not spam
        the evidence list.
        """
        if not excerpt.evidence_id:
            # Reserve a provisional id; it will only be used if the excerpt is kept.
            excerpt.evidence_id = f"E{len(self.evidence_excerpts) + 1}"

        new_key = (
            excerpt.document_id or excerpt.file_path or excerpt.url,
            excerpt.line_start,
            excerpt.line_end,
            _evidence_content_key(excerpt.content),
        )

        for existing in self.evidence_excerpts:
            existing_key = (
                existing.document_id or existing.file_path or existing.url,
                existing.line_start,
                existing.line_end,
                _evidence_content_key(existing.content),
            )
            if existing_key == new_key:
                return existing

            if _evidence_overlap(existing, excerpt):
                merged = _merge_excerpts(existing, excerpt)
                if merged is not None:
                    return merged

        self.evidence_excerpts.append(excerpt)
        if excerpt.evidence_type != "navigation":
            self.read_sources.add(excerpt.file_path or excerpt.document_id)
        self.record_urls(excerpt.content)
        return excerpt

    @property
    def read_count(self) -> int:
        return len(self.read_sources)


ToolFactory = Callable[[SourceTracker], list[object]]
ToolLoop = Callable[..., Awaitable[object]]


class NjuQaAgent:
    """Two-stage evidence-first Agent."""

    def __init__(
        self,
        context: object,
        tools: ToolFactory,
        tool_loop: ToolLoop | None = None,
        docs_root: Path | None = None,
        index: Any = None,
        diagnostics: bool = False,
    ):
        self.context = context
        self.tools = tools
        self._tool_loop = tool_loop
        self.docs_root = docs_root
        self.index = index
        self.diagnostics = diagnostics

    async def answer(
        self,
        event: object,
        prompt: str,
        tracker: SourceTracker | None = None,
        *,
        base_system_prompt: str = "",
        contexts: list[dict[str, str]] | None = None,
        include_sources: bool = True,
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        logger.info("NOVA agent start: provider=%s prompt=%r", provider_id, prompt)
        if not provider_id:
            logger.warning("NOVA agent: no chat provider configured")
            return NO_PROVIDER

        if _is_small_talk(prompt):
            return await self._answer_small_talk(
                event, prompt, base_system_prompt, contexts or []
            )

        if tracker is None:
            tracker = SourceTracker()
        tracker.diagnostics = self.diagnostics
        await self._research_phase(
            event,
            prompt,
            tracker,
            base_system_prompt,
            contexts or [],
        )

        if not tracker.evidence_excerpts:
            logger.info("NOVA agent: no evidence excerpts after research")
            return NO_EVIDENCE

        self._log_evidence_summary(tracker)
        return await self._answer_phase(
            event,
            prompt,
            tracker,
            base_system_prompt,
            contexts or [],
            include_sources,
        )

    async def _answer_small_talk(
        self,
        event: object,
        prompt: str,
        base_system_prompt: str,
        contexts: list[dict[str, str]],
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=f"{base_system_prompt}\n\n{SMALL_TALK_SYSTEM_PROMPT}",
                contexts=contexts,
                tracker=SourceTracker(),
            )
        except Exception:
            logger.exception("NOVA agent small-talk failed")
            return AGENT_ERROR
        text = str(getattr(response, "completion_text", "")).strip()
        logger.info("NOVA agent small-talk: length=%d", len(text))
        return text

    async def _research_phase(
        self,
        event: object,
        prompt: str,
        tracker: SourceTracker,
        base_system_prompt: str = "",
        contexts: list[dict[str, str]] | None = None,
    ) -> None:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        if self.diagnostics:
            logger.info(
                "NOVA agent research start: prompt=%r tools=%s",
                prompt,
                [getattr(t, "name", "") for t in self.tools(tracker)],
            )
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=f"{base_system_prompt}\n\n{RESEARCH_SYSTEM_PROMPT}",
                contexts=contexts or [],
                tracker=tracker,
                max_steps=_MAX_RESEARCH_STEPS,
            )
        except Exception:
            logger.exception("NOVA agent research phase failed")
            return
        # Ignore any natural-language answer produced during research; only the
        # evidence recorded by the tools matters.
        text = str(getattr(response, "completion_text", "")).strip()
        if self.diagnostics and text:
            logger.info("NOVA agent research discarded text: %d chars", len(text))

    def _select_excerpts(
        self, tracker: SourceTracker, max_excerpts: int = _MAX_EVIDENCE
    ) -> list[EvidenceExcerpt]:
        """Choose the best concrete evidence for the answer prompt."""
        # Prefer direct reads over outlines and navigation summaries.
        type_order = {"read": 0, "details": 1, "outline": 2, "navigation": 3}
        scored = sorted(
            tracker.evidence_excerpts,
            key=lambda e: (
                type_order.get(e.evidence_type, 4),
                -e.score,
                -len(e.content),
            ),
        )
        return scored[:max_excerpts]

    async def _answer_phase(
        self,
        event: object,
        prompt: str,
        tracker: SourceTracker,
        base_system_prompt: str = "",
        contexts: list[dict[str, str]] | None = None,
        include_sources: bool = True,
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        excerpts = self._select_excerpts(tracker)
        tracker.selected_excerpts = excerpts

        if not excerpts:
            logger.info("NOVA agent: no selected excerpts for answer")
            return NO_EVIDENCE

        grounded_prompt = self._build_answer_prompt(prompt, excerpts)
        answer_tools = self._answer_tools(tracker)

        if self.diagnostics:
            logger.info(
                "NOVA agent answer start: selected=%d ids=%s tools=%s",
                len(excerpts),
                [e.evidence_id for e in excerpts],
                [getattr(t, "name", "") for t in answer_tools],
            )

        for attempt in range(2):
            try:
                response = await self._run_tool_loop(
                    event=event,
                    chat_provider_id=provider_id,
                    prompt=grounded_prompt,
                    system_prompt=f"{base_system_prompt}\n\n{ANSWER_SYSTEM_PROMPT}",
                    contexts=contexts or [],
                    tracker=tracker,
                    tools=answer_tools,
                    max_steps=_MAX_ANSWER_STEPS,
                )
            except Exception:
                logger.exception("NOVA agent answer phase failed")
                return AGENT_ERROR
            text = str(getattr(response, "completion_text", "")).strip()
            used_ids = _extract_used_evidence_ids(text)
            if used_ids or _is_pure_no_evidence(text):
                break
            logger.warning(
                "NOVA agent: answer missing evidence markers (attempt %d)", attempt + 1
            )
            grounded_prompt = (
                grounded_prompt
                + "\n\n注意：你刚才的回答没有使用任何 [E#] 标记。"
                "请重新回答，并确保每个事实都附带 [E#] 标记。"
            )
        else:
            logger.error("NOVA agent: answer still missing evidence markers")
            return SAFE_FAILURE

        return self._finalize_answer(
            text,
            excerpts,
            tracker.verified_urls,
            include_sources=include_sources,
        )

    def _build_answer_prompt(
        self, question: str, excerpts: list[EvidenceExcerpt]
    ) -> str:
        parts = [f"请回答原问题：{question}\n"]
        parts.append(
            "你只能使用下面标记的证据回答问题。"
            "若某条证据明确标记 no_answer，只能说明该事项暂无可靠资料，"
            "不能用其他相邻材料推断；但不得因此忽略其他已有正面证据的问题部分。"
        )
        for excerpt in excerpts:
            loc = ""
            if excerpt.line_start is not None:
                end = excerpt.line_end if excerpt.line_end is not None else excerpt.line_start
                loc = f" 位置：第 {excerpt.line_start}—{end} 行"
            note = ""
            if excerpt.historical:
                note = "（历史资料）"
            if excerpt.qa_status == "no_answer":
                note += "（该证据明确说明暂无可靠资料）"

            version_meta: list[str] = []
            if excerpt.version_status:
                version_meta.append(f"版本状态：{excerpt.version_status}")
            if excerpt.document_year is not None:
                version_meta.append(f"文档年份：{excerpt.document_year}")
            if excerpt.applicable_years:
                version_meta.append(
                    f"适用年份：{', '.join(str(y) for y in excerpt.applicable_years)}"
                )
            if excerpt.applicable_cohorts:
                version_meta.append(
                    f"适用年级：{', '.join(excerpt.applicable_cohorts)}"
                )
            if excerpt.historical_reason:
                version_meta.append(f"判定原因：{excerpt.historical_reason}")
            meta_part = ("\n".join(version_meta) + "\n") if version_meta else ""

            header = (
                f"[{excerpt.evidence_id}]\n"
                f"来源：《{excerpt.title}》{loc}{note}\n"
                f"URL：{excerpt.url or 'n/a'}\n"
                f"{meta_part}内容：\n{excerpt.content}"
            )
            parts.append(header)
        return "\n\n".join(parts)

    def _answer_tools(self, tracker: SourceTracker) -> list[object]:
        """Return an empty tool set for the answer phase.

        The answer model must only use the evidence selected during research.
        Opening more tools here would allow new evidence that cannot be cited.
        """
        return []

    def _finalize_answer(
        self,
        text: str,
        excerpts: list[EvidenceExcerpt],
        verified_urls: set[str] | None = None,
        *,
        include_sources: bool = True,
    ) -> str:
        used_ids = _extract_used_evidence_ids(text)
        seen: set[str] = set()
        used: list[EvidenceExcerpt] = []
        for e in excerpts:
            if e.evidence_id in used_ids and e.evidence_id not in seen:
                seen.add(e.evidence_id)
                used.append(e)
        if not used and not _is_pure_no_evidence(text):
            logger.warning("NOVA agent: no evidence markers in final answer")
            return SAFE_FAILURE

        # Strip internal markers from the visible answer.
        visible = re.sub(r"\[E\d+]", "", text).strip()
        if _is_pure_no_evidence(visible):
            logger.info("NOVA agent: no-evidence answer, suppressing citations")
            return visible or NO_EVIDENCE

        # Remove any URLs the model hallucinated; keep URLs that appear in the
        # evidence content or in the cited excerpts.
        allowed_urls: set[str] = set()
        if include_sources:
            allowed_urls.update(verified_urls or set())
            allowed_urls.update(e.url for e in used if e.url)
        visible = _strip_unverified_urls(visible, allowed_urls)

        citations = self._build_citations(used)
        logger.info(
            "NOVA agent: available=%d used=%d citations=%d",
            len(excerpts),
            len(used),
            len(citations),
        )
        if self.diagnostics:
            logger.info(
                "NOVA agent final: available_ids=%s used_ids=%s citation_count=%d",
                [e.evidence_id for e in excerpts],
                used_ids,
                len(citations),
            )
        if not include_sources or not citations:
            return visible
        return f"{visible}\n\n参考来源：\n" + "\n".join(citations)

    def _build_citations(self, used: list[EvidenceExcerpt]) -> list[str]:
        seen: set[str] = set()
        citations: list[str] = []
        for excerpt in sorted(used, key=lambda e: e.evidence_id):
            key = excerpt.document_id or excerpt.url or excerpt.file_path
            if not key or key in seen:
                continue
            seen.add(key)
            citations.append(f"{len(citations) + 1}. 《{excerpt.title}》：{excerpt.url or excerpt.file_path}")
        return citations

    def _log_evidence_summary(self, tracker: SourceTracker) -> None:
        if not self.diagnostics:
            return
        logger.info("NOVA evidence summary: candidates=%d excerpts=%d read=%d",
                    len(tracker.candidate_sources),
                    len(tracker.evidence_excerpts),
                    tracker.read_count)
        for i, cand in enumerate(tracker.candidate_sources[:20], 1):
            logger.info(
                "NJU candidate %d: title=%s path=%s score=%s",
                i,
                getattr(cand.document, "title", "?"),
                getattr(cand.document, "path", "?"),
                cand.score,
            )
        for e in tracker.evidence_excerpts:
            logger.info(
                "NOVA evidence excerpt: id=%s type=%s doc=%s lines=%s:%s chars=%d "
                "qa=%s historical=%s years=%s cohorts=%s doc_year=%s version=%s reason=%s",
                e.evidence_id,
                e.evidence_type,
                e.document_id,
                e.line_start,
                e.line_end,
                len(e.content),
                e.qa_status,
                e.historical,
                e.applicable_years,
                e.applicable_cohorts,
                e.document_year,
                e.version_status,
                e.historical_reason,
            )

    async def _run_tool_loop(
        self, *, tracker: SourceTracker, **kwargs: object
    ) -> object:
        # Allow callers to override the tool set (e.g. answer phase).
        if "tools" not in kwargs:
            kwargs["tools"] = self.tools(tracker)
        max_steps = int(kwargs.pop("max_steps", 12))

        if self._tool_loop is not None:
            return await self._tool_loop(max_steps=max_steps, **kwargs)

        from astrbot.core.agent.tool import ToolSet

        # The real AstrBot tool loop does not accept our internal tracker; it is
        # passed to the tools factory above.
        kwargs.pop("tracker", None)
        return await self.context.tool_loop_agent(
            tools=ToolSet(kwargs.pop("tools")),
            max_steps=max_steps,
            tool_call_timeout=60,
            **kwargs,
        )

    def _read_source_body(self, source: SearchResult, limit: int = 8000) -> str:
        """Return full document body when docs_root is configured, else chunk snippet."""
        if self.docs_root is None or source.document.path is None:
            return (
                source.chunk.content_snippet[:limit]
                if source.chunk
                else source.document.body[:limit]
            )
        try:
            result = read_document_content(
                self.docs_root, str(source.document.path), offset=0, limit=limit
            )
            return result["content"]
        except ValueError:
            return (
                source.chunk.content_snippet[:limit]
                if source.chunk
                else source.document.body[:limit]
            )


_DIRECT_ANSWER_SYSTEM_PROMPT = """你是 NOVA CAC 内容包问答助手。

请直接回答用户，不要执行“研究阶段—证据阶段—回答阶段”的内部证据门控，也不要要求
输出 [E#] 标记。你可以根据问题自行决定是否调用本地知识库工具。在同一次 Agent 调用
中可以连续多轮使用工具：先用向量与关键词混合检索寻找候选，结果不足时改写查询或使用
grep 缩小范围，再按需浏览目录、读取原文；资料足够后直接形成最终回答。工具只是帮助你
寻找资料，不要向用户汇报检索过程。

涉及 NOVA 的事实、制度、活动或理念时，应优先使用工具核对本地文档；资料没有明确写出
答案时，按照内容包中的事实边界自然说明不确定之处，不要套用固定拒答句。普通交流可以
直接回应，不需要为了调用工具而调用工具。简单问题默认不分标题，非必要不用列表；
不要试图在一次回答中把整个主题说清楚，只展开当前问题需要的部分，能用两三段说清楚
就及时收住。最终回答必须自然、口语化，不附参考来源、文档路径、原文链接或引用编号。"""

_STYLE_REWRITE_SYSTEM_PROMPT = """你只负责把已有回答改成 CAC 式的自然聊天表达。

不要重新研究问题，不要增加新事实，也不要提到改写过程。保留原回答中与问题直接相关、
且资料能够支持的核心意思，删掉为了显得完整而添加的铺陈。

强制要求：
- 直接进入回答，不使用标题或粗体小标题。除非原问题明确要求列举、步骤或清单，
  否则不使用列表；简单定义问题禁止列清单。
- 使用两到三个自然短段落，像在群聊里认真解释，不像社团介绍、百科或汇报材料。
- 少堆抽象名词，能用普通说法就不用“核心定位、关键特点、能力培养、组织方式”等套话。
- 不要试图一次把整个主题讲完；简单问题控制在大约 350 个汉字以内。
- 不以“如果你还想了解……可以继续问”这类客服式追问收尾。
- 不输出来源、文档名、路径、链接或引用编号。"""


class NovaCacAgent(NjuQaAgent):
    """Single-pass AstrBot Agent with the ported local retrieval tools."""

    async def answer(
        self,
        event: object,
        prompt: str,
        tracker: SourceTracker | None = None,
        *,
        base_system_prompt: str = "",
        contexts: list[dict[str, str]] | None = None,
        include_sources: bool = False,
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        if not provider_id:
            return NO_PROVIDER

        tracker = tracker or SourceTracker()
        tracker.diagnostics = self.diagnostics
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=(
                    f"{base_system_prompt}\n\n{_DIRECT_ANSWER_SYSTEM_PROMPT}"
                ),
                contexts=contexts or [],
                tracker=tracker,
                max_steps=_MAX_RESEARCH_STEPS,
            )
        except Exception:
            logger.exception("NOVA CAC direct Agent call failed")
            return AGENT_ERROR

        text = str(getattr(response, "completion_text", "") or "").strip()
        if not text:
            return AGENT_ERROR
        text = _strip_unverified_urls(text, set())
        text = _strip_source_appendix(text)
        if _needs_style_rewrite(prompt, text):
            text = await self._rewrite_style(
                event,
                provider_id,
                prompt,
                text,
                base_system_prompt,
            )
            text = _strip_source_appendix(_strip_unverified_urls(text, set()))
            if _needs_style_rewrite(prompt, text):
                text = _enforce_conversational_shape(prompt, text)
        return text or AGENT_ERROR

    async def _rewrite_style(
        self,
        event: object,
        provider_id: str,
        question: str,
        draft: str,
        base_system_prompt: str,
    ) -> str:
        rewrite_prompt = (
            f"<原问题>\n{question}\n</原问题>\n\n"
            f"<待改写回答>\n{draft}\n</待改写回答>"
        )
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=rewrite_prompt,
                system_prompt=(
                    f"{base_system_prompt}\n\n{_STYLE_REWRITE_SYSTEM_PROMPT}"
                ),
                contexts=[],
                tracker=SourceTracker(),
                tools=[],
                max_steps=1,
            )
        except Exception:
            logger.exception("NOVA CAC style rewrite failed")
            return _enforce_conversational_shape(question, draft)
        rewritten = str(getattr(response, "completion_text", "") or "").strip()
        return rewritten or _enforce_conversational_shape(question, draft)
