"""Runtime-level regression tests for the evidence-first Agent."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from nju_qa.agent import (
    NO_EVIDENCE,
    NjuQaAgent,
    SourceTracker,
    _MAX_RESEARCH_STEPS,
    _is_small_talk,
)
from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.tools.documents import ReadDocTool


@dataclass
class Response:
    completion_text: str


class Event:
    unified_msg_origin = "test:session"


class Context:
    def __init__(self, provider="provider"):
        self.provider = provider

    async def get_current_chat_provider_id(self, _umo):
        return self.provider


# ---------------------------------------------------------------------------
# 1. Real AstrBot tool_loop_agent branch must not duplicate max_steps.
# ---------------------------------------------------------------------------


class CapturingContext:
    def __init__(self):
        self.calls = []

    async def get_current_chat_provider_id(self, _umo):
        return "provider"

    async def tool_loop_agent(self, **kwargs):
        self.calls.append(kwargs)
        return Response("research response")


def test_real_tool_loop_agent_branch_no_duplicate_max_steps():
    ctx = CapturingContext()
    agent = NjuQaAgent(ctx, lambda tracker: [], tool_loop=None)
    answer = asyncio.run(agent.answer(Event(), "校园卡在哪里补办？"))
    assert answer == NO_EVIDENCE
    assert len(ctx.calls) == 1
    call = ctx.calls[0]
    assert call["max_steps"] == _MAX_RESEARCH_STEPS
    assert list(call.keys()).count("max_steps") == 1
    assert "tool_call_timeout" in call


# ---------------------------------------------------------------------------
# 2. read_doc line numbers must propagate into EvidenceExcerpt.
# ---------------------------------------------------------------------------


def _make_doc(tmp_path: Path) -> Document:
    path = tmp_path / "card.md"
    path.write_text(
        "## 校园卡\n\n校园卡可在服务中心补办。\n\n## 其他\n\n其他内容。",
        encoding="utf-8",
    )
    return Document(
        yuque_id="card",
        title="校园卡",
        repository="guide",
        namespace="nju/guide",
        slug="card",
        url="https://yuque.test/nju/guide/card",
        created_at="a",
        updated_at="b",
        body=path.read_text(encoding="utf-8"),
        path=Path("card.md"),
    )


def test_read_doc_line_numbers_in_evidence(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    doc = _make_doc(tmp_path)
    index.upsert(doc)

    tracker = SourceTracker()
    read = ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker)
    result = asyncio.run(read._run(file_path="card.md", start_line=1, end_line=3))
    assert "error" not in result

    assert len(tracker.evidence_excerpts) == 1
    excerpt = tracker.evidence_excerpts[0]
    assert excerpt.line_start == 1
    assert excerpt.line_end == 3
    assert excerpt.evidence_type == "read"


# ---------------------------------------------------------------------------
# 3. Answer phase must not expose any tools.
# ---------------------------------------------------------------------------


def test_answer_phase_has_no_tools(tmp_path: Path):
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        if "研究" in kwargs.get("system_prompt", ""):
            return Response("research")
        return Response("补办地点在服务中心 [E1]")

    agent = NjuQaAgent(Context(), lambda tracker: [object()], loop)
    tracker = SourceTracker()
    doc = _make_doc(tmp_path)
    tracker.add_read_document(doc, "校园卡可在服务中心补办")

    answer = asyncio.run(agent._answer_phase(Event(), "校园卡在哪里补办？", tracker))
    assert "服务中心" in answer
    answer_calls = [c for c in calls if "研究" not in c.get("system_prompt", "")]
    assert answer_calls
    assert answer_calls[-1]["tools"] == []


# ---------------------------------------------------------------------------
# 4. Small-talk detection must require the whole message to be small talk.
# ---------------------------------------------------------------------------


def test_small_talk_only_for_pure_greetings():
    assert _is_small_talk("你好")
    assert _is_small_talk("在吗？")
    assert _is_small_talk("你是谁")
    assert _is_small_talk("谢谢！")
    assert not _is_small_talk("你好，校园卡怎么补办？")
    assert not _is_small_talk("请问你知道鼓楼澡堂几点关门吗？")


# ---------------------------------------------------------------------------
# 5. A no_answer excerpt must not veto another positively evidenced subquestion.
# ---------------------------------------------------------------------------


def test_no_answer_does_not_veto_positive_evidence(tmp_path: Path):
    async def loop(**kwargs):
        if "研究" in kwargs.get("system_prompt", ""):
            return Response("research")
        return Response("校园卡可在服务中心补办 [E1]")

    agent = NjuQaAgent(Context(), lambda tracker: [], loop)
    tracker = SourceTracker()
    doc = _make_doc(tmp_path)
    tracker.add_read_document(doc, "校园卡可在服务中心补办")
    tracker.add_read_document(
        Document(
            yuque_id="na",
            title="澡堂",
            repository="guide",
            namespace="nju/guide",
            slug="bath",
            url="",
            created_at="a",
            updated_at="b",
            body="暂无",
            path=Path("bath.md"),
        ),
        "知识库中暂未找到可靠资料",
    )

    answer = asyncio.run(agent._answer_phase(Event(), "校园卡和澡堂", tracker))
    assert answer != NO_EVIDENCE
    assert "服务中心" in answer


# ---------------------------------------------------------------------------
# 6. Diagnostics=True logs evidence summary without full text.
# ---------------------------------------------------------------------------


def test_diagnostics_logs_evidence_summary(caplog, tmp_path: Path):
    async def loop(**kwargs):
        if "研究" in kwargs.get("system_prompt", ""):
            return Response("research")
        return Response("补办地点在服务中心 [E1]")

    agent = NjuQaAgent(
        Context(), lambda tracker: [], loop, diagnostics=True
    )
    tracker = SourceTracker()
    doc = _make_doc(tmp_path)
    tracker.add_read_document(doc, "校园卡可在服务中心补办")

    with caplog.at_level(logging.INFO, logger="astrbot"):
        asyncio.run(agent.answer(Event(), "校园卡在哪里补办？", tracker=tracker))

    logs = caplog.text
    assert "NOVA evidence summary" in logs
    assert "NOVA agent final" in logs
    assert "available_ids=" in logs or "used_ids=" in logs
    # Full document body should not be printed.
    assert "校园卡可在服务中心补办" not in logs


def test_nova_pack_and_cac_context_reach_both_agent_phases(tmp_path: Path):
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        if "研究" in kwargs.get("system_prompt", ""):
            return Response("research")
        return Response("NOVA 的事实 [E1]")

    agent = NjuQaAgent(Context(), lambda tracker: [], loop)
    tracker = SourceTracker()
    tracker.add_read_document(_make_doc(tmp_path), "NOVA 的事实")
    contexts = [{"role": "user", "content": "上一轮 /cac 问题"}]
    answer = asyncio.run(
        agent.answer(
            Event(),
            "那这个呢？",
            tracker=tracker,
            base_system_prompt="AGENTS\nSOUL\nSPIRIT\nVOICE",
            contexts=contexts,
            include_sources=False,
        )
    )

    assert len(calls) == 2
    assert all("AGENTS\nSOUL\nSPIRIT\nVOICE" in call["system_prompt"] for call in calls)
    assert all(call["contexts"] == contexts for call in calls)
    assert "参考来源" not in answer
