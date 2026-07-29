"""Approximate-name handling: users may use names close to the KB entry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nju_qa.agent import NO_EVIDENCE, NjuQaAgent
from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.tools.documents import GrepLocalDocsTool, ReadDocTool


class _Context:
    async def get_current_chat_provider_id(self, _):
        return "provider"


class _Response:
    def __init__(self, text: str):
        self.completion_text = text


class _Event:
    unified_msg_origin = "u"


def _doc(
    tmp_path: Path,
    rel: str,
    title: str,
    body: str,
    yuque_id: str,
    namespace: str = "nju/guide",
    repository: str = "guide",
) -> Document:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return Document(
        yuque_id=yuque_id,
        title=title,
        repository=repository,
        namespace=namespace,
        slug=Path(rel).stem,
        url=f"https://yuque.test/{namespace}/{Path(rel).stem}",
        created_at="a",
        updated_at="b",
        body=body,
        path=Path(rel),
    )


@pytest.fixture
def kaijia_index(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    docs = [
        _doc(
            tmp_path,
            "academy/开甲书院.md",
            "开甲书院",
            "## 开甲书院大一培养方案\n\n"
            "开甲书院大一学生主要修读通识教育课程、高等数学、程序设计基础等。",
            "kaijia",
            namespace="nju/guide",
        ),
        _doc(
            tmp_path,
            "academy/计算机学院.md",
            "计算机学院",
            "## 计算机学院\n\n计算机学院开设计算机科学与技术专业。",
            "cs",
            namespace="nju/guide",
        ),
    ]
    for doc in docs:
        index.upsert(doc)
    return index, tmp_path


def test_approximate_name_falls_back_to_stable_keyword(kaijia_index):
    """“开甲学院” should resolve to the real KB entry “开甲书院”."""
    index, root = kaijia_index

    def tool_factory(tracker):
        return [
            GrepLocalDocsTool(index=index, docs_root=root, tracker=tracker),
            ReadDocTool(index=index, docs_root=root, tracker=tracker),
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        system_prompt = kwargs.get("system_prompt", "")

        # Research phase: collect evidence.
        if "研究" in system_prompt:
            grep = next(t for t in tools if t.name == "grep_local_docs")
            read = next(t for t in tools if t.name == "read_doc")
            first = await grep._run("开甲学院 大一 学什么")
            hits = first["results"]
            if not hits:
                second = await grep._run("开甲")
                hits = second["results"]
            if hits:
                await read._run(file_path=hits[0]["path"])
            return _Response("research done")

        # Answer phase: produce grounded answer with citation markers.
        return _Response(
            "知识库中的相关正式名称是“开甲书院”。"
            "开甲书院大一主要修读通识教育课程、高等数学和程序设计基础 [E1]。"
        )

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=root)
    answer = asyncio.run(agent.answer(_Event(), "开甲学院大一要学什么？"))
    assert "开甲书院" in answer
    assert "通识教育" in answer
    assert "计算机学院" not in answer
    assert "人工智能学院" not in answer
    assert "[E1]" not in answer
    assert "参考来源" in answer


def test_unknown_academy_returns_no_evidence(kaijia_index):
    """A completely unknown entity should not borrow information from other academies."""
    index, root = kaijia_index

    def tool_factory(tracker):
        return [
            GrepLocalDocsTool(index=index, docs_root=root, tracker=tracker),
            ReadDocTool(index=index, docs_root=root, tracker=tracker),
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        system_prompt = kwargs.get("system_prompt", "")

        if "研究" in system_prompt:
            grep = next(t for t in tools if t.name == "grep_local_docs")
            await grep._run("量子魔法书院 大一")
            await grep._run("量子魔法")
            return _Response("research done")

        return _Response("暂未找到相关资料。")

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=root)
    answer = asyncio.run(agent.answer(_Event(), "量子魔法书院大一要学什么？"))
    assert answer == NO_EVIDENCE
    assert "开甲书院" not in answer
    assert "计算机学院" not in answer
