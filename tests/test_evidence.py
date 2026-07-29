"""Evidence tracking, grep reliability, and evidence-first Agent tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nju_qa.agent import NO_EVIDENCE, NjuQaAgent, SourceTracker
from nju_qa.document_index import DocumentIndex
from nju_qa.evidence import (
    EvidenceExcerpt,
    build_chunk_from_grep_hit,
    build_document_from_grep_hit,
    document_from_index_row,
    evaluate_grep_reliability,
    evidence_excerpt_from_read,
    evidence_excerpt_from_text,
    grep_hits_to_search_results,
    score_grep_hit,
)
from nju_qa.models import Document
from nju_qa.tools.documents import (
    GetDocDetailsTool,
    GrepLocalDocsTool,
    ReadDocTool,
)


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


def _index_with_docs(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    docs = [
        _doc(
            tmp_path,
            "card.md",
            "校园卡补办指南",
            "## 校园卡的挂失与补办\n\n"
            "校园卡丢失后可在信息化建设管理服务中心一楼大厅补办。\n"
            "鼓楼校区请到综合服务大厅办理。补卡费用为 20 元。",
            "card123",
        ),
        _doc(
            tmp_path,
            "photo.md",
            "校园卡照片采集",
            "## 照片采集\n\n新生入学后需要上传校园卡照片。",
            "photo456",
        ),
        _doc(
            tmp_path,
            "dorm.md",
            "宿舍分配与入住",
            "## 宿舍\n\n新生报到后按学院分配宿舍。",
            "dorm789",
        ),
        _doc(
            tmp_path,
            "00_index.md",
            "00_index",
            "## 目录\n\n- 校园卡\n- 宿舍\n- 教务",
            "idx000",
        ),
        _doc(
            tmp_path,
            "id_card.md",
            "学生证补办",
            "## 学生证补办\n\n学生证丢失后由学院教务处负责补办。",
            "idcard321",
        ),
    ]
    for doc in docs:
        index.upsert(doc)
    return index


def _hit(
    title: str,
    yuque_id: str,
    matched_keywords: list[str],
    snippet: str,
    path: str = "doc.md",
) -> dict:
    return {
        "yuque_id": yuque_id,
        "title": title,
        "repository": "guide",
        "namespace": "nju/guide",
        "slug": Path(path).stem,
        "url": f"https://yuque.test/nju/guide/{Path(path).stem}",
        "created_at": "a",
        "updated_at": "b",
        "path": path,
        "matched_keywords": matched_keywords,
        "matches": [{"line_start": 1, "line_end": 1, "snippet": snippet}],
    }


@pytest.fixture
def sample_hits():
    return {
        "direct": _hit(
            "校园卡补办指南",
            "card123",
            ["校园卡", "补办"],
            "校园卡的挂失与补办地点如下：仙林校区信息化建设管理服务中心一楼大厅。",
            "card.md",
        ),
        "photo": _hit(
            "校园卡照片采集",
            "photo456",
            ["校园卡"],
            "新生入学后需要上传校园卡照片。",
            "photo.md",
        ),
        "dorm": _hit(
            "宿舍分配与入住",
            "dorm789",
            ["校园卡"],
            "新生报到后按学院分配宿舍。",
            "dorm.md",
        ),
        "index": _hit(
            "00_index",
            "idx000",
            ["校园卡", "补办"],
            "- 校园卡\n- 宿舍\n- 教务",
            "00_index.md",
        ),
        "id_card": _hit(
            "学生证补办",
            "idcard321",
            ["补办"],
            "学生证丢失后由学院教务处负责补办。",
            "id_card.md",
        ),
    }


def test_evidence_excerpt_factories():
    doc = _doc(Path("/tmp"), "x.md", "t", "body", "1")
    excerpt = evidence_excerpt_from_read(doc, "正文")
    assert excerpt.document_id == "1"
    assert excerpt.title == "t"
    assert excerpt.evidence_type == "read"

    nav = evidence_excerpt_from_text("大纲", title="t", file_path="x.md")
    assert nav.evidence_type == "navigation"
    assert nav.content == "大纲"


def test_source_tracker_assigns_evidence_ids():
    tracker = SourceTracker()
    tracker.add_evidence(EvidenceExcerpt(content="a"))
    tracker.add_evidence(EvidenceExcerpt(content="b"))
    assert [e.evidence_id for e in tracker.evidence_excerpts] == ["E1", "E2"]


def test_grep_reliability_direct_answer_is_reliable(sample_hits):
    reliable, diag = evaluate_grep_reliability(
        sample_hits["direct"], ["校园卡", "补办"]
    )
    assert reliable is True
    assert diag["core_coverage"] == 1.0
    assert diag["same_window_core_coverage"] is True


def test_grep_reliability_partial_keyword_is_not_reliable(sample_hits):
    for key in ("photo", "dorm", "id_card"):
        reliable, _ = evaluate_grep_reliability(sample_hits[key], ["校园卡", "补办"])
        assert reliable is False, key


def test_grep_reliability_index_is_not_reliable_even_with_full_coverage(sample_hits):
    reliable, diag = evaluate_grep_reliability(sample_hits["index"], ["校园卡", "补办"])
    assert reliable is False
    assert diag["is_index_document"] is True


def test_grep_reliability_no_answer_status_is_unreliable():
    hit = _hit(
        "问题 QA",
        "qa1",
        ["校园卡", "补办"],
        "status: no_answer_found\n校园卡补办地点暂无。",
    )
    reliable, diag = evaluate_grep_reliability(hit, ["校园卡", "补办"])
    assert reliable is False
    assert diag["qa_status"] == "no_answer"


def test_grep_reliability_resolved_status_is_reliable():
    hit = _hit(
        "问题 QA",
        "qa2",
        ["校园卡", "补办"],
        "status: resolved\n校园卡可在信息化建设管理服务中心补办。",
    )
    reliable, _ = evaluate_grep_reliability(hit, ["校园卡", "补办"])
    assert reliable is True


def test_grep_ranking_direct_answer_wins(sample_hits):
    terms = ["校园卡", "补办"]
    scores = {k: score_grep_hit(v, terms) for k, v in sample_hits.items()}
    assert scores["direct"] > scores["photo"]
    assert scores["direct"] > scores["dorm"]
    assert scores["direct"] > scores["index"]
    assert scores["direct"] > scores["id_card"]


def test_grep_same_window_coverage_outranks_split_match():
    split = {
        **_hit(
            "分散命中",
            "split1",
            ["校园卡", "补办"],
            "第一章介绍校园卡。",
        ),
        "matches": [
            {"line_start": 1, "line_end": 1, "snippet": "第一章介绍校园卡。"},
            {"line_start": 10, "line_end": 10, "snippet": "学生证可以补办。"},
        ],
    }
    together = _hit(
        "同窗口命中",
        "together1",
        ["校园卡", "补办"],
        "校园卡的挂失与补办地点如下。",
    )
    terms = ["校园卡", "补办"]
    assert score_grep_hit(together, terms) > score_grep_hit(split, terms)
    _, diag_together = evaluate_grep_reliability(together, terms)
    _, diag_split = evaluate_grep_reliability(split, terms)
    assert diag_together["same_window_core_coverage"] is True
    assert diag_split["same_window_core_coverage"] is False


def test_grep_hits_convert_to_search_results():
    hits = [
        _hit("校园卡补办指南", "c1", ["校园卡", "补办"], "校园卡可补办。", "c.md"),
        _hit("校园卡照片采集", "c2", ["校园卡"], "照片采集。", "p.md"),
    ]
    results = grep_hits_to_search_results(hits, ["校园卡", "补办"])
    assert len(results) == 2
    assert results[0].document.yuque_id == "c1"
    assert results[0].reliable is True
    assert results[1].reliable is False


def test_source_tracker_adds_grep_candidates(sample_hits):
    tracker = SourceTracker()
    tracker.add_candidates(grep_hits_to_search_results(
        [sample_hits["direct"], sample_hits["photo"]], ["校园卡", "补办"]
    ))
    assert len(tracker.candidate_sources) == 2
    assert any(s.document.yuque_id == "card123" for s in tracker.candidate_sources)


def test_source_tracker_deduplicates_candidates_by_yuque_id():
    tracker = SourceTracker()
    doc = build_document_from_grep_hit(
        _hit("校园卡补办指南", "c1", ["校园卡", "补办"], " snippet ", "c.md")
    )
    chunk = build_chunk_from_grep_hit(
        _hit("校园卡补办指南", "c1", ["校园卡", "补办"], " snippet ", "c.md"),
        ["校园卡", "补办"],
    )
    from nju_qa.models import SearchResult
    tracker.add_candidates(
        [
            SearchResult("G1", doc, 0.5, chunk=chunk, reliable=False, retrieval_methods=("grep",)),
            SearchResult("S1", doc, 0.9, chunk=chunk, reliable=True, retrieval_methods=("keyword",)),
        ]
    )
    assert len(tracker.candidate_sources) == 1
    assert tracker.candidate_sources[0].score == 0.9


def test_document_from_index_row():
    index = DocumentIndex(Path("/nonexistent") / "x.sqlite3")
    doc = Document(
        "1", "t", "r", "n", "s", "u", "a", "b", "body", path=Path("p.md")
    )
    index.upsert(doc)
    row = index.all_documents()[0]
    rebuilt = document_from_index_row(row, "new body")
    assert rebuilt.yuque_id == "1"
    assert rebuilt.body == "new body"


def test_grep_tool_registers_candidates(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )
    result = asyncio.run(tool._run("校园卡 补办"))
    assert result["count"] >= 1
    assert len(tracker.candidate_sources) >= 1
    assert any(s.document.yuque_id == "card123" for s in tracker.candidate_sources)


def test_grep_required_phrases_filters_results(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(index=index, docs_root=tmp_path, tracker=tracker)
    result = asyncio.run(tool._run("校园卡", required_phrases="鼓楼 20元"))
    assert result["count"] == 0
    result = asyncio.run(tool._run("校园卡", required_phrases="补办 20"))
    assert result["count"] >= 1


def test_read_doc_records_evidence_excerpt(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker)
    result = asyncio.run(tool._run(file_path="card.md"))
    assert "content" in result
    assert len(tracker.evidence_excerpts) == 1
    assert tracker.evidence_excerpts[0].document_id == "card123"
    assert "card.md" in tracker.read_sources


def test_get_doc_details_content_registers_evidence(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GetDocDetailsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )

    result = asyncio.run(tool._run(yuque_id="card123", include_content=True))
    assert "content" in result
    assert len(tracker.evidence_excerpts) == 1
    assert tracker.evidence_excerpts[0].document_id == "card123"


def test_get_doc_details_without_content_does_not_register_evidence(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GetDocDetailsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )

    result = asyncio.run(tool._run(yuque_id="card123", include_content=False))
    assert "results" in result
    assert len(tracker.evidence_excerpts) == 0
    assert tracker.read_count == 0


def test_empty_grep_does_not_create_fake_sources(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )
    result = asyncio.run(tool._run("完全不存在的词"))
    assert result["count"] == 0
    assert len(tracker.candidate_sources) == 0


def test_grep_hit_with_missing_path_is_handled():
    hit = _hit("t", "id", ["校园卡"], "snippet")
    hit.pop("path")
    doc = build_document_from_grep_hit(hit)
    assert doc.path is None
    assert doc.yuque_id == "id"


def test_grep_hit_with_empty_matches_is_unreliable():
    hit = _hit("t", "id", ["校园卡"], "")
    hit["matches"] = []
    reliable, _ = evaluate_grep_reliability(hit, ["校园卡"])
    assert reliable is False


def test_grep_fallback_to_bigram_terms(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )
    result = asyncio.run(tool._run("信息建设中"))
    assert isinstance(result, dict)
    assert "count" in result


def test_agent_returns_no_evidence_without_read_content():
    async def loop(**kwargs):
        return _Response("校园卡可在中心补办。")

    agent = NjuQaAgent(_Context(), lambda tracker: [], loop)
    answer = asyncio.run(agent.answer(_Event(), "校园卡在哪里补办？"))
    assert answer == NO_EVIDENCE


def test_agent_answer_uses_only_read_evidence_and_strips_markers(tmp_path):
    index = _index_with_docs(tmp_path)
    index  # ensure index created in tmp

    def tool_factory(tracker):
        return [
            GrepLocalDocsTool(index=index, docs_root=tmp_path, tracker=tracker),
            ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker),
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        system_prompt = kwargs.get("system_prompt", "")
        if "研究" in system_prompt:
            read = next(t for t in tools if t.name == "read_doc")
            await read._run(file_path="card.md")
            return _Response("research done")
        return _Response("校园卡可在信息化建设管理服务中心补办 [E1]。")

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=tmp_path)
    answer = asyncio.run(agent.answer(_Event(), "校园卡在哪里补办？"))
    assert "信息化建设管理服务中心" in answer
    assert "[E1]" not in answer
    assert "参考来源" in answer
    assert "https://yuque.test/nju/guide/card" in answer

    index.close()


def test_agent_strips_fake_url_and_keeps_verified_url(tmp_path):
    index = _index_with_docs(tmp_path)

    def tool_factory(tracker):
        return [
            GrepLocalDocsTool(index=index, docs_root=tmp_path, tracker=tracker),
            ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker),
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        system_prompt = kwargs.get("system_prompt", "")
        if "研究" in system_prompt:
            read = next(t for t in tools if t.name == "read_doc")
            await read._run(file_path="card.md")
            return _Response("research done")
        return _Response("校园卡可在中心补办 [E1]，详情见 https://fake.test。")

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=tmp_path)
    answer = asyncio.run(agent.answer(_Event(), "校园卡在哪里补办？"))
    assert "https://fake.test" not in answer
    assert "https://yuque.test/nju/guide/card" in answer

    index.close()


def test_agent_suppresses_verified_url_when_sources_were_not_requested(tmp_path):
    index = _index_with_docs(tmp_path)

    def tool_factory(tracker):
        return [
            GrepLocalDocsTool(index=index, docs_root=tmp_path, tracker=tracker),
            ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker),
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        system_prompt = kwargs.get("system_prompt", "")
        if "研究" in system_prompt:
            read = next(t for t in tools if t.name == "read_doc")
            await read._run(file_path="card.md")
            return _Response("research done")
        return _Response(
            "校园卡可在中心补办 [E1]，详情见 "
            "https://yuque.test/nju/guide/card。"
        )

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=tmp_path)
    answer = asyncio.run(
        agent.answer(
            _Event(),
            "校园卡在哪里补办？",
            include_sources=False,
        )
    )
    assert "校园卡可在中心补办" in answer
    assert "https://" not in answer
    assert "参考来源" not in answer

    index.close()
