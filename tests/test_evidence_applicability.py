"""Deterministic tests for evidence applicability, version handling, and QA blocks."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from nju_qa.agent import NjuQaAgent, SourceTracker
from nju_qa.document_index import DocumentIndex
from nju_qa.evidence import (
    EvidenceExcerpt,
    QaEvidenceStatus,
    classify_qa_window,
    classify_version_status,
    evidence_excerpt_from_text,
    evidence_excerpts_from_read,
    extract_applicable_cohorts,
    extract_applicable_years,
    extract_document_year,
)
from nju_qa.knowledge_structure import (
    _namespace_matches,
    build_knowledge_base_summaries,
    normalize_namespace,
)
from nju_qa.models import Document
from nju_qa.tools.documents import (
    GetDocDetailsTool,
    ReadDocTool,
    _clamp_read_range,
    _recompute_line_end,
    _truncate_read_result,
    _row_matches_scope,
)


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


def _index_with_docs(tmp_path: Path, docs: list[Document]) -> DocumentIndex:
    index = DocumentIndex(tmp_path / "index.sqlite3")
    for doc in docs:
        index.upsert(doc)
    return index


# ---------------------------------------------------------------------------
# Version / applicability metadata extraction
# ---------------------------------------------------------------------------


def test_extract_applicable_years():
    assert extract_applicable_years("适用于 2024 级学生") == [2024]
    assert extract_applicable_years("2023-2025 学年有效") == [2023, 2024, 2025]
    assert extract_applicable_years("2023, 2024 和 2025 级") == [2023, 2024, 2025]
    assert extract_applicable_years("没有年份") == []


def test_extract_applicable_cohorts():
    assert extract_applicable_cohorts("2024级入学") == ["2024级"]
    assert extract_applicable_cohorts("2023 届毕业生") == ["2023届"]
    assert set(extract_applicable_cohorts("2021级与 2021 级")) == {"2021级"}
    assert extract_applicable_cohorts("无届别") == []


def test_extract_document_year():
    assert extract_document_year("培养方案 2024") == 2024
    assert extract_document_year("指南", path="docs/2023/guide.md") == 2023
    assert extract_document_year("指南") is None


def test_classify_version_status():
    assert classify_version_status("", "归档/old.md", "", None, None)[0] == "archived"
    assert classify_version_status("旧版手册", "x.md", "", None, None)[0] == "historical"
    assert (
        classify_version_status("2024 培养方案", "x.md", "", [2024], 2024)[0]
        == "historical"
    )
    assert (
        classify_version_status(
            "2024 培养方案", "x.md", "本条例为最新版本", [2024], 2024
        )[0]
        == "current"
    )
    assert (
        classify_version_status("培养方案", "x.md", "2024级适用", [2024], None)[0]
        == "historical"
    )
    assert classify_version_status("培养方案", "x.md", "", None, None)[0] == "unknown"


def test_version_no_accidental_year_in_body():
    body = "该通知发布于2024年，请同学们留意。"
    assert extract_applicable_years(body) == []
    assert extract_applicable_cohorts(body) == []
    status, reason = classify_version_status("通知", None, body, [], None)
    assert status == "unknown"


def test_version_year_title_not_current_from_body_current_word():
    body = "当前学生请关注本方案。"
    status, reason = classify_version_status("2024 培养方案", None, body, [2024], 2024)
    assert status == "historical"


def test_version_explicit_current_phrase_makes_current():
    body = "本条例为最新版本，适用于全体本科生。"
    status, reason = classify_version_status("2024 培养方案", None, body, [2024], 2024)
    assert status == "current"


def test_version_unknown_when_no_version_info():
    status, reason = classify_version_status(
        "校园卡指南", "guide/card.md", "校园卡可在服务中心补办。", [], None
    )
    assert status == "unknown"


def test_answer_prompt_includes_version_metadata():
    """Historical evidence must expose its version metadata in the grounded prompt."""

    class _Ctx:
        async def get_current_chat_provider_id(self, _umo):
            return "provider"

    agent = NjuQaAgent(_Ctx(), lambda tracker: [], None)
    excerpt = EvidenceExcerpt(
        evidence_id="E1",
        title="2024级培养方案",
        file_path="plan.md",
        line_start=1,
        line_end=5,
        content="适用于 2024 级学生。",
        historical=True,
        version_status="historical",
        document_year=2024,
        applicable_years=[2024],
        applicable_cohorts=["2024级"],
        historical_reason="文档标题/路径包含年份 2024",
    )
    prompt = agent._build_answer_prompt("大一学什么？", [excerpt])
    assert "版本状态：historical" in prompt
    assert "文档年份：2024" in prompt
    assert "适用年份：2024" in prompt
    assert "适用年级：2024级" in prompt
    assert "判定原因：" in prompt
    assert "（历史资料）" in prompt


# ---------------------------------------------------------------------------
# QA block splitting
# ---------------------------------------------------------------------------


def test_qa_block_splitting_only_with_status_markers():
    qa = (
        "Q1 校园卡如何补办？\n"
        "status: resolved\n"
        "可在服务中心补办。\n"
        "Q2 宿舍床尺寸\n"
        "status: no_answer_found\n"
        "暂无可靠资料。"
    )
    from nju_qa.evidence import split_qa_blocks

    blocks = split_qa_blocks(qa)
    assert len(blocks) == 2
    assert classify_qa_window(blocks[0]) is QaEvidenceStatus.RELIABLE
    assert classify_qa_window(blocks[1]) is QaEvidenceStatus.NO_ANSWER


def test_qa_blocks_keep_distinct_line_numbers():
    qa = (
        "1: Q1 校园卡如何补办？\n"
        "2: status: resolved\n"
        "3: 可在服务中心补办。\n"
        "4: Q2 宿舍床尺寸\n"
        "5: status: no_answer_found\n"
        "6: 暂无可靠资料。"
    )
    doc = Document(
        yuque_id="qa1",
        title="QA",
        repository="r",
        namespace="n",
        slug="s",
        url="",
        created_at="a",
        updated_at="b",
        body=qa,
        path=Path("qa.md"),
    )
    excerpts = evidence_excerpts_from_read(doc, qa, line_start=1, line_end=6)
    assert len(excerpts) == 2
    resolved = [e for e in excerpts if e.qa_status == "reliable"][0]
    no_answer = [e for e in excerpts if e.qa_status == "no_answer"][0]
    assert resolved.line_start == 1
    assert resolved.line_end == 3
    assert no_answer.line_start == 4
    assert no_answer.line_end == 6
    assert resolved.line_end < no_answer.line_start


def test_ordinary_article_not_split_by_headings():
    article = "## 校园卡\n\n可在服务中心补办。\n\n## 宿舍\n\n按学院分配。"
    from nju_qa.evidence import split_qa_blocks

    blocks = split_qa_blocks(article)
    assert len(blocks) == 1
    assert blocks[0] == article.strip()


def test_evidence_excerpts_from_read_populate_version_metadata(tmp_path: Path):
    body = "2024级学生请按 2024-2025 学年方案执行。"
    doc = _doc(tmp_path, "plan.md", "培养方案", body, "plan1")
    excerpts = evidence_excerpts_from_read(doc, body)
    assert len(excerpts) == 1
    excerpt = excerpts[0]
    assert excerpt.applicable_years == [2024, 2025]
    assert excerpt.applicable_cohorts == ["2024级", "2025学年"]
    assert excerpt.document_year is None
    assert excerpt.version_status == "historical"


# ---------------------------------------------------------------------------
# Evidence deduplication / overlap handling
# ---------------------------------------------------------------------------


def test_source_tracker_dedup_exact_duplicate():
    tracker = SourceTracker()
    a = EvidenceExcerpt(
        file_path="doc.md",
        line_start=1,
        line_end=5,
        content="校园卡可在服务中心补办。",
        evidence_type="read",
    )
    b = EvidenceExcerpt(
        file_path="doc.md",
        line_start=1,
        line_end=5,
        content="校园卡可在服务中心补办。",
        evidence_type="read",
    )
    tracker.add_evidence(a)
    tracker.add_evidence(b)
    assert len(tracker.evidence_excerpts) == 1
    assert tracker.evidence_excerpts[0].evidence_id == "E1"
    assert tracker.read_count == 1


def test_source_tracker_merge_overlapping_substring():
    tracker = SourceTracker()
    a = EvidenceExcerpt(
        file_path="doc.md",
        line_start=1,
        line_end=3,
        content="1: 校园卡\n2: 可在\n3: 服务中心",
        evidence_type="read",
    )
    b = EvidenceExcerpt(
        file_path="doc.md",
        line_start=2,
        line_end=3,
        content="2: 可在\n3: 服务中心",
        evidence_type="read",
    )
    tracker.add_evidence(a)
    tracker.add_evidence(b)
    assert len(tracker.evidence_excerpts) == 1
    merged = tracker.evidence_excerpts[0]
    assert merged.line_start == 1
    assert merged.line_end == 3
    assert "服务中心" in merged.content


def test_source_tracker_merge_partial_overlap_preserves_full_lines():
    tracker = SourceTracker()
    a = EvidenceExcerpt(
        file_path="doc.md",
        line_start=1,
        line_end=3,
        content="1: 开头\n2: 中间\n3: 共享",
        evidence_type="read",
    )
    b = EvidenceExcerpt(
        file_path="doc.md",
        line_start=3,
        line_end=5,
        content="3: 共享\n4: 继续\n5: 结尾",
        evidence_type="read",
    )
    tracker.add_evidence(a)
    tracker.add_evidence(b)
    assert len(tracker.evidence_excerpts) == 1
    merged = tracker.evidence_excerpts[0]
    assert merged.line_start == 1
    assert merged.line_end == 5
    first_line, last_line = merged.content.splitlines()[0], merged.content.splitlines()[-1]
    assert "开头" in first_line
    assert "结尾" in last_line
    assert "1:" in first_line
    assert "5:" in last_line


def test_source_tracker_keeps_separate_non_overlapping_ranges():
    tracker = SourceTracker()
    a = EvidenceExcerpt(
        file_path="doc.md",
        line_start=1,
        line_end=2,
        content="1: 开头",
        evidence_type="read",
    )
    b = EvidenceExcerpt(
        file_path="doc.md",
        line_start=10,
        line_end=11,
        content="10: 结尾",
        evidence_type="read",
    )
    tracker.add_evidence(a)
    tracker.add_evidence(b)
    assert len(tracker.evidence_excerpts) == 2


def test_navigation_evidence_does_not_count_as_read():
    tracker = SourceTracker()
    tracker.add_evidence(
        evidence_excerpt_from_text(
            "知识库列表", title="列表", file_path="", evidence_type="navigation"
        )
    )
    assert len(tracker.evidence_excerpts) == 1
    assert tracker.read_count == 0
    assert len(tracker.read_sources) == 0


# ---------------------------------------------------------------------------
# Namespace normalization and index-category exclusion
# ---------------------------------------------------------------------------


def test_normalize_namespace_unifies_underscore_and_slash():
    assert normalize_namespace("qc19gt_fqpid3") == "qc19gt/fqpid3"
    assert normalize_namespace("qc19gt/fqpid3") == "qc19gt/fqpid3"
    assert normalize_namespace("_qc19gt_fqpid3_") == "qc19gt/fqpid3"


def test_namespace_matches_uses_normalized_segments():
    assert _namespace_matches(
        ["qc19gt", "fqpid3", "doc.md"],
        ["qc19gt", "fqpid3"],
    )
    assert not _namespace_matches(
        ["qc19gt", "other", "doc.md"],
        ["qc19gt", "fqpid3"],
    )


def test_row_matches_scope_with_underscore_namespace(tmp_path: Path):
    _doc(tmp_path, "qc19gt/fqpid3/card.md", "卡片", "正文", "c1")
    row = {
        "path": "qc19gt/fqpid3/card.md",
        "repository": "repo",
        "namespace": "qc19gt/fqpid3",
    }
    assert _row_matches_scope(row, namespace="qc19gt_fqpid3")
    assert not _row_matches_scope(row, namespace="qc19gt_other")


def test_summaries_exclude_top_level_index_categories():
    rows = [
        {"path": "kb/index.md", "repository": "r", "namespace": "kb", "title": "index"},
        {"path": "kb/00_index.md", "repository": "r", "namespace": "kb", "title": "00"},
        {"path": "kb/README.md", "repository": "r", "namespace": "kb", "title": "readme"},
        {"path": "kb/curriculum/a.md", "repository": "r", "namespace": "kb", "title": "a"},
        {"path": "kb/life/b.md", "repository": "r", "namespace": "kb", "title": "b"},
    ]
    summaries = build_knowledge_base_summaries(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.document_count == 5
    category_names = {c.name for c in summary.top_level_categories}
    assert category_names == {"curriculum", "life"}
    assert "index.md" not in category_names
    assert "00_index.md" not in category_names
    assert "README.md" not in category_names


# ---------------------------------------------------------------------------
# Read range clamping / truncation
# ---------------------------------------------------------------------------


def test_clamp_read_range():
    # 1-based inclusive ranges.
    assert _clamp_read_range(1, 40, 100) == (1, 40, [])
    assert _clamp_read_range(5, 15, 100) == (5, 15, [])

    start, end, warnings = _clamp_read_range(1, 100, 100)
    assert (start, end) == (1, 40)
    assert warnings and "收窄" in warnings[0]

    start, end, warnings = _clamp_read_range(51, 150, 200)
    assert (start, end) == (51, 90)
    assert warnings and "收窄" in warnings[0]


def test_clamp_read_range_handles_none():
    start, end, warnings = _clamp_read_range(None, None, 30)
    assert (start, end) == (1, 30)
    assert warnings == []


def test_recompute_line_end_from_prefixed_content():
    content = "1: a\n2: b\n3: c"
    assert _recompute_line_end(content, 99) == 3
    assert _recompute_line_end("no prefix", 7) == 7
    assert _recompute_line_end("", 5) == 5


def test_truncate_read_result():
    result = {"content": "x" * 3000, "end_line": 100}
    out = _truncate_read_result(result)
    assert out["truncated"] is True
    assert len(out["content"]) <= 2400
    assert out["end_line"] is not None
    assert out["end_line"] <= 100

    unchanged = {"content": "short", "end_line": 2}
    assert _truncate_read_result(unchanged) == unchanged


def test_truncate_read_result_ends_at_full_line():
    # Build prefixed lines so the result must end on a complete line boundary.
    lines = [f"{i + 1}: " + "x" * 60 for i in range(60)]
    content = "\n".join(lines)
    result = _truncate_read_result({"content": content, "end_line": 60})
    assert result["truncated"] is True
    assert len(result["content"]) <= 2400
    # The cut must land exactly on a newline, not inside a line.
    assert content[len(result["content"])] == "\n"
    # The last kept line is a complete prefixed line.
    last_line = result["content"].rsplit("\n", 1)[-1]
    assert re.match(r"^\d+: x+", last_line)
    assert result["end_line"] == int(last_line.split(":", 1)[0])


def test_read_doc_tool_clamps_large_line_range(tmp_path: Path):
    # Each paragraph is preserved as one line by clean_document_body.
    body = "\n\n".join(f"line {i}" for i in range(1, 101))
    doc = _doc(tmp_path, "long.md", "长文档", body, "long1")
    index = _index_with_docs(tmp_path, [doc])
    tracker = SourceTracker()
    tool = ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker)
    result = asyncio.run(tool._run(file_path="long.md", start_line=0, end_line=100))
    assert "error" not in result
    assert result["end_line"] == 40
    assert result["warnings"]
    assert len(tracker.evidence_excerpts) == 1
    assert tracker.read_count == 1


def test_read_doc_tool_truncates_long_content(tmp_path: Path):
    # 60 lines of 80 characters each => 4800 chars, exceeding the 2400 budget.
    body = "\n".join(f"line {i:02d} " + "x" * 70 for i in range(60))
    doc = _doc(tmp_path, "wide.md", "宽文档", body, "wide1")
    index = _index_with_docs(tmp_path, [doc])
    tracker = SourceTracker()
    tool = ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker)
    result = asyncio.run(tool._run(file_path="wide.md", start_line=0, end_line=40))
    assert "error" not in result
    assert result.get("truncated") is True
    assert len(result["content"]) <= 2400
    assert result["end_line"] is not None
    assert result["end_line"] < 40


def test_get_doc_details_truncates_content_for_evidence(tmp_path: Path):
    body = "x" * 5000
    doc = _doc(tmp_path, "detail.md", "详情", body, "detail1")
    index = _index_with_docs(tmp_path, [doc])
    tracker = SourceTracker()
    tool = GetDocDetailsTool(index=index, docs_root=tmp_path, tracker=tracker)
    result = asyncio.run(tool._run(yuque_id="detail1", include_content=True))
    assert "error" not in result
    assert len(result["content"]) == 5000
    assert len(tracker.evidence_excerpts) == 1
    assert len(tracker.evidence_excerpts[0].content) == 2400


# ---------------------------------------------------------------------------
# Summary behavior
# ---------------------------------------------------------------------------


def test_evidence_summary_counts_only_reads_and_details():
    tracker = SourceTracker()
    tracker.add_evidence(
        evidence_excerpt_from_text(
            "导航列表", title="nav", file_path="nav.md", evidence_type="navigation"
        )
    )
    tracker.add_read_document(
        Document(
            "1", "t", "r", "n", "s", "", "a", "b", "body", path=Path("doc.md")
        ),
        "正文内容",
    )
    assert len(tracker.evidence_excerpts) == 2
    assert tracker.read_count == 1
    assert "doc.md" in tracker.read_sources
    assert "nav.md" not in tracker.read_sources
