"""Tests for structure-aware document tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.tools.documents import (
    GetDocOutlineTool,
    GrepLocalDocsTool,
    ListKnowledgeBasesTool,
    ListRepoDocsTool,
    ListRepoTreeTool,
)


def _doc(tmp_path: Path, rel: str, title: str, body: str) -> Document:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    segments = Path(rel).parts
    namespace = segments[0]
    return Document(
        yuque_id=Path(rel).stem,
        title=title,
        repository="repo",
        namespace=namespace,
        slug=Path(rel).stem,
        url=f"https://yuque.test/{namespace}/{Path(rel).stem}",
        created_at="a",
        updated_at="b",
        body=body,
        path=Path(rel),
    )


@pytest.fixture
def sample_index(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    docs = [
        _doc(
            tmp_path,
            "QA/01_入学与行政事务/A.md",
            "A",
            "## 入学\n\n报到流程。\n\n## 户口\n\n户口迁移说明。",
        ),
        _doc(
            tmp_path,
            "QA/01_入学与行政事务/B.md",
            "B",
            "## 缴费\n\n学费缴纳。",
        ),
        _doc(
            tmp_path,
            "QA/02_教务与学业/课程与选课/C.md",
            "C",
            "## 选课\n\n选课系统。",
        ),
        _doc(
            tmp_path,
            "QA/02_教务与学业/培养方案/D.md",
            "D",
            "## 培养方案\n\n培养方案说明。",
        ),
        _doc(
            tmp_path,
            "QA/归档/G.md",
            "G",
            "## 旧文档\n\n已归档。",
        ),
        _doc(
            tmp_path,
            "QA/00_index.md",
            "00_index",
            "## 目录\n\n- 入学\n- 教务",
        ),
    ]
    for doc in docs:
        index.upsert(doc)
    return index, tmp_path


@pytest.mark.asyncio
async def test_list_knowledge_bases_includes_categories(sample_index):
    index, tmp_path = sample_index
    tool = ListKnowledgeBasesTool(index=index, docs_root=tmp_path)
    result = await tool._run()
    assert "knowledge_bases" in result
    kb = next(k for k in result["knowledge_bases"] if k["namespace"] == "QA")
    assert kb["document_count"] == 6
    assert kb["documents"] == 6
    assert any(c["name"] == "01_入学与行政事务" for c in kb["top_level_categories"])


@pytest.mark.asyncio
async def test_list_repo_docs_scoped_by_prefix(sample_index):
    index, tmp_path = sample_index
    tool = ListRepoDocsTool(index=index, docs_root=tmp_path)
    result = await tool._run(namespace="QA", path_prefix="02_教务与学业")
    assert len(result["documents"]) == 2
    assert all("02_教务与学业" in d["path"] for d in result["documents"])
    names = {c["name"] for c in result["categories"]}
    assert names == {"课程与选课", "培养方案"}


@pytest.mark.asyncio
async def test_list_repo_docs_excludes_archived_by_default(sample_index):
    index, tmp_path = sample_index
    tool = ListRepoDocsTool(index=index, docs_root=tmp_path)
    result = await tool._run(namespace="QA")
    assert "归档" not in {d["path"] for d in result["documents"]}


@pytest.mark.asyncio
async def test_list_repo_tree_returns_text_and_structure(sample_index):
    index, tmp_path = sample_index
    tool = ListRepoTreeTool(index=index, docs_root=tmp_path)
    result = await tool._run(namespace="QA", max_depth=2)
    assert "tree_text" in result
    assert result["count"] == 5  # archived excluded by default
    assert "01_入学与行政事务" in result["tree_text"]


@pytest.mark.asyncio
async def test_list_repo_tree_path_prefix(sample_index):
    index, tmp_path = sample_index
    tool = ListRepoTreeTool(index=index, docs_root=tmp_path)
    result = await tool._run(
        namespace="QA", path_prefix="02_教务与学业", max_depth=2
    )
    assert result["count"] == 2
    assert "课程与选课" in result["tree_text"]


@pytest.mark.asyncio
async def test_get_doc_outline(sample_index):
    index, tmp_path = sample_index
    tool = GetDocOutlineTool(index=index, docs_root=tmp_path)
    result = await tool._run(file_path="QA/01_入学与行政事务/A.md")
    assert result["section_count"] == 2
    titles = {s["title"] for s in result["sections"]}
    assert "入学" in titles
    assert "户口" in titles


@pytest.mark.asyncio
async def test_get_doc_outline_query_ranking(sample_index):
    index, tmp_path = sample_index
    tool = GetDocOutlineTool(index=index, docs_root=tmp_path)
    result = await tool._run(
        file_path="QA/01_入学与行政事务/A.md", query="户口"
    )
    assert result["sections"][0]["title"] == "户口"


@pytest.mark.asyncio
async def test_grep_scoped_by_path_prefix(sample_index):
    index, tmp_path = sample_index
    tool = GrepLocalDocsTool(index=index, docs_root=tmp_path)
    result = await tool._run(
        keywords="选课", namespace="QA", path_prefix="02_教务与学业/课程与选课"
    )
    assert result["count"] == 1
    assert result["results"][0]["title"] == "C"


@pytest.mark.asyncio
async def test_grep_excludes_archived_when_configured(sample_index):
    index, tmp_path = sample_index
    tool = GrepLocalDocsTool(index=index, docs_root=tmp_path)
    result_all = await tool._run(keywords="旧文档", namespace="QA", include_archived=True)
    result_live = await tool._run(
        keywords="旧文档", namespace="QA", include_archived=False
    )
    assert result_all["count"] == 1
    assert result_live["count"] == 0


@pytest.mark.asyncio
async def test_list_repo_docs_default_limit_is_20(sample_index):
    index, tmp_path = sample_index
    tool = ListRepoDocsTool(index=index, docs_root=tmp_path)
    result = await tool._run(namespace="QA")
    assert len(result["documents"]) <= 20
    assert result["count"] <= 20


@pytest.mark.asyncio
async def test_list_repo_docs_pagination(sample_index):
    index, tmp_path = sample_index
    tool = ListRepoDocsTool(index=index, docs_root=tmp_path)
    first = await tool._run(namespace="QA", limit=2, offset=0)
    assert len(first["documents"]) == 2
    assert first["has_more"] is True
    second = await tool._run(namespace="QA", limit=2, offset=2)
    assert len(second["documents"]) == 2
    assert first["documents"][0]["path"] != second["documents"][0]["path"]
