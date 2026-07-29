"""Tests for nju_qa.knowledge_structure."""

from __future__ import annotations

import pytest

from nju_qa.knowledge_structure import (
    build_knowledge_base_summaries,
    build_knowledge_tree,
    list_documents_under_prefix,
    tree_to_text,
)


def _row(path: str, **kwargs) -> dict:
    return {"path": path, "repository": "repo", "namespace": path.split("/")[0], **kwargs}


@pytest.fixture
def sample_rows():
    return [
        _row("QA/01_入学与行政事务/A.md", title="A", yuque_id="a"),
        _row("QA/01_入学与行政事务/B.md", title="B", yuque_id="b"),
        _row("QA/02_教务与学业/课程与选课/C.md", title="C", yuque_id="c"),
        _row("QA/02_教务与学业/培养方案/D.md", title="D", yuque_id="d"),
        _row("QA/02_教务与学业/考试与成绩/E.md", title="E", yuque_id="e"),
        _row("QA/03_校园生活/F.md", title="F", yuque_id="f"),
        _row("QA/归档/G.md", title="G", yuque_id="g"),
        _row("QA/00_index.md", title="00_index", yuque_id="idx"),
    ]


def test_build_tree_counts(sample_rows):
    tree = build_knowledge_tree(sample_rows, namespace="QA")
    assert tree is not None
    assert tree.name == "QA"
    assert tree.document_count == 8
    assert len(tree.children) == 5  # 00_index, 01, 02, 03, 归档
    # Actually 00_index is a leaf child, not a directory, so 5 children.
    assert len(tree.children) == 5


def test_build_tree_max_depth(sample_rows):
    tree = build_knowledge_tree(sample_rows, namespace="QA", max_depth=1)
    assert tree is not None
    for child in tree.children:
        assert child.depth == 1
        assert child.children == ()


def test_build_tree_namespace_filter(sample_rows):
    tree = build_knowledge_tree(sample_rows, namespace="Other")
    assert tree is None


def test_build_tree_path_prefix(sample_rows):
    tree = build_knowledge_tree(
        sample_rows, namespace="QA", path_prefix="02_教务与学业"
    )
    assert tree is not None
    names = {child.name for child in tree.children}
    assert names == {"课程与选课", "培养方案", "考试与成绩"}


def test_exclude_archived(sample_rows):
    tree = build_knowledge_tree(sample_rows, namespace="QA", include_archived=False)
    assert tree is not None
    assert "归档" not in {child.name for child in tree.children}


def test_summaries(sample_rows):
    summaries = build_knowledge_base_summaries(sample_rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.namespace == "QA"
    assert summary.document_count == 8
    categories = {c.name: c.document_count for c in summary.top_level_categories}
    assert categories["01_入学与行政事务"] == 2
    assert categories["02_教务与学业"] == 3
    assert categories["03_校园生活"] == 1
    assert categories["归档"] == 1


def test_list_documents_pagination(sample_rows):
    docs, has_more = list_documents_under_prefix(sample_rows, namespace="QA", limit=3)
    assert len(docs) == 3
    assert has_more is True

    docs2, has_more2 = list_documents_under_prefix(
        sample_rows, namespace="QA", limit=100
    )
    assert len(docs2) == 8
    assert has_more2 is False


def test_list_documents_title_query(sample_rows):
    docs, _ = list_documents_under_prefix(
        sample_rows, namespace="QA", title_query="C"
    )
    assert len(docs) == 1
    assert docs[0]["title"] == "C"


def test_list_documents_exclude_index_and_archived(sample_rows):
    docs, _ = list_documents_under_prefix(
        sample_rows,
        namespace="QA",
        include_index=False,
        include_archived=False,
    )
    assert all(not d["is_index"] for d in docs)
    assert all("归档" not in d["path"] for d in docs)


def test_tree_to_text_renders(sample_rows):
    tree = build_knowledge_tree(sample_rows, namespace="QA", max_depth=1)
    text = tree_to_text(tree)
    assert "QA" in text
    assert "01_入学与行政事务" in text
