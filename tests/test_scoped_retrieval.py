"""Tests for scoped hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from nju_qa.chunk_store import ChunkStore
from nju_qa.chunking import Chunk, _content_hash, _stable_chunk_id
from nju_qa.config import PluginConfig
from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.retriever import HybridRetriever


def _chunk(
    document_id: str,
    chunk_index: int,
    content: str,
    *,
    title: str = "",
    repository: str = "repo",
    namespace: str = "QA",
    slug: str = "",
    file_path: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=_stable_chunk_id(document_id, chunk_index, content),
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        content_hash=_content_hash(content),
        title=title,
        repository=repository,
        namespace=namespace,
        slug=slug or document_id,
        file_path=file_path or f"{namespace}/{document_id}.md",
        source_url="",
        updated_at="b",
    )


def _doc(
    tmp_path: Path,
    rel: str,
    title: str,
    body: str,
    yuque_id: str,
    namespace: str = "QA",
    repository: str = "repo",
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
def scoped_index(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    docs = [
        _doc(
            tmp_path,
            "QA/01_入学与行政事务/校园卡.md",
            "校园卡",
            "校园卡可在信息化建设管理服务中心补办。",
            "card",
        ),
        _doc(
            tmp_path,
            "QA/02_教务与学业/选课.md",
            "选课",
            "选课系统在教务系统开放。",
            "course",
        ),
        _doc(
            tmp_path,
            "Other/校园卡.md",
            "其他校园卡",
            "其他校园卡说明。",
            "other_card",
            namespace="Other",
        ),
        _doc(
            tmp_path,
            "QA/归档/旧文档.md",
            "旧文档",
            "旧文档内容包含校园卡。",
            "old",
        ),
    ]
    for doc in docs:
        index.upsert(doc)

    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.save_document_chunks(
        "card",
        [
            _chunk(
                "card",
                0,
                "校园卡可在信息化建设管理服务中心补办。",
                title="校园卡",
                file_path="QA/01_入学与行政事务/校园卡.md",
            ),
        ],
    )
    chunk_store.save_document_chunks(
        "course",
        [
            _chunk(
                "course",
                0,
                "选课系统在教务系统开放。",
                title="选课",
                file_path="QA/02_教务与学业/选课.md",
            ),
        ],
    )
    chunk_store.save_document_chunks(
        "other_card",
        [
            _chunk(
                "other_card",
                0,
                "其他校园卡说明。",
                title="其他校园卡",
                namespace="Other",
                file_path="Other/校园卡.md",
            ),
        ],
    )
    chunk_store.save_document_chunks(
        "old",
        [
            _chunk(
                "old",
                0,
                "旧文档内容包含校园卡。",
                title="旧文档",
                file_path="QA/归档/旧文档.md",
            ),
        ],
    )

    config = PluginConfig.from_mapping(
        {
            "enable_vector_search": False,
            "retrieval_top_k": 5,
            "score_threshold": 0.1,
        }
    )
    retriever = HybridRetriever(index, config, chunk_store=chunk_store)
    return retriever


@pytest.mark.asyncio
async def test_scope_namespace_filters_results(scoped_index):
    results = await scoped_index.search("校园卡", namespace="QA", include_archived=False)
    assert len(results) == 1
    assert results[0].document.yuque_id == "card"


@pytest.mark.asyncio
async def test_scope_path_prefix_filters_results(scoped_index):
    results = await scoped_index.search("校园卡", namespace="QA", path_prefix="01_入学与行政事务")
    assert len(results) == 1
    assert results[0].document.yuque_id == "card"


@pytest.mark.asyncio
async def test_scope_document_ids_filters_results(scoped_index):
    results = await scoped_index.search("校园卡", document_ids={"old"})
    assert len(results) == 1
    assert results[0].document.yuque_id == "old"


@pytest.mark.asyncio
async def test_scope_exclude_archived(scoped_index):
    results = await scoped_index.search(
        "校园卡", namespace="QA", include_archived=False
    )
    ids = {r.document.yuque_id for r in results}
    assert "old" not in ids
    assert "card" in ids


@pytest.mark.asyncio
async def test_scope_repository_filters_results(scoped_index):
    # Update repository for one doc via chunk metadata and force keyword index rebuild.
    scoped_index.chunk_store.update_document_metadata(
        "card", repository="special"
    )
    scoped_index._keyword_index_signature = ""
    results = await scoped_index.search("校园卡", repository="special")
    assert len(results) == 1
    assert results[0].document.yuque_id == "card"


@pytest.mark.asyncio
async def test_unscoped_search_returns_all(scoped_index):
    results = await scoped_index.search("校园卡")
    ids = {r.document.yuque_id for r in results}
    assert ids == {"card", "other_card", "old"}
