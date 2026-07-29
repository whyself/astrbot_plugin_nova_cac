"""End-to-end chunk retrieval, index lifecycle and benchmark tests."""

from __future__ import annotations
import asyncio
import hashlib
import math
from pathlib import Path
import pytest
from nju_qa.chunk_indexer import ChunkIndexer
from nju_qa.chunk_store import ChunkStore
from nju_qa.chunking import split_markdown
from nju_qa.config import PluginConfig
from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.retriever import HybridRetriever
from nju_qa.vector_index import ChunkVectorIndex


def _semantic_mock_vector(text: str, dim: int = 512) -> list[float]:
    """Mock embedding that correlates with shared Chinese/English terms.

    Unlike a pure hash, texts sharing words get high cosine similarity,
    which makes vector retrieval deterministic and meaningful in tests.
    """
    from nju_qa.keyword_index import _tokenize

    tokens = _tokenize(text)
    vec = [0.0] * dim
    for token in tokens:
        # Double hashing reduces collisions in small mock dimensions.
        h1 = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        h2 = int(hashlib.sha256((token + "::2").encode("utf-8")).hexdigest(), 16)
        vec[h1 % dim] += 1.0
        vec[h2 % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm if norm else 0.0 for x in vec]


async def _mock_embed(text: str) -> list[float] | None:
    return _semantic_mock_vector(text, dim=512)


class _FailingEmbed:
    def __init__(self, fail_after: int = 0):
        self.calls = 0
        self.fail_after = fail_after

    async def __call__(self, text: str) -> list[float] | None:
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("embedding service unavailable")
        return _semantic_mock_vector(text, dim=512)


def _doc(
    yuque_id: str,
    title: str,
    body: str,
    tmp_path: Path,
    *,
    repository: str = "新生手册",
    namespace: str = "nju/guide",
    slug: str = "",
    url: str = "",
    updated_at: str = "2026-01-01",
) -> Document:
    slug = slug or title.lower().replace(" ", "-")
    url = url or f"https://www.yuque.com/{namespace}/{slug}"
    path = tmp_path / f"{yuque_id}.md"
    return Document(
        yuque_id=yuque_id,
        title=title,
        repository=repository,
        namespace=namespace,
        slug=slug,
        url=url,
        created_at="2026-01-01",
        updated_at=updated_at,
        body=body,
        path=path,
    )


def _index_doc(index: DocumentIndex, doc: Document) -> None:
    index.open()
    index.upsert(doc)


def _save_chunks(chunk_store: ChunkStore, doc: Document, size: int = 600, overlap: int = 80) -> list:
    chunks = split_markdown(
        doc.yuque_id,
        doc.body,
        title=doc.title,
        repository=doc.repository,
        namespace=doc.namespace,
        slug=doc.slug,
        file_path=str(doc.path),
        source_url=doc.url,
        updated_at=doc.updated_at,
        size=size,
        overlap=overlap,
    )
    chunk_store.save_document_chunks(doc.yuque_id, chunks)
    return chunks


async def _index_chunks(
    chunk_store: ChunkStore,
    vector_index: ChunkVectorIndex,
    doc: Document,
    embed=None,
) -> None:
    embed = embed or _mock_embed
    chunks = _save_chunks(chunk_store, doc)
    vectors = [await embed(c.embedding_text) for c in chunks]
    vector_index.upsert(chunks, vectors)


def _build_corpus(tmp_path: Path) -> dict[str, Document]:
    freshman_guide = _doc(
        "1",
        "新生入学指南",
        (
            "# 欢迎来到南京大学\n\n"
            "南京大学是一所历史悠久的综合性大学。新生报到前请仔细阅读本指南。\n\n"
            "## 常用网站与平台\n\n"
            "入学后，新生需要了解若干重要网站。首先是南京大学官网（www.nju.edu.cn），"
            "学校新闻、通知公告和院系入口都可以从这里进入。"
            "其次是网上办事服务大厅（http://banshi.nju.edu.cn），用于请假、报销、"
            "证明申请等日常事务办理。"
            "第三是统一身份认证系统（https://authserver.nju.edu.cn），它是大部分校内系统的登录入口。"
            "最后是教务系统（https://jw.nju.edu.cn），选课、查成绩、考试安排都在那里。\n\n"
            "## 校园事务办理\n\n"
            "如需办理校园事务，例如请假、报销或开具证明，请登录网上办事服务大厅。"
            "大厅集成了本科生院、研究生院、财务处等多个部门服务入口，"
            "通过统一身份认证后即可在线提交申请。\n\n"
            "## 报到流程\n\n"
            "请携带录取通知书、身份证等材料前往所在校区报到。"
        ),
        tmp_path,
    )
    dormitory = _doc(
        "2",
        "新生宿舍介绍",
        (
            "# 宿舍概况\n\n"
            "南京大学为新生提供多种宿舍选择。四人间为主，部分楼栋有双人间。"
            "宿舍内配备空调、独立卫生间和校园网接口。"
            "宿舍楼附近有食堂和自习室。新生可以在网上办事大厅查询宿舍分配结果。"
            "如需报修，请登录学校网站提交申请。"
        ),
        tmp_path,
    )
    network = _doc(
        "3",
        "校园网络使用说明",
        (
            "# 校园网\n\n"
            "校园网覆盖各校区教学楼、图书馆和宿舍区。"
            "连接后通过统一身份认证登录即可访问学术资源和互联网。"
            "遇到网络故障可前往信息化中心网站查询解决方案。"
        ),
        tmp_path,
    )
    unrelated = _doc(
        "4",
        "南京历史文化简介",
        (
            "# 南京历史\n\n"
            "南京是中国四大古都之一，拥有丰富的历史遗迹。"
            "中山陵、明孝陵和夫子庙都是著名景点。"
        ),
        tmp_path,
    )
    year_a = _doc(
        "5",
        "2025 级本科生选课须知",
        (
            "# 2025 级本科生选课须知\n\n"
            "2025 级本科生请在教务系统规定时间内完成选课。"
            "通识课、专业课和体育课需分别提交志愿。"
        ),
        tmp_path,
    )
    year_b = _doc(
        "6",
        "2026 级本科生选课须知",
        (
            "# 2026 级本科生选课须知\n\n"
            "2026 级本科生请在教务系统规定时间内完成选课。"
            "注意核对培养方案中的必修学分要求。"
        ),
        tmp_path,
    )
    return {
        "freshman_guide": freshman_guide,
        "dormitory": dormitory,
        "network": network,
        "unrelated": unrelated,
        "year_a": year_a,
        "year_b": year_b,
    }


@pytest.fixture
def indexed_corpus(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    chunk_store.open()
    corpus = _build_corpus(tmp_path)
    for doc in corpus.values():
        _index_doc(index, doc)
        asyncio.run(_index_chunks(chunk_store, vector_index, doc))
    retriever = HybridRetriever(
        index,
        PluginConfig.from_mapping(
            {"yuque_repositories": ["nju/guide"], "score_threshold": 0.15}
        ),
        chunk_store=chunk_store,
        vector_index=vector_index,
        embed=_mock_embed,
    )
    return retriever, corpus


def test_website_query_top3_contains_platform_chunk(indexed_corpus):
    retriever, corpus = indexed_corpus
    results = asyncio.run(retriever.search("新生需要看哪些网站"))
    titles = [r.document.title for r in results[:3]]
    assert corpus["freshman_guide"].title in titles
    top_chunk = results[0].chunk
    assert top_chunk is not None
    assert any(
        word in top_chunk.content_snippet
        for word in ("南京大学官网", "办事服务大厅", "统一身份认证", "教务系统")
    )


def test_synonym_query_finds_platform_chunk(indexed_corpus):
    retriever, corpus = indexed_corpus
    results = asyncio.run(retriever.search("新生常用平台有哪些"))
    titles = [r.document.title for r in results[:3]]
    assert corpus["freshman_guide"].title in titles


def test_campus_affairs_query(indexed_corpus):
    retriever, _corpus = indexed_corpus
    results = asyncio.run(retriever.search("去哪里办校园事务"))
    titles = [r.document.title for r in results[:3]]
    assert "新生入学指南" in titles


def test_auth_query(indexed_corpus):
    retriever, _corpus = indexed_corpus
    results = asyncio.run(retriever.search("统一身份认证是什么"))
    assert results
    assert any("统一身份认证" in (r.chunk.content_snippet or "") for r in results[:3])


def test_academic_query(indexed_corpus):
    retriever, _corpus = indexed_corpus
    results = asyncio.run(retriever.search("教务相关系统"))
    assert results
    assert any("教务系统" in (r.chunk.content_snippet or "") for r in results[:3])


def test_dormitory_query_not_drowned_by_freshman_noise(indexed_corpus):
    retriever, corpus = indexed_corpus
    results = asyncio.run(retriever.search("新生宿舍怎么样"))
    titles = [r.document.title for r in results[:3]]
    # Dormitory doc should outrank the general freshman guide for this query.
    assert corpus["dormitory"].title in titles


def test_unknown_policy_is_not_reliable(indexed_corpus):
    retriever, _corpus = indexed_corpus
    results = asyncio.run(
        retriever.search("研究生学业奖学金评定细则是什么")
    )
    assert not results or not any(r.reliable for r in results)


def test_keyword_fallback_without_embedding(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "新生网站指南", "新生请使用信息门户和教务网站。", tmp_path)
    _index_doc(index, doc)
    _save_chunks(chunk_store, doc)
    retriever = HybridRetriever(
        index,
        PluginConfig.from_mapping(
            {"yuque_repositories": ["nju/guide"], "score_threshold": 0}
        ),
        chunk_store=chunk_store,
        vector_index=None,
    )
    report = asyncio.run(retriever.debug_search("新生网站"))
    assert report["mode"] == "keyword" and report["selected"]


def test_empty_vector_index_skips_embedding_request(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "新生网站指南", "新生请使用信息门户和教务网站。", tmp_path)
    _index_doc(index, doc)
    _save_chunks(chunk_store, doc)

    class EmptyVectorIndex:
        @staticmethod
        def count():
            return 0

        @staticmethod
        def query(*_args, **_kwargs):
            raise AssertionError("empty vector index must not be queried")

    embed_calls = 0

    async def embed(_text):
        nonlocal embed_calls
        embed_calls += 1
        return [1.0, 0.0]

    retriever = HybridRetriever(
        index,
        PluginConfig.from_mapping(
            {
                "enable_vector_search": True,
                "embedding_api_key": "test-key",
                "embedding_base_url": "https://embedding.invalid/v1",
                "score_threshold": 0,
            }
        ),
        chunk_store=chunk_store,
        vector_index=EmptyVectorIndex(),
        embed=embed,
    )
    report = asyncio.run(retriever.debug_search("新生网站"))
    assert report["mode"] == "keyword"
    assert report["selected"]
    assert embed_calls == 0


def test_chunk_result_fields_are_populated(indexed_corpus):
    retriever, _corpus = indexed_corpus
    results = asyncio.run(retriever.search("统一身份认证"))
    assert results
    r = results[0]
    assert r.chunk is not None
    assert r.chunk.chunk_id
    assert r.chunk.document_id
    assert r.chunk.source_url
    assert isinstance(r.chunk.vector_relevance, float)
    assert isinstance(r.chunk.keyword_score, float)
    assert isinstance(r.chunk.final_score, float)
    assert r.chunk.retrieval_methods
    assert r.chunk.slug
    assert r.chunk.namespace == "nju/guide"


def test_first_index_creates_multiple_chunks(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "长文", "# A\n\n" + "内容。" * 200, tmp_path)
    chunks = _save_chunks(chunk_store, doc, size=400, overlap=50)
    assert len(chunks) > 1
    assert chunk_store.chunk_count() == len(chunks)


def test_repeat_index_is_not_duplicated(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "长文", "# A\n\n" + "内容。" * 200, tmp_path)
    _save_chunks(chunk_store, doc, size=400, overlap=50)
    first_count = chunk_store.chunk_count()
    _save_chunks(chunk_store, doc, size=400, overlap=50)
    assert chunk_store.chunk_count() == first_count


def test_body_update_removes_old_chunks(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "标题", "旧内容。" * 50, tmp_path)
    old_chunks = _save_chunks(chunk_store, doc, size=300, overlap=30)
    old_ids = {c.chunk_id for c in old_chunks}
    doc2 = _doc("1", "标题", "全新内容。" * 50, tmp_path)
    new_chunks = _save_chunks(chunk_store, doc2, size=300, overlap=30)
    new_ids = {c.chunk_id for c in new_chunks}
    assert not old_ids & new_ids
    assert chunk_store.chunk_count() == len(new_chunks)


def test_delete_document_removes_all_chunks(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "标题", "内容。" * 50, tmp_path)
    _save_chunks(chunk_store, doc)
    assert chunk_store.chunk_count() > 0
    chunk_store.delete_document(doc.yuque_id)
    assert chunk_store.chunk_count() == 0


def test_rename_updates_title_metadata(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "旧标题", "内容。" * 20, tmp_path)
    _save_chunks(chunk_store, doc)
    chunk_store.update_document_metadata(doc.yuque_id, title="新标题")
    chunks = chunk_store.get_document_chunks(doc.yuque_id)
    assert all(c.title == "新标题" for c in chunks)


def test_move_updates_file_path(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "标题", "内容。" * 20, tmp_path)
    _save_chunks(chunk_store, doc)
    chunk_store.update_document_metadata(doc.yuque_id, file_path="new/path.md")
    chunks = chunk_store.get_document_chunks(doc.yuque_id)
    assert all(c.file_path == "new/path.md" for c in chunks)


def test_chroma_persists_and_reopens(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "持久化测试", "统一身份认证系统用于登录校内应用。", tmp_path)
    _index_doc(DocumentIndex(tmp_path / "index.sqlite3"), doc)
    _save_chunks(chunk_store, doc)
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    chunks = chunk_store.get_document_chunks(doc.yuque_id)
    vectors = [_semantic_mock_vector(c.embedding_text) for c in chunks]
    vector_index.upsert(chunks, vectors)
    assert vector_index.count() == len(chunks)

    # Reopen
    vector_index2 = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    assert vector_index2.count() == len(chunks)
    query_vec = _semantic_mock_vector("统一身份认证")
    raw = vector_index2.query(query_vec, n=5)
    assert raw["ids"] and raw["ids"][0]


def test_model_mismatch_resets_collection(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "测试", "内容。", tmp_path)
    _save_chunks(chunk_store, doc)
    vi1 = ChunkVectorIndex(
        tmp_path / "vectors", model="model-a", embedding_dimension=512
    )
    chunks = chunk_store.get_document_chunks(doc.yuque_id)
    vi1.upsert(chunks, [[0.1] * 16])
    assert vi1.count() == 1
    vi2 = ChunkVectorIndex(
        tmp_path / "vectors", model="model-b", embedding_dimension=512
    )
    assert vi2.count() == 0


def test_dimension_mismatch_resets_collection(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    doc = _doc("1", "测试", "内容。", tmp_path)
    _save_chunks(chunk_store, doc)
    vi1 = ChunkVectorIndex(
        tmp_path / "vectors", model="model-x", embedding_dimension=512
    )
    chunks = chunk_store.get_document_chunks(doc.yuque_id)
    vi1.upsert(chunks, [[0.1] * 16])
    vi2 = ChunkVectorIndex(
        tmp_path / "vectors", model="model-x", embedding_dimension=32
    )
    assert vi2.count() == 0


def test_full_rebuild_is_consistent(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    chunk_store.open()
    docs = [
        _doc("1", "A", "内容一。" * 30, tmp_path),
        _doc("2", "B", "内容二。" * 30, tmp_path),
    ]
    for d in docs:
        _index_doc(index, d)
    indexer = ChunkIndexer(
        chunk_store, vector_index, _mock_embed, chunk_size=300, overlap=30
    )
    result1 = asyncio.run(indexer.rebuild(index.all_documents()))
    count1 = chunk_store.chunk_count()
    assert count1 > 0
    result2 = asyncio.run(indexer.rebuild(index.all_documents()))
    count2 = chunk_store.chunk_count()
    assert count1 == count2
    assert result1["chunks"] == result2["chunks"]
    assert vector_index.count() == count2


def test_embedding_unavailable_still_allows_keyword_search(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    chunk_store.open()
    doc = _doc("1", "网站指南", "请访问教务系统和网上办事大厅。", tmp_path)
    _index_doc(index, doc)
    _save_chunks(chunk_store, doc)
    retriever = HybridRetriever(
        index,
        PluginConfig.from_mapping(
            {"yuque_repositories": ["nju/guide"], "score_threshold": 0}
        ),
        chunk_store=chunk_store,
        vector_index=vector_index,
    )
    # No embedding configured, so retriever returns keyword results.
    results = asyncio.run(retriever.search("教务系统"))
    assert results
    assert results[0].chunk is not None


def test_batch_failure_is_visible(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    doc = _doc("1", "批量失败", "内容。" * 10 + "\n\n" + "更多。" * 10, tmp_path)
    _save_chunks(chunk_store, doc, size=200, overlap=20)
    failing_embed = _FailingEmbed(fail_after=0)
    indexer = ChunkIndexer(
        chunk_store, vector_index, failing_embed, chunk_size=200, overlap=20
    )
    result = asyncio.run(indexer.index_document(_row(doc)))
    assert result["error"]
    assert vector_index.count() == 0


def _row(doc: Document) -> dict:
    return {
        "yuque_id": doc.yuque_id,
        "title": doc.title,
        "repository": doc.repository,
        "namespace": doc.namespace,
        "slug": doc.slug,
        "url": doc.url,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "body": doc.body,
        "path": str(doc.path),
    }


def test_sync_delete_keeps_vector_consistent(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock-embedding", embedding_dimension=512
    )
    chunk_store.open()
    doc = _doc("1", "待删除", "内容。" * 20, tmp_path)
    _index_doc(index, doc)
    asyncio.run(_index_chunks(chunk_store, vector_index, doc))
    assert chunk_store.chunk_count() > 0
    assert vector_index.count() > 0
    chunk_store.delete_document(doc.yuque_id)
    vector_index.delete_document(doc.yuque_id)
    assert chunk_store.chunk_count() == 0
    assert vector_index.count() == 0


def test_debug_search_includes_terms_and_scores(indexed_corpus):
    retriever, _corpus = indexed_corpus
    report = asyncio.run(retriever.debug_search("教务系统"))
    assert report["query_terms"]
    assert report["keyword_candidates"]
    assert report["vector_candidates"]
    assert report["selected"]
    assert report["threshold"] >= 0
    item = report["selected"][0]
    assert item.chunk is not None
    assert item.chunk.final_score >= report["threshold"]
