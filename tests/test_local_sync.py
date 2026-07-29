from __future__ import annotations

import asyncio
import json

import nju_qa.local_sync as local_sync
from nju_qa.chunk_store import ChunkStore
from nju_qa.config import PluginConfig
from nju_qa.document_index import DocumentIndex
from nju_qa.document_store import DocumentStore
from nju_qa.local_sync import LocalPackSync
from nju_qa.tools import ParseYuqueUrlTool
from nju_qa.vector_index import ChunkVectorIndex


def test_local_pack_replaces_yuque_sync_and_tracks_changes(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    first = knowledge / "first.md"
    first.write_text(
        '---\ntitle: "第一篇"\nsource_url: "https://www.yuque.com/nova/guide/first"\n---\n# 第一篇\n\n旧内容。',
        encoding="utf-8",
    )
    data = tmp_path / "data"
    store = DocumentStore(data / "documents")
    index = DocumentIndex(data / "index.sqlite3")
    chunks = ChunkStore(data / "chunks.sqlite3")
    vectors = ChunkVectorIndex(data / "vectors", "text-embedding-3-small")
    config = PluginConfig.from_mapping({"enable_vector_search": False})
    syncer = LocalPackSync(
        knowledge,
        store,
        index,
        chunks,
        vectors,
        config,
        state_path=data / "state.json",
    )

    initial = asyncio.run(syncer.refresh())
    assert initial["documents"] == 1
    assert initial["chunks"] > 0
    assert index.all_documents()[0]["url"] == "https://www.yuque.com/nova/guide/first"
    parsed = asyncio.run(
        ParseYuqueUrlTool(index=index, docs_root=store.root)._run(
            url="https://www.yuque.com/nova/guide/first?from=test#part"
        )
    )
    assert parsed["count"] == 1
    assert parsed["results"][0]["title"] == "第一篇"

    first.write_text("# 第一篇\n\n新内容。", encoding="utf-8")
    second = knowledge / "second.md"
    second.write_text("# 第二篇\n\n第二篇内容。", encoding="utf-8")
    changed = asyncio.run(syncer.refresh())
    assert changed["rebuilt"] is True
    assert index.document_count() == 2
    assert any("新内容" in row["body"] for row in index.all_documents())

    first.unlink()
    asyncio.run(syncer.refresh())
    assert index.document_count() == 1
    assert index.all_documents()[0]["title"] == "第二篇"

    index.close()
    chunks.close()
    vectors.close()


def test_embedding_failure_keeps_keyword_index_and_retries_once_per_process(
    tmp_path,
    monkeypatch,
):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "first.md").write_text("# 第一篇\n\n关键词甲。", encoding="utf-8")
    (knowledge / "second.md").write_text("# 第二篇\n\n关键词乙。", encoding="utf-8")
    data = tmp_path / "data"
    store = DocumentStore(data / "documents")
    index = DocumentIndex(data / "index.sqlite3")
    chunks = ChunkStore(data / "chunks.sqlite3")

    class FakeVectorIndex:
        def __init__(self):
            self.clear_calls = 0

        def count(self):
            return 0

        def clear(self):
            self.clear_calls += 1

        def delete_document(self, _document_id):
            return None

    rebuild_calls = 0

    class FailingChunkIndexer:
        def __init__(self, *_args, **_kwargs):
            pass

        async def rebuild(self, _rows):
            nonlocal rebuild_calls
            rebuild_calls += 1
            chunks.clear()
            return {
                "chunks": 0,
                "failed_documents": 2,
                "errors": ["embedding unavailable"],
            }

    monkeypatch.setattr(local_sync, "ChunkIndexer", FailingChunkIndexer)
    vectors = FakeVectorIndex()
    config = PluginConfig.from_mapping(
        {
            "enable_vector_search": True,
            "embedding_api_key": "test-key",
            "embedding_base_url": "https://embedding.invalid/v1",
        }
    )
    state_path = data / "state.json"
    syncer = LocalPackSync(
        knowledge,
        store,
        index,
        chunks,
        vectors,
        config,
        state_path=state_path,
    )

    first = asyncio.run(syncer.refresh())
    assert first["vector_ready"] is False
    assert first["vector_error"] == "embedding unavailable"
    assert rebuild_calls == 1
    assert chunks.chunk_count() >= 2
    indexed_text = " ".join(chunk.content for chunk in chunks.all_chunks())
    assert "关键词甲" in indexed_text
    assert "关键词乙" in indexed_text
    assert json.loads(state_path.read_text(encoding="utf-8"))["vector_ready"] is False

    second = asyncio.run(syncer.refresh())
    assert second["rebuilt"] is False
    assert second["vector_ready"] is False
    assert rebuild_calls == 1

    index.close()
    chunks.close()
