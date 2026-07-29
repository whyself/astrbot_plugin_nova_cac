from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from nova_cac.chunk_store import ChunkStore
from nova_cac.chunking import split_markdown
from nova_cac.local_corpus import LocalCorpus
from nova_cac.retriever import HybridRetriever


class ChunkingTests(unittest.TestCase):
    def test_markdown_chunks_are_stable_and_heading_aware(self) -> None:
        body = "---\ntitle: Example\n---\n# 加入 NOVA\n\n没有技术基础也可以申请加入。\n"
        first = split_markdown(
            "doc-1",
            body,
            title="加入说明",
            file_path="03_规章与活动/加入说明.md",
            size=300,
            overlap=30,
        )
        second = split_markdown(
            "doc-1",
            body,
            title="加入说明",
            file_path="03_规章与活动/加入说明.md",
            size=300,
            overlap=30,
        )

        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])
        self.assertIn("加入 NOVA", first[0].content)
        self.assertNotIn("title: Example", first[0].content)

    def test_soft_boundary_chunking_never_skips_text(self) -> None:
        marker = "这段文字绝对不能丢失"
        body = "甲" * 400 + "。" + marker + "乙" * 1500
        chunks = split_markdown("doc", body, size=1000, overlap=180)
        self.assertIn(marker, "\n".join(chunk.content for chunk in chunks))


class LocalCorpusTests(unittest.TestCase):
    def test_chunk_settings_are_part_of_persisted_index_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            knowledge = base / "knowledge"
            knowledge.mkdir()
            (knowledge / "doc.md").write_text("# 标题\n\n" + "内容。" * 400, encoding="utf-8")
            store = ChunkStore(base / "chunks.sqlite3")
            first = LocalCorpus(knowledge, store, chunk_size=300, chunk_overlap=30)
            self.assertTrue(asyncio.run(first.refresh())["rebuilt"])
            self.assertFalse(asyncio.run(first.refresh())["rebuilt"])

            changed = LocalCorpus(knowledge, store, chunk_size=500, chunk_overlap=50)
            self.assertTrue(asyncio.run(changed.refresh())["rebuilt"])
            changed.close()

    def test_refresh_replaces_changed_and_deleted_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            knowledge = base / "knowledge"
            knowledge.mkdir()
            first_path = knowledge / "first.md"
            second_path = knowledge / "second.md"
            first_path.write_text(
                '---\ntitle: "第一篇"\nsource_url: "https://example/1"\n---\n# 第一篇\n\n旧内容。',
                encoding="utf-8",
            )
            second_path.write_text("# 第二篇\n\n保留内容。", encoding="utf-8")
            store = ChunkStore(base / "chunks.sqlite3")
            corpus = LocalCorpus(
                knowledge,
                store,
                vector_index=None,
                embed_many=None,
                chunk_size=300,
                chunk_overlap=30,
            )

            asyncio.run(corpus.refresh())
            first_ids = {chunk.chunk_id for chunk in store.all_chunks()}
            self.assertEqual(2, len(corpus.documents))

            first_path.write_text("# 第一篇\n\n全新内容。", encoding="utf-8")
            second_path.unlink()
            asyncio.run(corpus.refresh())

            chunks = store.all_chunks()
            self.assertEqual(1, len(corpus.documents))
            self.assertIn("全新内容", chunks[0].content)
            self.assertFalse(first_ids & {chunk.chunk_id for chunk in chunks})
            corpus.close()


class HybridRetrieverTests(unittest.TestCase):
    def test_keyword_search_boosts_title_and_falls_back_without_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            knowledge = base / "knowledge"
            knowledge.mkdir()
            (knowledge / "membership.md").write_text(
                "# NOVA 加入条件\n\n没有技术基础也可以加入，前提是认同共同价值观。",
                encoding="utf-8",
            )
            (knowledge / "activity.md").write_text(
                "# 活动安排\n\n本学期组织分享活动。",
                encoding="utf-8",
            )
            store = ChunkStore(base / "chunks.sqlite3")
            corpus = LocalCorpus(knowledge, store, vector_index=None, embed_many=None)
            asyncio.run(corpus.refresh())
            retriever = HybridRetriever(
                corpus,
                store,
                vector_index=None,
                embed_one=None,
                top_k=3,
                score_threshold=0.0,
            )

            results = asyncio.run(retriever.search("NOVA 加入条件"))

            self.assertTrue(results)
            self.assertEqual("NOVA 加入条件", results[0].document.title)
            self.assertEqual(("keyword",), results[0].retrieval_methods)
            corpus.close()

    def test_vector_and_keyword_scores_are_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            knowledge = base / "knowledge"
            knowledge.mkdir()
            (knowledge / "pbl.md").write_text(
                "# PBL\n\nPBL 是以真实问题推动学习的方式。",
                encoding="utf-8",
            )
            store = ChunkStore(base / "chunks.sqlite3")
            corpus = LocalCorpus(knowledge, store, vector_index=None, embed_many=None)
            asyncio.run(corpus.refresh())
            chunk = store.all_chunks()[0]

            class FakeVectorIndex:
                def query(self, embedding, n=20):
                    return {
                        "documents": [[chunk.content]],
                        "metadatas": [[{
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "title": chunk.title,
                            "file_path": chunk.file_path,
                            "source_url": chunk.source_url,
                        }]],
                        "distances": [[0.1]],
                    }

            async def embed_one(_text: str) -> list[float]:
                return [1.0, 0.0]

            retriever = HybridRetriever(
                corpus,
                store,
                vector_index=FakeVectorIndex(),
                embed_one=embed_one,
                top_k=3,
                score_threshold=0.0,
            )
            results = asyncio.run(retriever.search("PBL 学习"))

            self.assertTrue(results)
            self.assertEqual(("vector", "keyword"), results[0].retrieval_methods)
            self.assertGreater(results[0].score, 0)
            corpus.close()


if __name__ == "__main__":
    unittest.main()
