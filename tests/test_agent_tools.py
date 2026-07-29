from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from nova_cac.chunk_store import ChunkStore
from nova_cac.evidence import SourceTracker
from nova_cac.local_corpus import LocalCorpus
from nova_cac.retriever import HybridRetriever
from nova_cac.tools import (
    GetDocOutlineTool,
    GrepLocalDocsTool,
    ListDocsTool,
    ReadDocTool,
    SearchKnowledgeBaseTool,
)


class AgentToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        knowledge = base / "knowledge" / "03_规章与活动"
        knowledge.mkdir(parents=True)
        (knowledge / "章程.md").write_text(
            "---\n"
            'title: "NOVA章程"\n'
            'source_url: "https://example/nova"\n'
            "---\n"
            "# 共同价值观\n\n"
            "无论是否具备技术基础，只要认同共同价值观，NOVA欢迎本科学生加入。\n\n"
            "## 社团活动\n\n活动以分享和真实问题为核心。",
            encoding="utf-8",
        )
        self.store = ChunkStore(base / "chunks.sqlite3")
        self.corpus = LocalCorpus(
            base / "knowledge",
            self.store,
            vector_index=None,
            embed_many=None,
        )
        asyncio.run(self.corpus.refresh())
        self.retriever = HybridRetriever(
            self.corpus,
            self.store,
            vector_index=None,
            embed_one=None,
            score_threshold=0,
        )
        self.tracker = SourceTracker()

    def tearDown(self) -> None:
        self.corpus.close()
        self.temp.cleanup()

    def test_search_is_candidate_and_read_is_concrete_evidence(self) -> None:
        search = SearchKnowledgeBaseTool(
            retriever=self.retriever,
            tracker=self.tracker,
        )
        result = asyncio.run(search._run("没有技术基础能加入吗"))
        self.assertTrue(result["candidates"])
        self.assertEqual([], self.tracker.evidence_excerpts)

        candidate = result["candidates"][0]
        read = ReadDocTool(corpus=self.corpus, tracker=self.tracker)
        read_result = asyncio.run(
            read._run(
                document_id=candidate["document_id"],
                start_line=1,
                end_line=12,
            )
        )
        self.assertEqual("E1", read_result["evidence_id"])
        self.assertEqual(1, len(self.tracker.evidence_excerpts))
        self.assertIn("共同价值观", self.tracker.evidence_excerpts[0].content)

    def test_grep_returns_line_ranges_and_navigation_tools_work(self) -> None:
        grep = GrepLocalDocsTool(corpus=self.corpus)
        result = asyncio.run(grep._run("技术基础 加入"))
        self.assertTrue(result["results"])
        self.assertGreaterEqual(result["results"][0]["line_start"], 1)

        listing = asyncio.run(ListDocsTool(corpus=self.corpus)._run("规章"))
        self.assertEqual(1, len(listing["documents"]))
        outline = asyncio.run(
            GetDocOutlineTool(corpus=self.corpus)._run(
                document_id=listing["documents"][0]["document_id"]
            )
        )
        self.assertGreaterEqual(len(outline["headings"]), 2)


if __name__ == "__main__":
    unittest.main()
