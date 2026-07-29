from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from nova_cac.agent import NO_EVIDENCE, SAFE_FAILURE, NovaCacAgent
from nova_cac.chunk_store import ChunkStore
from nova_cac.evidence import SourceTracker
from nova_cac.local_corpus import LocalCorpus
from nova_cac.tools import ReadDocTool


@dataclass
class Response:
    completion_text: str


class Event:
    unified_msg_origin = "test:session"


class Context:
    async def get_current_chat_provider_id(self, _origin):
        return "provider"


class AgentRuntimeTests(unittest.TestCase):
    def test_research_has_tools_answer_has_none_and_contexts_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            knowledge = base / "knowledge"
            knowledge.mkdir()
            (knowledge / "doc.md").write_text(
                "---\n"
                'title: "NOVA加入"\n'
                'source_url: "https://example/nova"\n'
                "---\n"
                "# 加入\n\n没有技术基础也可以加入。",
                encoding="utf-8",
            )
            corpus = LocalCorpus(
                knowledge,
                ChunkStore(base / "chunks.sqlite3"),
                vector_index=None,
                embed_many=None,
            )
            asyncio.run(corpus.refresh())
            calls = []

            async def loop(**kwargs):
                calls.append(kwargs)
                if "研究阶段" in kwargs["system_prompt"]:
                    tool = next(item for item in kwargs["tools"] if item.name == "read_doc")
                    document_id = next(iter(corpus.documents))
                    await tool._run(document_id=document_id, start_line=1, end_line=10)
                    return Response("research ignored")
                return Response("没有技术基础也可以加入。[E1]")

            def tools(tracker: SourceTracker):
                return [ReadDocTool(corpus=corpus, tracker=tracker)]

            agent = NovaCacAgent(Context(), tools, tool_loop=loop)
            contexts = [{"role": "user", "content": "上一轮问题"}]
            answer = asyncio.run(
                agent.answer(
                    Event(),
                    "那没有基础呢？",
                    base_system_prompt="CORE PACK",
                    contexts=contexts,
                )
            )

            self.assertEqual("没有技术基础也可以加入。", answer)
            self.assertTrue(calls[0]["tools"])
            self.assertEqual([], calls[1]["tools"])
            self.assertEqual(contexts, calls[0]["contexts"])
            self.assertIn("CORE PACK", calls[0]["system_prompt"])
            self.assertIn("CORE PACK", calls[1]["system_prompt"])
            corpus.close()

    def test_no_read_evidence_returns_safe_no_evidence(self) -> None:
        async def loop(**kwargs):
            return Response("research only")

        agent = NovaCacAgent(Context(), lambda tracker: [], tool_loop=loop)
        answer = asyncio.run(
            agent.answer(
                Event(),
                "NOVA是什么？",
                base_system_prompt="CORE",
            )
        )
        self.assertEqual(NO_EVIDENCE, answer)

    def test_sources_are_only_shown_when_explicitly_requested(self) -> None:
        async def run(question: str) -> str:
            calls = 0

            async def loop(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    tracker = kwargs["tracker"]
                    from nova_cac.models import LocalDocument

                    tracker.add_read(
                        LocalDocument(
                            document_id="d",
                            title="原文",
                            relative_path="doc.md",
                            absolute_path=Path("doc.md"),
                            category="test",
                            source_url="https://example/source",
                            created_at="",
                            updated_at="",
                            body="事实",
                        ),
                        "事实",
                    )
                    return Response("research")
                return Response("回答。[E1]")

            agent = NovaCacAgent(Context(), lambda tracker: [], tool_loop=loop)
            return await agent.answer(Event(), question, base_system_prompt="CORE")

        plain = asyncio.run(run("这个理念是什么？"))
        sourced = asyncio.run(run("这个理念的原文出处是什么？"))
        self.assertNotIn("来源：", plain)
        self.assertIn("来源：", sourced)
        self.assertIn("https://example/source", sourced)

    def test_unknown_evidence_marker_is_rejected(self) -> None:
        calls = 0

        async def loop(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                tracker = kwargs["tracker"]
                from nova_cac.models import LocalDocument

                tracker.add_read(
                    LocalDocument(
                        document_id="d",
                        title="原文",
                        relative_path="doc.md",
                        absolute_path=Path("doc.md"),
                        category="test",
                        source_url="",
                        created_at="",
                        updated_at="",
                        body="事实",
                    ),
                    "事实",
                )
                return Response("research")
            return Response("真实内容。[E1] 虚构内容。[E999]")

        agent = NovaCacAgent(Context(), lambda tracker: [], tool_loop=loop)
        answer = asyncio.run(
            agent.answer(Event(), "NOVA是什么？", base_system_prompt="CORE")
        )
        self.assertEqual(SAFE_FAILURE, answer)

    def test_model_authored_no_evidence_cannot_smuggle_a_claim(self) -> None:
        calls = 0

        async def loop(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                tracker = kwargs["tracker"]
                from nova_cac.models import LocalDocument

                tracker.add_read(
                    LocalDocument(
                        document_id="d",
                        title="原文",
                        relative_path="doc.md",
                        absolute_path=Path("doc.md"),
                        category="test",
                        source_url="",
                        created_at="",
                        updated_at="",
                        body="事实",
                    ),
                    "事实",
                )
                return Response("research")
            return Response("资料不足，但群号是 123456。")

        agent = NovaCacAgent(Context(), lambda tracker: [], tool_loop=loop)
        answer = asyncio.run(
            agent.answer(Event(), "群号是什么？", base_system_prompt="CORE")
        )
        self.assertEqual(NO_EVIDENCE, answer)


if __name__ == "__main__":
    unittest.main()
