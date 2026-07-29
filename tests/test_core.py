from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova_cac.core import ConversationMemory, KnowledgeIndex, PackLoader


class PackLoaderTests(unittest.TestCase):
    def test_core_files_are_reread_for_every_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in PackLoader.CORE_FILES:
                (root / name).write_text(f"{name}:v1", encoding="utf-8")

            loader = PackLoader(root)
            first = loader.build_system_prompt()
            (root / "soul.md").write_text("soul:v2", encoding="utf-8")
            second = loader.build_system_prompt()

            self.assertIn("soul.md:v1", first)
            self.assertIn("soul:v2", second)
            self.assertNotIn("soul.md:v1", second)
            for name in PackLoader.CORE_FILES:
                self.assertIn(f"文件：{name}", second)


class KnowledgeIndexTests(unittest.TestCase):
    def test_retrieval_prefers_current_rules_for_membership_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "knowledge"
            current_dir = root / "03_规章与活动"
            history_dir = root / "01_认识NOVA"
            current_dir.mkdir(parents=True)
            history_dir.mkdir(parents=True)
            (current_dir / "NOVA管理章程（2026版）.md").write_text(
                "---\n"
                'title: "NOVA管理章程（2026版）"\n'
                'updated_at: "2026-07-23"\n'
                "---\n"
                "# 章程\n\n"
                "无论是否具备技术基础，只要认同共同价值观，NOVA欢迎本科学生加入。",
                encoding="utf-8",
            )
            (history_dir / "旧活动记录.md").write_text(
                "# 旧活动记录\n\n曾经讨论过加入流程与技术基础。",
                encoding="utf-8",
            )

            results = KnowledgeIndex(root).search(
                "没有技术基础能加入 NOVA 吗？",
                top_k=2,
                max_chars=4000,
            )

            self.assertTrue(results)
            self.assertEqual("NOVA管理章程（2026版）", results[0].title)
            self.assertIn("欢迎本科学生加入", results[0].content)

    def test_embedded_fall_activity_query_prefers_current_plan(self) -> None:
        pack_root = Path(__file__).resolve().parents[1] / "knowledge_pack"
        results = KnowledgeIndex(pack_root / "knowledge").search(
            "2026 秋季有哪些活动？",
            top_k=3,
            max_chars=5000,
        )

        self.assertTrue(results)
        self.assertEqual("NOVA-2026秋活动方案", results[0].title)


class ConversationMemoryTests(unittest.TestCase):
    def test_memory_is_bounded_and_can_be_cleared(self) -> None:
        memory = ConversationMemory(history_turns=2, max_sessions=2)
        memory.append_exchange("a", "q1", "a1")
        memory.append_exchange("a", "q2", "a2")
        memory.append_exchange("a", "q3", "a3")

        self.assertEqual(
            [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
            memory.contexts("a"),
        )

        memory.append_exchange("b", "qb", "ab")
        memory.append_exchange("c", "qc", "ac")
        self.assertEqual([], memory.contexts("a"))
        self.assertTrue(memory.clear("b"))
        self.assertEqual([], memory.contexts("b"))
        self.assertFalse(memory.clear("missing"))

    def test_memory_also_respects_character_budget(self) -> None:
        memory = ConversationMemory(
            history_turns=10,
            max_sessions=2,
            max_chars=100,
        )
        memory.append_exchange("a", "q1" * 10, "a1" * 20)
        memory.append_exchange("a", "q2" * 10, "a2" * 20)

        contexts = memory.contexts("a")
        self.assertEqual(2, len(contexts))
        self.assertEqual("q2" * 10, contexts[0]["content"])
        self.assertLessEqual(
            sum(len(item["content"]) for item in contexts),
            100,
        )

    def test_only_explicitly_appended_cac_exchanges_exist(self) -> None:
        memory = ConversationMemory(history_turns=6)
        self.assertEqual([], memory.contexts("group:1"))
        memory.append_exchange("group:1", "/cac 问题", "回答")
        self.assertEqual(2, len(memory.contexts("group:1")))


if __name__ == "__main__":
    unittest.main()
