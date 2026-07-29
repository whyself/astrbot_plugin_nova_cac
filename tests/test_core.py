from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova_cac.core import ConversationMemory, PackLoader


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
