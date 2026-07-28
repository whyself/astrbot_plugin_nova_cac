from __future__ import annotations

import unittest

from nova_cac.routing import command_allowed, extract_cac_query


class RoutingTests(unittest.TestCase):
    def test_private_requires_cac_command(self) -> None:
        self.assertTrue(command_allowed(is_private=True, bot_mentioned=False))
        self.assertEqual("NOVA 是什么？", extract_cac_query("/cac NOVA 是什么？"))
        self.assertIsNone(extract_cac_query("NOVA 是什么？"))
        self.assertIsNone(extract_cac_query("cac NOVA 是什么？"))
        self.assertIsNone(extract_cac_query("/cache hello"))

    def test_group_requires_cac_command_and_bot_mention(self) -> None:
        self.assertFalse(command_allowed(is_private=False, bot_mentioned=False))
        self.assertTrue(command_allowed(is_private=False, bot_mentioned=True))
        self.assertEqual("PBL 是什么？", extract_cac_query("@NovaBot /cac PBL 是什么？"))

    def test_empty_command_has_empty_query(self) -> None:
        self.assertEqual("", extract_cac_query("/cac"))
        self.assertEqual("", extract_cac_query("@NovaBot /cac   "))


if __name__ == "__main__":
    unittest.main()
