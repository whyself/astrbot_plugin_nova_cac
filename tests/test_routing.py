from __future__ import annotations

import unittest

from nova_cac.routing import extract_cac_query


class RoutingTests(unittest.TestCase):
    def test_private_and_group_use_the_same_cac_command(self) -> None:
        self.assertEqual("NOVA 是什么？", extract_cac_query("/cac NOVA 是什么？"))
        self.assertEqual("PBL 是什么？", extract_cac_query("/cac PBL 是什么？"))
        self.assertIsNone(extract_cac_query("NOVA 是什么？"))
        self.assertIsNone(extract_cac_query("cac NOVA 是什么？"))
        self.assertIsNone(extract_cac_query("/cache hello"))

    def test_empty_command_has_empty_query(self) -> None:
        self.assertEqual("", extract_cac_query("/cac"))


if __name__ == "__main__":
    unittest.main()
