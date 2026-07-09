"""home-stats top_hubs / top_paper_hubs：首页互链枢纽字段透传的单元测试。"""

from __future__ import annotations

import unittest

import generate_home_stats as ghs


class HubEntriesTest(unittest.TestCase):
    """覆盖字段裁剪、缺 detail_id 条目的跳过与容错。"""

    def setUp(self) -> None:
        self.graph_stats = {
            "top_hubs": [
                {
                    "id": "wiki/concepts/sim2real.md",
                    "detail_id": "wiki-concepts-sim2real",
                    "label": "Sim2Real",
                    "type": "concept",
                    "degree": 262,
                },
                # 旧格式条目（无 detail_id）：跳过，避免首页渲染出断链
                {
                    "id": "wiki/tasks/loco-manipulation.md",
                    "label": "Loco-Manipulation",
                    "degree": 258,
                },
            ],
            "top_paper_hubs": [
                {
                    "id": "wiki/entities/paper-omniretarget.md",
                    "detail_id": "wiki-entities-paper-omniretarget",
                    "label": "OmniRetarget",
                    "type": "entity",
                    "degree": 42,
                },
            ],
        }

    def test_keeps_only_site_link_fields(self) -> None:
        result = ghs.hub_entries(self.graph_stats, "top_hubs")
        self.assertEqual(
            result,
            [
                {
                    "detail_id": "wiki-concepts-sim2real",
                    "label": "Sim2Real",
                    "type": "concept",
                    "degree": 262,
                }
            ],
        )

    def test_missing_or_invalid_key(self) -> None:
        self.assertEqual(ghs.hub_entries({}, "top_hubs"), [])
        self.assertEqual(ghs.hub_entries({"top_hubs": None}, "top_hubs"), [])

    def test_payload_carries_both_hub_lists(self) -> None:
        coverage = {"covered": 1, "total": 2, "percent": 50}
        payload = ghs.build_payload(self.graph_stats, coverage)
        self.assertEqual(len(payload["top_hubs"]), 1)
        self.assertEqual(
            payload["top_paper_hubs"][0]["detail_id"], "wiki-entities-paper-omniretarget"
        )

    def test_payload_omits_fields_when_no_hubs(self) -> None:
        coverage = {"covered": 1, "total": 2, "percent": 50}
        payload = ghs.build_payload({}, coverage)
        self.assertNotIn("top_hubs", payload)
        self.assertNotIn("top_paper_hubs", payload)


if __name__ == "__main__":
    unittest.main()
