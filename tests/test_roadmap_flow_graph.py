"""Roadmap flow_graph (Cytoscape elements) generation."""

import unittest

from export_minimal import build_roadmap_flow_graph


class RoadmapFlowGraphTests(unittest.TestCase):
    def test_requires_two_stages(self):
        item = {"id": "roadmap-x", "stages": [{"id": "l0", "title": "only"}]}
        self.assertIsNone(build_roadmap_flow_graph(item, {}))

    def test_motion_control_includes_dual_trunk(self):
        item = {
            "id": "roadmap-motion-control",
            "stages": [
                {"id": "l0", "title": "数学与编程基础", "related_items": []},
                {"id": "l1", "title": "机器人学骨架", "related_items": ["wiki-concepts-test"]},
            ],
        }
        item_map = {"wiki-concepts-test": {"title": "测试概念页"}}
        fg = build_roadmap_flow_graph(item, item_map)
        self.assertIsNotNone(fg)
        assert fg is not None
        self.assertEqual(fg["version"], 1)
        self.assertIn("overview", fg)
        self.assertIsNotNone(fg.get("dual_trunk"))
        dual = fg["dual_trunk"]
        assert dual is not None
        ids = {el["data"].get("id") for el in dual["elements"] if "id" in el.get("data", {})}
        self.assertIn("mc-dual-T", ids)
        self.assertIn("mc-dual-L", ids)

    def test_other_roadmap_has_no_dual_trunk(self):
        item = {
            "id": "roadmap-if-goal-locomotion-rl",
            "stages": [
                {"id": "l0", "title": "A", "related_items": []},
                {"id": "l1", "title": "B", "related_items": []},
            ],
        }
        fg = build_roadmap_flow_graph(item, {})
        self.assertIsNotNone(fg)
        assert fg is not None
        self.assertIsNone(fg.get("dual_trunk"))


if __name__ == "__main__":
    unittest.main()
