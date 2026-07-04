"""home-stats top_communities：Top-N 社区提取、短名解析与“其他”排除的单元测试。"""

from __future__ import annotations

import unittest

import generate_home_stats as ghs


class CommunityShortLabelTest(unittest.TestCase):
    """覆盖「中文（English） 社区」→「中文」的短名解析。"""

    def test_standard_label(self) -> None:
        self.assertEqual(
            ghs.community_short_label("移动操作（Loco-Manipulation, Loco-Manip） 社区"),
            "移动操作",
        )

    def test_label_without_english_part(self) -> None:
        self.assertEqual(ghs.community_short_label("强化学习 社区"), "强化学习")

    def test_label_without_suffix(self) -> None:
        self.assertEqual(ghs.community_short_label("物理引擎（MuJoCo）"), "物理引擎")


class TopCommunitiesTest(unittest.TestCase):
    """覆盖排序、限量、“其他”排除与容错。"""

    def setUp(self) -> None:
        self.graph_stats = {
            "community_distribution": {
                "其他（Other） 社区": 999,
                "移动操作（Loco-Manipulation） 社区": 314,
                "强化学习（Reinforcement Learning, RL） 社区": 172,
                "全身控制（Whole-Body Control, WBC） 社区": 102,
                "物理引擎（MuJoCo） 社区": 79,
                "动作重定向（Motion Retargeting） 社区": 70,
                "人形论文深读笔记（Humanoid Paper Notebooks） 社区": 215,
                "操作（Manipulation） 社区": 48,
            }
        }

    def test_sorted_by_size_excluding_other(self) -> None:
        result = ghs.top_communities(self.graph_stats)
        self.assertEqual(
            [item["label"] for item in result],
            ["移动操作", "人形论文深读笔记", "强化学习", "全身控制", "物理引擎", "动作重定向"],
        )
        self.assertEqual(result[0]["size"], 314)

    def test_limit(self) -> None:
        self.assertEqual(len(ghs.top_communities(self.graph_stats, limit=3)), 3)

    def test_missing_distribution(self) -> None:
        self.assertEqual(ghs.top_communities({}), [])
        self.assertEqual(ghs.top_communities({"community_distribution": None}), [])

    def test_payload_carries_top_communities(self) -> None:
        coverage = {"covered": 1, "total": 2, "percent": 50}
        payload = ghs.build_payload(self.graph_stats, coverage)
        self.assertIn("top_communities", payload)
        self.assertEqual(len(payload["top_communities"]), 6)

    def test_payload_omits_field_when_no_communities(self) -> None:
        coverage = {"covered": 1, "total": 2, "percent": 50}
        payload = ghs.build_payload({}, coverage)
        self.assertNotIn("top_communities", payload)


if __name__ == "__main__":
    unittest.main()
