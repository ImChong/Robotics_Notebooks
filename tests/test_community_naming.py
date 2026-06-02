"""图谱社区命名：中文（English） 社区 格式校验与 override 覆盖。"""

from __future__ import annotations

import unittest

import generate_link_graph as glg


class CommunityHubNamePatternTest(unittest.TestCase):
    def test_valid_hub_names(self) -> None:
        valid = [
            "强化学习（Reinforcement Learning, RL）",
            "规模化运动跟踪（SONIC）",
            "人形硬件技术地图（Humanoid Hardware 101）",
            "机器人学习（Robot Learning）",
            "行为基础模型技术地图（BFM）",
            "导航与 SLAM（Navigation / 自动驾驶）",
        ]
        for name in valid:
            with self.subTest(name=name):
                self.assertIsNotNone(glg.COMMUNITY_HUB_NAME_RE.fullmatch(name))

    def test_invalid_hub_names(self) -> None:
        invalid = [
            "SONIC（规模化运动跟踪人形控制）",
            "Robot Learning Overview",
            "Humanoid Hardware 101：七类子系统技术地图",
            "Reinforcement Learning (RL, 强化学习)",
        ]
        for name in invalid:
            with self.subTest(name=name):
                self.assertIsNone(glg.COMMUNITY_HUB_NAME_RE.fullmatch(name))

    def test_overrides_cover_current_hubs(self) -> None:
        """当前图谱全部命名社区（除「其他社区」）均应有 conforming override。"""
        nodes, edges = glg._build_graph_data()
        _communities, community_meta = glg.assign_communities(nodes, edges)
        for meta in community_meta.values():
            if meta["id"] == glg.OTHER_COMMUNITY_ID:
                continue
            label = str(meta["label"])
            self.assertTrue(label.endswith(" 社区"), label)
            hub_name = label[: -len(" 社区")]
            self.assertIsNotNone(
                glg.COMMUNITY_HUB_NAME_RE.fullmatch(hub_name),
                f"community {meta['id']!r} label={label!r} hub={meta.get('hub_id')!r}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
