"""图谱社区命名：中文（English） 社区 格式校验（读 exports/link-graph.json 快照）。"""

from __future__ import annotations

import json
import unittest
from typing import Any

import generate_link_graph as glg

LINK_GRAPH_SNAPSHOT = glg.OUT_PATH


def _load_exported_communities() -> list[dict[str, Any]]:
    """读取 make graph 产出的 link-graph.json，避免在 pytest 中重跑全库社区检测。"""
    if not LINK_GRAPH_SNAPSHOT.is_file():
        raise FileNotFoundError(
            f"缺少 {LINK_GRAPH_SNAPSHOT.relative_to(glg.REPO_ROOT)}；"
            "请先运行 make graph 或 make ci-preflight"
        )
    data = json.loads(LINK_GRAPH_SNAPSHOT.read_text(encoding="utf-8"))
    communities = data.get("communities")
    if not isinstance(communities, list) or not communities:
        raise ValueError(f"{LINK_GRAPH_SNAPSHOT.name} 缺少非空 communities 数组；请重新 make graph")
    return communities


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

    def test_community_name_overrides_match_pattern(self) -> None:
        """COMMUNITY_NAME_OVERRIDES 中每条基名应符合命名规范。"""
        for hub_id, hub_name in glg.COMMUNITY_NAME_OVERRIDES.items():
            with self.subTest(hub_id=hub_id):
                self.assertIsNotNone(
                    glg.COMMUNITY_HUB_NAME_RE.fullmatch(hub_name),
                    f"override {hub_id!r} name={hub_name!r}",
                )

    def test_exported_community_labels_conform_to_pattern(self) -> None:
        """快照里全部命名社区（除「其他社区」）的 label 应符合 中文（English） 社区。"""
        for meta in _load_exported_communities():
            if meta.get("id") == glg.OTHER_COMMUNITY_ID:
                continue
            label = str(meta.get("label", ""))
            with self.subTest(community_id=meta.get("id"), label=label):
                self.assertTrue(label.endswith(" 社区"), label)
                hub_name = label[: -len(" 社区")]
                self.assertIsNotNone(
                    glg.COMMUNITY_HUB_NAME_RE.fullmatch(hub_name),
                    f"community {meta.get('id')!r} label={label!r} hub={meta.get('hub_id')!r}",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
