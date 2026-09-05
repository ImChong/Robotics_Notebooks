"""图谱社区命名：中文（English） 社区 格式校验（读 exports/link-graph.json 快照）。"""

from __future__ import annotations

import json
import unittest
from typing import Any

import generate_link_graph as glg

LINK_GRAPH_SNAPSHOT = glg.OUT_PATH


def _load_exported_graph() -> dict[str, Any]:
    """读取 make graph 产出的 link-graph.json，避免在 pytest 中重跑全库社区检测。"""
    if not LINK_GRAPH_SNAPSHOT.is_file():
        raise FileNotFoundError(
            f"缺少 {LINK_GRAPH_SNAPSHOT.relative_to(glg.REPO_ROOT)}；"
            "请先运行 make graph 或 make ci-preflight"
        )
    data: dict[str, Any] = json.loads(LINK_GRAPH_SNAPSHOT.read_text(encoding="utf-8"))
    return data


def _load_exported_communities() -> list[dict[str, Any]]:
    communities = _load_exported_graph().get("communities")
    if not isinstance(communities, list) or not communities:
        raise ValueError(f"{LINK_GRAPH_SNAPSHOT.name} 缺少非空 communities 数组；请重新 make graph")
    return communities


class CommunityHubNamePatternTest(unittest.TestCase):
    def test_valid_hub_names(self) -> None:
        valid = [
            "强化学习（Reinforcement Learning, RL）",
            "规模化运动跟踪（Supersizing Motion Tracking for Natural Humanoid Control, SONIC）",
            "人形硬件技术地图（Humanoid Hardware 101）",
            "机器人学习（Robot Learning）",
            "行为基础模型技术地图（Behavior Foundation Model, BFM）",
            "导航与 SLAM（Navigation and Simultaneous Localization and Mapping, SLAM）",
            "视觉-语言导航（Vision-and-Language Navigation, VLN）",
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

    def test_exported_communities_have_no_duplicate_canonical_hubs(self) -> None:
        """同一 canonical 枢纽不应出现两个命名社区（如 Manipulation 与论文深读·Manipulation）。"""
        seen: dict[str, str] = {}
        for meta in _load_exported_communities():
            if meta.get("id") == glg.OTHER_COMMUNITY_ID:
                continue
            hub_id = str(meta.get("hub_id") or "")
            canonical = glg.canonical_community_hub(hub_id)
            label = str(meta.get("label", ""))
            with self.subTest(canonical=canonical, label=label):
                if canonical in seen:
                    self.fail(
                        f"duplicate canonical hub {canonical!r}: {seen[canonical]!r} and {label!r}"
                    )
                seen[canonical] = label

    def test_exported_community_hub_belongs_to_its_own_community(self) -> None:
        """社区命名枢纽页必须是该社区成员，否则详情页会显示错误社区（如 Locomotion 被标成操作）。"""
        data = _load_exported_graph()
        node_to_community = {
            str(node["id"]): str(node.get("community", "")) for node in data["nodes"]
        }
        labels = {str(meta.get("id")): str(meta.get("label", "")) for meta in data["communities"]}
        for meta in _load_exported_communities():
            hub_id = str(meta.get("hub_id") or "")
            if not hub_id:
                continue
            community_id = str(meta.get("id"))
            with self.subTest(community_id=community_id, hub_id=hub_id):
                actual = node_to_community.get(hub_id, "")
                self.assertEqual(
                    actual,
                    community_id,
                    f"hub {hub_id!r} 命名了 {meta.get('label')!r}，"
                    f"自身却归属 {labels.get(actual, actual)!r}",
                )

    def test_exported_community_labels_conform_to_pattern(self) -> None:
        """快照里全部社区（含兜底桶）的 label 应符合 中文（English） 社区。"""
        for meta in _load_exported_communities():
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
