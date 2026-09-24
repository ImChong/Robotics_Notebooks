"""图谱社区命名：中文（English） 社区 格式校验（读 exports/link-graph.json 快照）。"""

from __future__ import annotations

import json
import unittest
from typing import Any

import generate_link_graph as glg
from utils.community_labels import COMMUNITY_NAME_OVERRIDES

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
        """COMMUNITY_NAME_OVERRIDES（历史基名，现作搜索别名）每条应符合命名规范。"""
        for hub_id, hub_name in COMMUNITY_NAME_OVERRIDES.items():
            with self.subTest(hub_id=hub_id):
                self.assertIsNotNone(
                    glg.COMMUNITY_HUB_NAME_RE.fullmatch(hub_name),
                    f"override {hub_id!r} name={hub_name!r}",
                )

    def test_topic_labels_match_pattern(self) -> None:
        """schema/topics.json 每个主题 label 应符合 中文（English）。"""
        for topic in glg.TOPICS:
            with self.subTest(topic=topic["id"]):
                self.assertIsNotNone(glg.COMMUNITY_HUB_NAME_RE.fullmatch(topic["label"]))

    def test_exported_community_ids_are_registered_topics(self) -> None:
        """快照里的命名社区 id 均来自注册表（community-<topic-id>），不再是按规模排序的下标。"""
        registered = {glg.topic_community_id(tid) for tid in glg.TOPIC_BY_ID}
        for meta in _load_exported_communities():
            if meta.get("id") == glg.OTHER_COMMUNITY_ID:
                continue
            with self.subTest(community_id=meta.get("id")):
                self.assertIn(meta.get("id"), registered)

    def test_exported_community_hub_belongs_to_its_own_community(self) -> None:
        """主题锚点页（seeds[0]）必须归属本主题，否则详情页会显示错误社区（如 Locomotion 被标成操作）。"""
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
