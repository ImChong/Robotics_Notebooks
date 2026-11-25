"""compute_node_trends：近期知识库节点趋势统计单元测试。"""

from __future__ import annotations

import unittest

import generate_link_graph as glg


def _node(
    detail_id: str,
    label: str,
    *,
    action: str = "added",
    path: str | None = None,
    node_type: str = "entity",
    community_label: str = "",
    has_repo: bool = False,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "detail_id": detail_id,
        "label": label,
        "type": node_type,
        "path": path or f"wiki/entities/{detail_id}.md",
        "action": action,
    }
    if community_label:
        entry["community_label"] = community_label
    if has_repo:
        entry["has_repo"] = True
    return entry


class ComputeNodeTrendsTest(unittest.TestCase):
    """覆盖空输入、窗口切分、环比、首次新增去重、社区排序与代表节点清单。"""

    def test_empty_activity_returns_valid_skeleton(self) -> None:
        trends = glg.compute_node_trends(
            [],
            total_nodes=120,
            total_edges=450,
            total_communities=12,
            as_of="2026-11-23",
        )
        self.assertEqual(trends["as_of"], "2026-11-23")
        self.assertEqual(trends["latest_activity_date"], "")
        self.assertEqual(trends["total_nodes"], 120)
        self.assertEqual(trends["total_edges"], 450)
        self.assertEqual(trends["total_communities"], 12)
        self.assertEqual(trends["windows"], {})
        self.assertEqual(trends["daily_recent"], [])
        self.assertEqual(trends["monthly_recent"], [])
        self.assertEqual(trends["recent_added_nodes"], [])

    def test_windows_dedup_communities_and_recent_nodes(self) -> None:
        activity_days = [
            # 超过 90 天窗口（2026-08-26..2026-11-23），但应计入 monthly_recent
            {
                "date": "2026-08-20",
                "count": 1,
                "nodes": [
                    _node(
                        "old-1",
                        "Old Node",
                        community_label="强化学习（RL） 社区",
                        node_type="concept",
                    )
                ],
            },
            # 落在 90d 与 prev_30d（2026-09-25..2026-10-24）
            {
                "date": "2026-10-15",
                "count": 2,
                "maintained_count": 1,
                "nodes": [
                    _node(
                        "oct-1",
                        "Oct Concept",
                        community_label="全身控制（WBC） 社区",
                        node_type="concept",
                    ),
                    _node(
                        "old-1-m",
                        "Old Node",
                        path="wiki/entities/old-1.md",
                        action="maintained",
                    ),
                ],
            },
            # 落在 30d（2026-10-25..2026-11-23）与 prev_14d（2026-10-27..2026-11-09）
            {
                "date": "2026-11-05",
                "count": 2,
                "nodes": [
                    _node(
                        "nov-1",
                        "Nov Method",
                        community_label="视觉-语言-动作（VLA） 社区",
                        node_type="method",
                        has_repo=True,
                    ),
                    # 与 10-15 同 path 的重复 added，应按最早日期去重，不计入 11-05 新增
                    _node(
                        "oct-1",
                        "Oct Concept Duplicate",
                        path="wiki/entities/oct-1.md",
                        community_label="全身控制（WBC） 社区",
                    ),
                ],
            },
            # 落在 14d（2026-11-10..2026-11-23）与 prev_7d（2026-11-10..2026-11-16）
            {
                "date": "2026-11-14",
                "count": 2,
                "nodes": [
                    _node(
                        "nov-2",
                        "Tactile Paper",
                        community_label="触觉感知（Tactile Sensing） 社区",
                        has_repo=True,
                    ),
                    _node("nov-1", "Nov Method", action="maintained"),
                ],
            },
            # 落在 7d（2026-11-17..2026-11-23）
            {
                "date": "2026-11-20",
                "count": 2,
                "nodes": [
                    _node(
                        "nov-3",
                        "Tactile Rig",
                        community_label="触觉感知（Tactile Sensing） 社区",
                        has_repo=True,
                    ),
                    _node(
                        "nov-other",
                        "Other Bucket Node",
                        community_label="其他（Other） 社区",
                        node_type="query",
                    ),
                ],
            },
            {
                "date": "2026-11-23",
                "count": 3,
                "nodes": [
                    _node(
                        "nov-4",
                        "VLA Policy",
                        community_label="视觉-语言-动作（VLA） 社区",
                        node_type="method",
                    ),
                    _node(
                        "nov-5",
                        "Tactile Data",
                        community_label="触觉感知（Tactile Sensing） 社区",
                        has_repo=False,
                    ),
                    _node("nov-2", "Tactile Paper", action="maintained"),
                ],
            },
        ]

        trends = glg.compute_node_trends(
            activity_days,
            total_nodes=8000,
            total_edges=65000,
            total_communities=36,
            as_of="2026-11-23",
            recent_nodes_limit=4,
        )

        self.assertEqual(trends["latest_activity_date"], "2026-11-23")
        w7 = trends["windows"]["7d"]
        self.assertEqual(w7["start_date"], "2026-11-17")
        self.assertEqual(w7["end_date"], "2026-11-23")
        self.assertEqual(w7["added"], 4)
        self.assertEqual(w7["prev_added"], 1)  # 11-14
        self.assertEqual(w7["maintained"], 1)
        self.assertEqual(w7["prev_maintained"], 1)
        self.assertEqual(w7["active_days"], 2)
        self.assertEqual(w7["open_source_added"], 1)  # nov-3
        # "其他" 社区应被跳过；触觉感知 (+2) 排第一
        self.assertEqual(
            w7["top_communities"],
            [
                {"label": "触觉感知", "added": 2},
                {"label": "视觉-语言-动作", "added": 1},
            ],
        )
        self.assertEqual(w7["type_breakdown"], {"entity": 2, "method": 1, "query": 1})

        w14 = trends["windows"]["14d"]
        self.assertEqual(w14["added"], 5)
        self.assertEqual(w14["prev_added"], 1)  # 11-05 (oct-1 dup skipped)
        self.assertEqual(w14["open_source_added"], 2)

        w30 = trends["windows"]["30d"]
        self.assertEqual(w30["added"], 6)
        self.assertEqual(w30["prev_added"], 1)  # 10-15
        self.assertEqual(w30["prev_maintained"], 1)

        w90 = trends["windows"]["90d"]
        self.assertEqual(w90["added"], 7)  # excludes 08-20

        # daily_recent 与 monthly_recent
        self.assertEqual(
            [d["date"] for d in trends["daily_recent"]][-2:], ["2026-11-20", "2026-11-23"]
        )
        self.assertEqual(
            trends["monthly_recent"],
            [
                {"month": "2026-08", "added": 1, "maintained": 0},
                {"month": "2026-10", "added": 1, "maintained": 1},
                {"month": "2026-11", "added": 6, "maintained": 2},
            ],
        )

        # recent_added_nodes 按日期倒序、限长 4
        recent = trends["recent_added_nodes"]
        self.assertEqual(len(recent), 4)
        self.assertEqual([n["detail_id"] for n in recent], ["nov-4", "nov-5", "nov-3", "nov-other"])
        self.assertEqual(recent[0]["date"], "2026-11-23")
        self.assertEqual(recent[0]["community_label"], "视觉-语言-动作")
        self.assertNotIn("community_label", recent[3])  # "其他" 不写入
        self.assertTrue(recent[2]["has_repo"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
