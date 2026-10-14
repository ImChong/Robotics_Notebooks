"""link-graph 下标边与 wiki-activity 紧凑格式：写出与无损还原（前端 expandLinkGraph / expandWikiActivity 的对应口径）。"""

from __future__ import annotations

import generate_link_graph as glg


def test_pack_edges_as_indices_follows_node_order_and_drops_unknown_endpoints(capsys) -> None:
    nodes = [{"id": "wiki/a.md"}, {"id": "wiki/b.md"}, {"id": "wiki/c.md"}]
    edges = [
        {"source": "wiki/a.md", "target": "wiki/b.md"},
        {"source": "wiki/c.md", "target": "wiki/a.md"},
        {"source": "wiki/c.md", "target": "wiki/missing.md"},
    ]
    assert glg.pack_edges_as_indices(nodes, edges) == [[0, 1], [2, 0]]
    assert "1 条边" in capsys.readouterr().err


def test_pack_activity_days_compact_round_trips_every_reference() -> None:
    days = [
        {
            "date": "2026-10-01",
            "count": 2,
            "added_count": 1,
            "maintained_count": 1,
            "nodes": [
                {
                    "detail_id": "entity-a",
                    "label": "A",
                    "type": "entity",
                    "path": "wiki/entities/a.md",
                    "action": "added",
                    "has_repo": True,
                    "community_label": "控制 社区",
                },
                {
                    "detail_id": "wiki-concepts-b",
                    "label": "B",
                    "type": "concept",
                    "path": "wiki/concepts/b.md",
                },
            ],
        },
        {
            "date": "2026-10-02",
            "count": 1,
            # 同一节点各日元数据相同（都取自当前节点状态），这是去重无损的前提
            "nodes": [
                {
                    "detail_id": "entity-a",
                    "label": "A",
                    "type": "entity",
                    "path": "wiki/entities/a.md",
                    "has_repo": True,
                    "community_label": "控制 社区",
                }
            ],
        },
    ]
    node_index, packed = glg.pack_activity_days_compact(days)

    assert node_index == {
        "entity-a": {
            "label": "A",
            "type": "entity",
            "path": "wiki/entities/a.md",
            "has_repo": True,
            "community_label": "控制 社区",
        },
        "wiki-concepts-b": {"label": "B", "type": "concept", "path": "wiki/concepts/b.md"},
    }
    assert packed[0]["nodes"] == [["entity-a", "added"], ["wiki-concepts-b"]]
    assert packed[1]["nodes"] == [["entity-a"]]
    assert packed[0]["added_count"] == 1 and packed[0]["maintained_count"] == 1
    assert days[0]["nodes"][0]["label"] == "A", "input days are not mutated"

    # 与 docs/main.js expandWikiActivity 相同的还原规则
    restored = [
        {
            **day,
            "nodes": [
                {
                    "detail_id": ref[0],
                    **node_index[ref[0]],
                    **({"action": ref[1]} if len(ref) > 1 else {}),
                }
                for ref in day["nodes"]
            ],
        }
        for day in packed
    ]
    assert restored == days


def test_exported_graph_uses_index_edges() -> None:
    nodes, edges = glg._build_graph_data()
    packed = glg.pack_edges_as_indices(nodes, edges)
    assert len(packed) == len(edges), "全库内链的端点都应在节点表里"
    ids = [n["id"] for n in nodes]
    assert {(ids[s], ids[t]) for s, t in packed} == {(e["source"], e["target"]) for e in edges}
