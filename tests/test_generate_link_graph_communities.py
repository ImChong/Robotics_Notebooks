"""Louvain 社区检测与合并上限（替代 Girvan-Newman 后的行为与性能）。"""

from __future__ import annotations

import time

import generate_link_graph as glg


def test_detect_communities_completes_quickly_on_medium_graph() -> None:
    """813 节点量级全库曾需 ~15min；合成 200 节点图应在 1s 内完成。"""
    n = 200
    adjacency: dict[str, set[str]] = {f"n{i}": set() for i in range(n)}
    for i in range(n - 1):
        adjacency[f"n{i}"].add(f"n{i + 1}")
        adjacency[f"n{i + 1}"].add(f"n{i}")
    for i in range(0, n - 10, 10):
        adjacency[f"n{i}"].add(f"n{i + 5}")
        adjacency[f"n{i + 5}"].add(f"n{i}")

    t0 = time.perf_counter()
    partition = glg.detect_communities(adjacency)
    elapsed = time.perf_counter() - t0

    assert partition
    assert sum(len(c) for c in partition) == n
    assert elapsed < 2.0, f"detect_communities too slow: {elapsed:.2f}s"


def test_merge_partition_by_hub_equivalence_merges_alias_hubs() -> None:
    """Paper Notebooks 分类页与对应 task 页应合并为同一社区分区。"""
    partition = [
        [
            "wiki/overview/paper-notebook-category-06-manipulation.md",
            "wiki/entities/paper-a.md",
        ],
        ["wiki/tasks/manipulation.md", "wiki/methods/foo.md"],
    ]
    degree_map = glg.Counter(
        {
            "wiki/overview/paper-notebook-category-06-manipulation.md": 50,
            "wiki/entities/paper-a.md": 1,
            "wiki/tasks/manipulation.md": 10,
            "wiki/methods/foo.md": 2,
        }
    )
    node_map = {
        nid: {"id": nid, "label": nid.split("/")[-1]}
        for nid in degree_map
    }

    merged = glg._merge_partition_by_hub_equivalence(partition, degree_map, node_map)
    assert len(merged) == 1
    assert sum(len(group) for group in merged) == 4


def test_merge_communities_to_cap_merges_smallest() -> None:
    partition = [["a", "b"], ["c"], ["d", "e", "f"]]
    adjacency = {
        "a": {"b", "c"},
        "b": {"a"},
        "c": {"a", "d"},
        "d": {"e", "f", "c"},
        "e": {"d", "f"},
        "f": {"d", "e"},
    }
    merged = glg._merge_communities_to_cap(partition, adjacency, cap=2)
    assert len(merged) == 2
    assert sum(len(c) for c in merged) == 6
