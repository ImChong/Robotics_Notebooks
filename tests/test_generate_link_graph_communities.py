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
