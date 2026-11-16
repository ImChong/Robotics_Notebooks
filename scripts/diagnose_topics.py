#!/usr/bin/env python3
"""
diagnose_topics.py — 图谱主题（社区）归属诊断

主题由 schema/topics.json 决定（见 generate_link_graph.assign_communities）；
本脚本用 Louvain 结构聚类做对照，输出 Markdown 报告到 stdout：

1. 各主题规模与来源构成（explicit / seed / tags / propagated）；
2. 结构簇与其多数主题；
3. 可疑归属：节点所在结构簇的多数主题与其主/次主题都不同（按互链度排序；
   explicit / seed 为人工固定，不参与）。

可疑项的处理方式：给页面加 frontmatter `topic:`，或调整其 tags / schema/topics.json。

用法：
  python3 scripts/diagnose_topics.py > /tmp/topic-diagnostics.md
  make topic-diagnose
"""

from __future__ import annotations

from collections import Counter, defaultdict

import generate_link_graph as glg

# 诊断用 Louvain 结构聚类配置
PRIMARY_COMMUNITY_CAP = 21
LARGE_COMMUNITY_SPLIT_RATIO = 0.40
LARGE_COMMUNITY_MIN_SIZE = 30
LOUVAIN_RESOLUTION = 1.15

# 仅对足够大、主题足够集中的结构簇报可疑项，避免小簇噪声。
MIN_CLUSTER_SIZE = 10
MIN_MAJORITY_SHARE = 0.6
MAX_SUSPECTS = 50


def connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda members: (-len(members), members[0] if members else ""))


def _merge_communities_to_cap(
    partition: list[list[str]],
    adjacency: dict[str, set[str]],
    cap: int,
) -> list[list[str]]:
    """将 Louvain 过细分区合并到不超过 cap 个社区（优先合并跨边最少的相邻小社区）。"""
    if len(partition) <= cap:
        return partition

    groups: list[set[str]] = [set(members) for members in partition]
    while len(groups) > cap:
        smallest_idx = min(range(len(groups)), key=lambda i: len(groups[i]))
        small = groups.pop(smallest_idx)
        best_j = 0
        best_cross = -1
        for j, other in enumerate(groups):
            cross = sum(
                1 for node in small for neighbor in adjacency.get(node, ()) if neighbor in other
            )
            if cross > best_cross:
                best_cross = cross
                best_j = j
        groups[best_j].update(small)

    return [sorted(members) for members in groups]


def louvain_communities(
    adjacency: dict[str, set[str]],
    resolution: float = 1.0,
) -> list[list[str]]:
    """纯 Python Louvain 单层局部移动，无外部依赖。

    模块度增益（无权图）：ΔQ = k_i_in - γ * Σ_tot * k_i / 2m
    """
    nodes = sorted(adjacency.keys())
    if not nodes:
        return []

    total_edges = sum(len(neighbors) for neighbors in adjacency.values()) / 2
    if total_edges == 0:
        return [[node] for node in nodes]

    m2 = 2 * total_edges
    degrees = {node: len(adjacency[node]) for node in nodes}
    node_to_comm = {node: idx for idx, node in enumerate(nodes)}
    comm_degree: dict[int, float] = {}
    for node in nodes:
        comm = node_to_comm[node]
        comm_degree[comm] = comm_degree.get(comm, 0.0) + degrees[node]

    improved = True
    iteration = 0
    max_iterations = 30
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for node in nodes:
            current_comm = node_to_comm[node]
            neighbor_weights: dict[int, int] = {}
            for neighbor in sorted(adjacency[node]):
                nc = node_to_comm[neighbor]
                neighbor_weights[nc] = neighbor_weights.get(nc, 0) + 1

            comm_degree[current_comm] -= degrees[node]
            k_i_in_current = neighbor_weights.get(current_comm, 0)
            best_comm = current_comm
            best_gain = k_i_in_current - resolution * comm_degree[current_comm] * degrees[node] / m2

            for candidate, k_i_in in sorted(neighbor_weights.items(), key=lambda kv: kv[0]):
                if candidate == current_comm:
                    continue
                gain = k_i_in - resolution * comm_degree.get(candidate, 0.0) * degrees[node] / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = candidate

            comm_degree[best_comm] = comm_degree.get(best_comm, 0.0) + degrees[node]
            node_to_comm[node] = best_comm
            if best_comm != current_comm:
                improved = True

    groups: dict[int, list[str]] = {}
    for node, comm in node_to_comm.items():
        groups.setdefault(comm, []).append(node)
    return sorted(
        (sorted(members) for members in groups.values()),
        key=lambda members: (-len(members), members[0] if members else ""),
    )


def refine_oversized_communities(
    partition: list[list[str]],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    """对超出阈值的巨型社区做 Louvain 二级拆分。"""
    total_nodes = sum(len(c) for c in partition)
    if total_nodes == 0:
        return partition

    refined: list[list[str]] = []
    for community in partition:
        ratio = len(community) / total_nodes
        if ratio <= LARGE_COMMUNITY_SPLIT_RATIO or len(community) < LARGE_COMMUNITY_MIN_SIZE:
            refined.append(community)
            continue

        members = set(community)
        sub_adj = {node: adjacency[node] & members for node in community}
        sub_groups = louvain_communities(sub_adj, resolution=LOUVAIN_RESOLUTION)
        if len(sub_groups) >= 2:
            refined.extend(sub_groups)
        else:
            refined.append(community)
    return refined


def detect_communities(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """主社区检测：Louvain（O(n log n) 量级），仅用于主题归属诊断对照。"""
    if not adjacency:
        return []

    partition = louvain_communities(adjacency, resolution=LOUVAIN_RESOLUTION)
    if not partition:
        partition = connected_components(adjacency)

    merged = _merge_communities_to_cap(partition, adjacency, PRIMARY_COMMUNITY_CAP)
    refined = refine_oversized_communities(merged, adjacency)
    return sorted(refined, key=lambda members: (-len(members), members[0] if members else ""))


def main() -> None:
    nodes, edges = glg._build_graph_data()
    _, community_meta = glg.assign_communities(nodes, edges)
    labels = {cid: meta["label"] for cid, meta in community_meta.items()}
    adjacency = glg.build_undirected_adjacency([n["id"] for n in nodes], edges)
    node_map = {n["id"]: n for n in nodes}

    print("# 图谱主题诊断\n")
    print(f"节点 {len(nodes)} · 边 {len(edges)} · 主题注册表 schema/topics.json\n")

    print("## 主题规模与来源\n")
    print("| 主题 | 节点 | explicit | seed | tags | propagated | 含次主题 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    members: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        members[node["community"]].append(node)
    for cid, meta in sorted(community_meta.items(), key=lambda kv: -int(kv[1]["size"])):
        group = members.get(cid, [])
        src = Counter(n["_topic_source"] for n in group)
        secondary = sum(1 for n in group if n.get("community_secondary"))
        print(
            f"| {meta['label']} | {len(group)} | {src['explicit']} | {src['seed']} | "
            f"{src['tags']} | {src['propagated']} | {secondary} |"
        )

    clusters = detect_communities(adjacency)
    print("\n## Louvain 结构簇 × 多数主题\n")
    print("| 结构簇 | 节点 | 多数主题 | 占比 |")
    print("|---:|---:|---|---:|")
    suspects: list[tuple[int, str, str, float]] = []
    for idx, cluster in enumerate(clusters):
        topic_counts = Counter(node_map[n]["community"] for n in cluster)
        major, count = topic_counts.most_common(1)[0]
        share = count / len(cluster)
        print(f"| {idx} | {len(cluster)} | {labels.get(major, major)} | {share:.0%} |")
        if len(cluster) < MIN_CLUSTER_SIZE or share < MIN_MAJORITY_SHARE:
            continue
        for node_id in cluster:
            node = node_map[node_id]
            if node["_topic_source"] in ("explicit", "seed"):
                continue
            if major in (node["community"], node.get("community_secondary")):
                continue
            suspects.append((len(adjacency[node_id]), node_id, major, share))

    print(f"\n## 可疑归属（Top {MAX_SUSPECTS}，按互链度）\n")
    if not suspects:
        print("无。")
    for degree, node_id, major, share in sorted(suspects, key=lambda s: (-s[0], s[1]))[
        :MAX_SUSPECTS
    ]:
        node = node_map[node_id]
        current = labels.get(node["community"], node["community"])
        print(
            f"- `{node_id}`（度 {degree}）：当前 {current}（{node['_topic_source']}）"
            f" → 结构上更像 {labels.get(major, major)}（簇内占比 {share:.0%}）"
        )


if __name__ == "__main__":
    main()
