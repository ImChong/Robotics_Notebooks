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

# 仅对足够大、主题足够集中的结构簇报可疑项，避免小簇噪声。
MIN_CLUSTER_SIZE = 10
MIN_MAJORITY_SHARE = 0.6
MAX_SUSPECTS = 50


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

    clusters = glg.detect_communities(adjacency)
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
