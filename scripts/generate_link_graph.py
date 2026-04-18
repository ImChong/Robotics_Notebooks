#!/usr/bin/env python3
"""
generate_link_graph.py — Wiki 内链图谱生成工具

扫描所有 wiki 页面的内链，生成 exports/link-graph.json，
供 docs/graph.html 的 D3.js 渲染使用。

输出格式：
  {
    "nodes": [
      {
        "id": "wiki/methods/mpc.md",
        "label": "MPC",
        "type": "method",
        "community": "community-0",
        "community_label": "Reinforcement Learning (RL) 社区"
      }
    ],
    "edges": [{"source": "wiki/methods/mpc.md", "target": "wiki/concepts/wbc.md"}],
    "communities": [{"id": "community-0", "label": "...", "size": 12}]
  }

用法：
  python3 scripts/generate_link_graph.py
  make graph
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
OUT_PATH = REPO_ROOT / "exports" / "link-graph.json"
STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
MAX_COMMUNITIES = 8
OTHER_COMMUNITY_ID = "community-other"
OTHER_COMMUNITY_LABEL = "其他社区"


def parse_frontmatter_type(content: str) -> str:
    if not content.startswith("---"):
        return ""
    end = content.find("\n---", 3)
    if end == -1:
        return ""
    for line in content[3:end].splitlines():
        if line.strip().startswith("type:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def extract_title(content: str) -> str:
    match = re.search(r"^# (.+)", content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_internal_links(content: str, source_path: Path) -> list[Path]:
    """提取页面中所有指向 wiki/ 目录内部的相对链接。"""
    targets = []
    for match in re.finditer(r"\]\(([^)]+\.md)\)", content):
        href = match.group(1).split("#")[0]
        if href.startswith("http"):
            continue
        resolved = (source_path.parent / href).resolve()
        try:
            resolved.relative_to(WIKI_DIR)
            if resolved.exists():
                targets.append(resolved)
        except ValueError:
            pass
    return targets


def build_undirected_adjacency(node_ids: list[str], edges: list[dict[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda members: (-len(members), members[0] if members else ""))


def edge_betweenness(adjacency: dict[str, set[str]]) -> dict[tuple[str, str], float]:
    """Brandes edge betweenness for small unweighted graphs."""
    betweenness: dict[tuple[str, str], float] = defaultdict(float)
    for source in adjacency:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {node: [] for node in adjacency}
        sigma: dict[str, float] = {node: 0.0 for node in adjacency}
        distance: dict[str, int] = {node: -1 for node in adjacency}
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            vertex = queue[head]
            head += 1
            stack.append(vertex)
            for neighbor in adjacency[vertex]:
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1
                if distance[neighbor] == distance[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        dependency: dict[str, float] = {node: 0.0 for node in adjacency}
        while stack:
            vertex = stack.pop()
            if sigma[vertex] == 0:
                continue
            for predecessor in predecessors[vertex]:
                contribution = (sigma[predecessor] / sigma[vertex]) * (1.0 + dependency[vertex])
                edge = tuple(sorted((predecessor, vertex)))
                betweenness[edge] += contribution
                dependency[predecessor] += contribution

    for edge in list(betweenness):
        betweenness[edge] /= 2.0
    return betweenness


def modularity(partition: list[list[str]], adjacency: dict[str, set[str]]) -> float:
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) / 2
    if edge_count == 0:
        return 0.0
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    score = 0.0
    for community in partition:
        community_set = set(community)
        for i in community:
            for j in community:
                a_ij = 1.0 if j in adjacency[i] else 0.0
                score += a_ij - degree[i] * degree[j] / (2 * edge_count)
    return score / (2 * edge_count)


def detect_communities(adjacency: dict[str, set[str]]) -> list[list[str]]:
    working = {node: set(neighbors) for node, neighbors in adjacency.items()}
    best_partition = connected_components(working)
    best_score = modularity(best_partition, adjacency)

    while sum(len(neighbors) for neighbors in working.values()) > 0:
        betweenness = edge_betweenness(working)
        if not betweenness:
            break
        max_value = max(betweenness.values())
        for left, right in [edge for edge, value in betweenness.items() if value == max_value]:
            working[left].discard(right)
            working[right].discard(left)
        partition = connected_components(working)
        if len(partition) > MAX_COMMUNITIES:
            break
        score = modularity(partition, adjacency)
        if len(partition) > 1 and score >= best_score:
            best_partition = partition
            best_score = score

    return best_partition


def assign_communities(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    node_ids = [node["id"] for node in nodes]
    degree_map = Counter()
    for edge in edges:
        degree_map[edge["source"]] += 1
        degree_map[edge["target"]] += 1

    adjacency = build_undirected_adjacency(node_ids, edges)
    sorted_groups = detect_communities(adjacency)

    node_map = {node["id"]: node for node in nodes}
    community_meta: dict[str, dict[str, object]] = {}
    node_to_community: dict[str, tuple[str, str]] = {}

    for idx, members in enumerate(sorted_groups):
        if idx < MAX_COMMUNITIES:
            community_id = f"community-{idx}"
            hub_id = max(members, key=lambda node_id: (degree_map.get(node_id, 0), node_map[node_id]["label"]))
            label = f"{node_map[hub_id]['label']} 社区"
        else:
            community_id = OTHER_COMMUNITY_ID
            label = OTHER_COMMUNITY_LABEL
        community_meta.setdefault(community_id, {"id": community_id, "label": label, "size": 0, "hub_id": None})
        community_meta[community_id]["size"] += len(members)
        if community_meta[community_id]["hub_id"] is None and community_id != OTHER_COMMUNITY_ID:
            community_meta[community_id]["hub_id"] = hub_id
        for node_id in members:
            node_to_community[node_id] = (community_id, label)

    for node in nodes:
        community_id, label = node_to_community.get(node["id"], (OTHER_COMMUNITY_ID, OTHER_COMMUNITY_LABEL))
        node["community"] = community_id
        node["community_label"] = label

    community_list = sorted(
        community_meta.values(),
        key=lambda item: (item["id"] == OTHER_COMMUNITY_ID, -int(item["size"]), str(item["label"])),
    )
    return community_list, community_meta


def main() -> None:
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    for page in sorted(WIKI_DIR.rglob("*.md")):
        if page.name == "README.md":
            continue
        content = page.read_text(encoding="utf-8")
        page_id = str(page.relative_to(REPO_ROOT))
        nodes.append(
            {
                "id": page_id,
                "label": extract_title(content) or page.stem,
                "type": parse_frontmatter_type(content),
            }
        )

        for target in extract_internal_links(content, page):
            target_id = str(target.relative_to(REPO_ROOT))
            key = (page_id, target_id)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": page_id, "target": target_id})

    communities, community_meta = assign_communities(nodes, edges)
    graph = {"nodes": nodes, "edges": edges, "communities": communities}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"✅ link-graph.json: {len(nodes)} nodes, {len(edges)} edges, "
        f"{len(communities)} communities → {OUT_PATH.relative_to(REPO_ROOT)}"
    )

    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    out_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    for edge in edges:
        out_degree[edge["source"]] = out_degree.get(edge["source"], 0) + 1
        in_degree[edge["target"]] = in_degree.get(edge["target"], 0) + 1

    total_degree = {
        node["id"]: in_degree.get(node["id"], 0) + out_degree.get(node["id"], 0)
        for node in nodes
    }

    top_hubs = sorted(nodes, key=lambda node: total_degree.get(node["id"], 0), reverse=True)[:10]
    hub_list = [
        {"id": node["id"], "label": node["label"], "degree": total_degree[node["id"]]}
        for node in top_hubs
    ]

    orphans = [
        {"id": node["id"], "label": node["label"], "out_degree": out_degree.get(node["id"], 0)}
        for node in nodes
        if in_degree.get(node["id"], 0) == 0
    ]

    type_dist: dict[str, int] = {}
    for node in nodes:
        node_type = node.get("type") or "unknown"
        type_dist[node_type] = type_dist.get(node_type, 0) + 1

    community_dist = {
        meta["label"]: int(meta["size"])
        for meta in sorted(community_meta.values(), key=lambda item: -int(item["size"]))
    }

    # 社区质量指标
    community_sizes = [int(meta["size"]) for meta in community_meta.values()
                       if meta["id"] != OTHER_COMMUNITY_ID]
    singleton_communities = [meta["label"] for meta in community_meta.values()
                              if int(meta["size"]) < 3 and meta["id"] != OTHER_COMMUNITY_ID]
    largest_size = max(community_sizes, default=0)
    largest_ratio = round(largest_size / max(len(nodes), 1), 3)
    community_quality_warning = largest_ratio > 0.45

    stats = {
        "generated_at": date.today().isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "community_count": len(communities),
        "top_hubs": hub_list,
        "orphan_nodes": orphans,
        "type_distribution": dict(sorted(type_dist.items(), key=lambda x: x[1], reverse=True)),
        "community_distribution": community_dist,
        "community_quality": {
            "singleton_communities": singleton_communities,
            "largest_community_ratio": largest_ratio,
            "community_quality_warning": community_quality_warning,
        },
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"✅ graph-stats.json: {len(orphans)} orphans, "
        f"top hub='{hub_list[0]['label'] if hub_list else '-'}' → {STATS_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
