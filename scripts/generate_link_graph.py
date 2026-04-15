#!/usr/bin/env python3
"""
generate_link_graph.py — Wiki 内链图谱生成工具

扫描所有 wiki 页面的内链，生成 exports/link-graph.json，
供 docs/graph.html 的 D3.js 渲染使用。

输出格式：
  {
    "nodes": [{"id": "wiki/methods/mpc.md", "label": "MPC", "type": "method"}],
    "edges": [{"source": "wiki/methods/mpc.md", "target": "wiki/concepts/wbc.md"}]
  }

用法：
  python3 scripts/generate_link_graph.py
  make graph
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
OUT_PATH = REPO_ROOT / "exports" / "link-graph.json"
STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"


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
    m = re.search(r'^# (.+)', content, re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_internal_links(content: str, source_path: Path) -> list[Path]:
    """提取页面中所有指向 wiki/ 目录内部的相对链接。"""
    targets = []
    for m in re.finditer(r'\]\(([^)]+\.md)\)', content):
        href = m.group(1).split("#")[0]
        if href.startswith("http"):
            continue
        resolved = (source_path.parent / href).resolve()
        # 只保留 wiki/ 目录内的链接
        try:
            resolved.relative_to(WIKI_DIR)
            if resolved.exists():
                targets.append(resolved)
        except ValueError:
            pass
    return targets


def main() -> None:
    nodes = []
    edges = []
    seen_edges: set[tuple[str, str]] = set()

    for page in sorted(WIKI_DIR.rglob("*.md")):
        if page.name == "README.md":
            continue
        content = page.read_text(encoding="utf-8")
        page_id = str(page.relative_to(REPO_ROOT))
        nodes.append({
            "id": page_id,
            "label": extract_title(content) or page.stem,
            "type": parse_frontmatter_type(content),
        })

        for target in extract_internal_links(content, page):
            target_id = str(target.relative_to(REPO_ROOT))
            key = (page_id, target_id)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": page_id, "target": target_id})

    graph = {"nodes": nodes, "edges": edges}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ link-graph.json: {len(nodes)} nodes, {len(edges)} edges → {OUT_PATH.relative_to(REPO_ROOT)}")

    # ── graph-stats.json ──
    # 计算度中心性、孤儿节点、类型分布
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    out_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        out_degree[e["source"]] = out_degree.get(e["source"], 0) + 1
        in_degree[e["target"]]  = in_degree.get(e["target"], 0) + 1

    total_degree = {n["id"]: in_degree.get(n["id"], 0) + out_degree.get(n["id"], 0)
                    for n in nodes}

    # Top 10 度中心性
    top_hubs = sorted(nodes, key=lambda n: total_degree.get(n["id"], 0), reverse=True)[:10]
    hub_list = [{"id": n["id"], "label": n["label"], "degree": total_degree[n["id"]]}
                for n in top_hubs]

    # 孤儿节点：入度 = 0（无其他页面指向它）
    orphans = [{"id": n["id"], "label": n["label"], "out_degree": out_degree.get(n["id"], 0)}
               for n in nodes if in_degree.get(n["id"], 0) == 0]

    # 按类型分布
    type_dist: dict[str, int] = {}
    for n in nodes:
        t = n.get("type") or "unknown"
        type_dist[t] = type_dist.get(t, 0) + 1

    stats = {
        "generated_at": __import__("datetime").date.today().isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "top_hubs": hub_list,
        "orphan_nodes": orphans,
        "type_distribution": dict(sorted(type_dist.items(), key=lambda x: x[1], reverse=True)),
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ graph-stats.json: {len(orphans)} orphans, top hub='{hub_list[0]['label'] if hub_list else '-'}' → {STATS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
