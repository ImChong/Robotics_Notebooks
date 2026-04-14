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


if __name__ == "__main__":
    main()
