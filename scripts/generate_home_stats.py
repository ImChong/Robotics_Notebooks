#!/usr/bin/env python3
"""generate_home_stats.py — 生成首页 Hero 统计所需的轻量 JSON。"""
import json
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GRAPH_STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
OUT_PATH = REPO_ROOT / "exports" / "home-stats.json"
HUMANOID_SITEMAP_URL = "https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/sitemap.xml"
HUMANOID_NOTES_URL_PREFIX = "https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/"

if not GRAPH_STATS_PATH.exists():
    print(f"Missing {GRAPH_STATS_PATH}. Run make graph first.", file=sys.stderr)
    sys.exit(1)

graph_stats = json.loads(GRAPH_STATS_PATH.read_text(encoding="utf-8"))
lint_result = subprocess.run(
    ["python3", "scripts/lint_wiki.py"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
    check=False, # We want to continue even if lint fails
)
all_output = lint_result.stdout + lint_result.stderr
coverage_match = re.search(r"Sources 覆盖率：(\d+)/(\d+) \((\d+)%\)", all_output)
if not coverage_match:
    print("Failed to parse Sources coverage from lint_wiki.py output", file=sys.stderr)
    print("Full output:", file=sys.stderr)
    print(all_output[-1000:], file=sys.stderr)
    sys.exit(1)


def fallback_notes_count() -> tuple[int, str]:
    """在网络不可达时，从已有 home-stats.json 回退 notes 计数。"""
    if OUT_PATH.exists():
        try:
            old_payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            notes = old_payload.get("paper_notes") or {}
            count = int(notes.get("count", 0))
            if count >= 0:
                return count, "fallback: previous home-stats.json"
        except Exception:
            pass
    return 0, "fallback: no cache"


def fetch_humanoid_notes_count() -> tuple[int, str]:
    """
    统计 Humanoid_Robot_Learning_Paper_Notebooks 的可访问笔记页数量。
    口径：sitemap.xml 中位于 /papers/ 下且以 .html 结尾的唯一 URL 数量。
    """
    try:
        with urllib.request.urlopen(HUMANOID_SITEMAP_URL, timeout=15) as response:
            content = response.read()
        root = ET.fromstring(content)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        loc_nodes = root.findall(".//sm:loc", namespace)
        urls = {
            (node.text or "").strip()
            for node in loc_nodes
            if node.text
            and (node.text or "").strip().startswith(HUMANOID_NOTES_URL_PREFIX)
            and (node.text or "").strip().endswith(".html")
        }
        return len(urls), "sitemap"
    except Exception:
        return fallback_notes_count()


notes_count, notes_source = fetch_humanoid_notes_count()

payload = {
    "generated_at": graph_stats.get("generated_at"),
    "node_count": graph_stats.get("node_count"),
    "edge_count": graph_stats.get("edge_count"),
    "coverage": {
        "covered": int(coverage_match.group(1)),
        "total": int(coverage_match.group(2)),
        "percent": int(coverage_match.group(3)),
    },
    "paper_notes": {
        "count": notes_count,
        "source": notes_source,
        "definition": "Humanoid_Paper_Notebooks sitemap 中 /papers/ 下 .html 页面唯一计数",
    },
}
OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(
    f"Wrote {OUT_PATH} with graph={payload['node_count']} nodes/{payload['edge_count']} edges, "
    f"coverage={payload['coverage']['covered']}/{payload['coverage']['total']}, "
    f"paper_notes={payload['paper_notes']['count']} ({payload['paper_notes']['source']})"
)
