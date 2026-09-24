"""link-graph 节点 summary 须来自 frontmatter summary / 正文，不得把 `title:` / `type:` 等 YAML 行当摘要（图谱浮窗）。"""

from __future__ import annotations

import re

import generate_link_graph as glg

_YAML_KEY_LINE = re.compile(r"^(title|type|status|created|updated|tags|summary|sources)\s*:")


def test_graph_node_summary_not_frontmatter_line() -> None:
    nodes, _edges = glg._build_graph_data()
    by_id = {str(n["id"]): n for n in nodes}

    assert by_id["wiki/concepts/motion-retargeting.md"]["summary"].startswith("将人类或动物参考动作映射到")
    bad = [n["id"] for n in nodes if _YAML_KEY_LINE.match(str(n.get("summary", "")))]
    assert bad == []
