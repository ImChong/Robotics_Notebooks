"""link-graph 节点写出 has_repo，供图谱「按开源」着色（与详情页 ⭐️ 同口径）。"""

from __future__ import annotations

import generate_link_graph as glg


def test_build_graph_nodes_write_public_has_repo() -> None:
    nodes, _edges = glg._build_graph_data()
    by_id = {str(n["id"]): n for n in nodes}

    sim2real = by_id["wiki/concepts/sim2real.md"]
    assert glg.wiki_has_repo_source(
        (glg.REPO_ROOT / "wiki/concepts/sim2real.md").read_text(encoding="utf-8")
    )
    assert sim2real.get("has_repo") is True
    assert "_has_repo_source" not in sim2real

    with_repo = [n for n in nodes if n.get("has_repo") is True]
    without = [n for n in nodes if "has_repo" not in n]
    assert with_repo, "应有关联 sources/repos 的开源节点"
    assert without, "应有未写出 has_repo 的未开源节点"
    assert all(n.get("has_repo") is True for n in with_repo)
    assert all("_has_repo_source" not in n for n in nodes)
