"""研究机构派生：从 frontmatter tags 派生「所属机构」，institutions 字段可覆盖。"""

from __future__ import annotations

import generate_link_graph as glg

REGISTRY = {
    "nvidia": {"label": "NVIDIA", "aliases": ["nvidia", "gear"]},
    "cmu": {"label": "卡内基梅隆大学（CMU）", "aliases": ["cmu"]},
    "tsinghua": {"label": "清华大学（Tsinghua）", "aliases": ["tsinghua", "thu"]},
}
ALIASES = glg._build_institution_alias_map(REGISTRY)


def _page(front: str) -> str:
    return f"---\n{front}\n---\n\n# Demo\n"


def test_alias_map_resolves_canonical_and_aliases() -> None:
    assert ALIASES["nvidia"] == "nvidia"
    assert ALIASES["gear"] == "nvidia"  # alias -> canonical
    assert ALIASES["thu"] == "tsinghua"
    assert "unknown" not in ALIASES


def test_parse_frontmatter_list_inline_and_block() -> None:
    inline = _page("tags: [paper, cmu, nvidia]")
    assert glg.parse_frontmatter_list(inline, "tags") == ["paper", "cmu", "nvidia"]
    block = _page("institutions:\n  - nvidia\n  - cmu")
    assert glg.parse_frontmatter_list(block, "institutions") == ["nvidia", "cmu"]
    assert glg.parse_frontmatter_list(inline, "institutions") == []


def test_derive_from_tags_dedupes_and_keeps_order() -> None:
    # gear -> nvidia, nvidia -> nvidia (去重); cmu -> cmu; humanoid 非机构丢弃。
    page = _page("tags: [gear, humanoid, nvidia, cmu]")
    assert glg.derive_node_institutions(page, ALIASES) == ["nvidia", "cmu"]


def test_explicit_institutions_override_tags() -> None:
    # 显式 institutions 非空时以其为准，忽略 tags 里的 nvidia。
    page = _page("tags: [nvidia, humanoid]\ninstitutions: [cmu, tsinghua]")
    assert glg.derive_node_institutions(page, ALIASES) == ["cmu", "tsinghua"]


def test_no_institution_tags_yields_empty() -> None:
    assert glg.derive_node_institutions(_page("tags: [paper, humanoid]"), ALIASES) == []
    assert glg.derive_node_institutions("# No frontmatter\n", ALIASES) == []


def test_build_summary_sorts_by_size_then_label() -> None:
    nodes = [
        {"id": "a", "institutions": ["nvidia", "cmu"]},
        {"id": "b", "institutions": ["nvidia"]},
        {"id": "c", "institutions": ["cmu"]},
        {"id": "d"},
    ]
    summary = glg.build_institutions_summary(nodes, REGISTRY)
    assert summary[0] == {"id": "nvidia", "label": "NVIDIA", "size": 2}
    assert {"id": "cmu", "label": "卡内基梅隆大学（CMU）", "size": 2} in summary
    assert all(item["size"] >= 1 for item in summary)
