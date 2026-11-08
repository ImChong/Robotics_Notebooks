"""健康度评分：frontmatter 规则与 roadmap/ 正文信号规则。

回归背景：roadmap/ 路线页按约定不带 frontmatter，旧逻辑统一走
``compute_health_score`` 导致 14 个路线节点健康度恒为 0。
"""

from __future__ import annotations

from pathlib import Path

import generate_link_graph as glg
import pytest
from scaffold_wiki_page import build_skeleton


def test_compute_health_score_requires_frontmatter() -> None:
    """无 frontmatter 页面在 wiki 规则下仍为 0（既有行为不变）。"""
    assert glg.compute_health_score("# 标题\n\n正文。\n", Path("wiki/x.md")) == 0


def test_roadmap_full_signals_score_three() -> None:
    content = "# 路线（纵深）：示例\n\n**摘要**：一句话摘要。\n\n## Stage 0 全景\n\n正文。\n"
    assert glg.compute_roadmap_health_score(content, internal_link_count=10) == 3


def test_roadmap_main_route_headings_count_as_stage_structure() -> None:
    """主路线的 ``## L−1``…``## L7`` 分层标题同样算阶段化结构。"""
    content = "# 主路线\n\n**首屏导读**：\n\n## L−1 序言\n\n## L0 数学\n"
    assert glg.compute_roadmap_health_score(content, internal_link_count=0) == 2


def test_roadmap_missing_all_signals_scores_zero() -> None:
    assert glg.compute_roadmap_health_score("# 标题\n\n正文。\n", internal_link_count=0) == 0


def test_roadmap_link_threshold_boundary() -> None:
    content = "# 标题\n\n正文。\n"
    threshold = glg.ROADMAP_HEALTH_MIN_INTERNAL_LINKS
    assert glg.compute_roadmap_health_score(content, internal_link_count=threshold - 1) == 0
    assert glg.compute_roadmap_health_score(content, internal_link_count=threshold) == 1


def test_all_repo_roadmap_pages_score_full() -> None:
    """仓库现有路线页应全部拿满 3 分（曾全部为 0 的回归防线）。"""
    pages = [p for p in sorted(glg.ROADMAP_DIR.glob("*.md")) if p.name != "README.md"]
    assert pages, "roadmap/ 下应存在路线页"
    for page in pages:
        content = page.read_text(encoding="utf-8")
        links = glg.extract_internal_links(content, page)
        score = glg.compute_roadmap_health_score(content, len(links))
        assert score == 3, f"{page.name} 健康度 {score} != 3"


def _skeleton(status: str = "complete", original: str | None = None) -> str:
    """scaffold 生成的标准骨架（覆盖全部阅读骨架小节），可替换 status、追加 original。"""
    content = build_skeleton("concept", "示例").replace("status: draft", f"status: {status}")
    if original:
        content = content.replace(
            "\nsources:", f"\noriginal:\n  - ../../sources/{original}\nsources:", 1
        )
    return content


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离仓库：sources/ 下 done.md 已完成、todo.md 未完成、bare.md 无通读字段。"""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "done.md").write_text("- **通读：** 已完成\n", encoding="utf-8")
    (sources / "todo.md").write_text("- **通读：** 未完成\n", encoding="utf-8")
    (sources / "bare.md").write_text("- **核对范围：** 已通读全文。\n", encoding="utf-8")
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    monkeypatch.setattr(glg, "SOURCES_DIR", sources)
    return tmp_path


def _score(repo: Path, content: str) -> int:
    return glg.compute_health_score(content, repo / "wiki" / "concepts" / "page.md")


def test_health_score_signals(repo: Path) -> None:
    """元数据 / 骨架达标 / 原文读完 各 +1；原文只认 original 指向且标「已完成」的归档。"""
    assert _score(repo, _skeleton(original="done.md")) == 3
    assert _score(repo, _skeleton(original="todo.md")) == 2
    assert _score(repo, _skeleton(original="bare.md")) == 2
    assert _score(repo, _skeleton()) == 2
    minimal = "---\nsummary: x\nsources: [a]\noriginal: ../../sources/done.md\n---\n\n正文。\n"
    assert _score(repo, minimal) == 2


def test_every_original_must_be_done(repo: Path) -> None:
    """声明多份原文时，任一份未完成就不算读完。"""
    content = _skeleton(original="done.md").replace(
        "  - ../../sources/done.md\n", "  - ../../sources/done.md\n  - ../../sources/todo.md\n"
    )
    assert _score(repo, content) == 2


def test_status_caps_health_by_ingest_completion(repo: Path) -> None:
    """作者自报未完成的页按 status 封顶：stub ≤1，draft ≤2，complete 不封顶。"""
    assert _score(repo, _skeleton("stub", "done.md")) == 1
    assert _score(repo, _skeleton("draft", "done.md")) == 2
    assert _score(repo, _skeleton("complete", "done.md")) == 3


def test_status_cap_does_not_raise_low_score(repo: Path) -> None:
    """封顶只降不升。"""
    assert _score(repo, "---\nstatus: complete\n---\n\n正文。\n") == 0


def test_reading_skeleton_accepts_lint_synonyms() -> None:
    """论文页常用的「核心机制」「局限与工程实践」视为覆盖核心原理 / 工程实践 / 局限与风险。"""
    heads = ["一句话定义", "英文缩写速查", "为什么重要", "核心机制", "局限与工程实践"]
    heads += ["关联页面", "参考来源", "推荐继续阅读"]
    content = "\n\n".join(f"## {h}\n\n正文" for h in heads)
    assert glg.has_reading_skeleton(content, "entity")
    assert not glg.has_reading_skeleton(content.replace("## 为什么重要", "## 核心信息"), "entity")


def test_query_reading_skeleton_uses_query_sections() -> None:
    content = "## 英文缩写速查\n\n## TL;DR 决策结论\n\n## 关联页面\n\n## 参考来源\n"
    assert glg.has_reading_skeleton(content, "query")
    assert not glg.has_reading_skeleton(content, "concept")


def _gaps(repo: Path, content: str) -> int:
    return glg.compute_health(content, repo / "wiki" / "concepts" / "page.md")[1]


def test_health_gaps_mark_each_missing_item(repo: Path) -> None:
    """缺项位：1 元数据 / 2 阅读骨架 / 4 原文通读 / 8 status 为 stub|draft；得分不变。"""
    assert _gaps(repo, _skeleton(original="done.md")) == 0
    assert _gaps(repo, _skeleton(original="todo.md")) == 4
    minimal = "---\nsummary: x\nsources: [a]\noriginal: ../../sources/done.md\n---\n\n正文。\n"
    assert _gaps(repo, minimal) == 2
    assert _gaps(repo, _skeleton("draft", "done.md")) == glg.HEALTH_GAP_STATUS
    assert _gaps(repo, "# 标题\n\n正文。\n") == 7
    for content in (_skeleton(original="todo.md"), _skeleton("stub", "done.md"), minimal):
        score, _ = glg.compute_health(content, repo / "wiki" / "concepts" / "page.md")
        assert score == _score(repo, content)


def test_roadmap_health_gaps() -> None:
    """路线页缺项位：1 首屏导读 / 2 站内链接 / 4 阶段化结构。"""
    content = "# 主路线\n\n**首屏导读**：\n\n## L0 数学\n"
    assert glg.compute_roadmap_health(content, internal_link_count=0) == (2, 2)
    assert glg.compute_roadmap_health("# 标题\n", internal_link_count=10) == (1, 5)


def test_list_rows_carry_health_fields() -> None:
    """榜单 / 更新记录行写出 health_score + health_gaps。"""
    node = {"health_score": 2, "health_gaps": 4}
    assert glg._health_fields(node) == {"health_score": 2, "health_gaps": 4}
    assert glg._health_fields({"health_score": 3}) == {"health_score": 3, "health_gaps": 0}
    assert glg._health_fields({}) == {}


@pytest.mark.full_only
def test_graph_nodes_carry_public_health_gaps() -> None:
    """link-graph 节点写出 health_gaps，图谱浮窗 / 侧栏据此列出缺项。"""
    nodes, _edges = glg._build_graph_data()
    assert all(isinstance(n.get("health_gaps"), int) for n in nodes)
    assert all("_health_gaps" not in n for n in nodes)


def test_route_health_distribution_counts_route_map_nodes() -> None:
    """纵深路线健康度：按路线知识地图节点统计 0–3 分个数，未知节点忽略、无节点的路线不出现。"""
    nodes = [
        {
            "detail_id": "roadmap-depth-x",
            "label": "X",
            "health_score": 3,
            "_route_items": ["a", "b", "c", "zz"],
        },
        {"detail_id": "roadmap-depth-y", "label": "Y", "health_score": 3, "_route_items": []},
        {"detail_id": "a", "label": "A", "health_score": 3},
        {"detail_id": "b", "label": "B", "health_score": 1},
        {"detail_id": "c", "label": "C", "health_score": 1},
    ]
    assert glg.route_health_distribution(nodes) == [
        {"detail_id": "roadmap-depth-x", "label": "X", "counts": [0, 2, 0, 1]}
    ]


@pytest.mark.full_only
def test_repo_routes_have_route_map_nodes() -> None:
    """roadmap/ 下的主路线与纵深路线都能解析出知识地图节点（图表不会缺路线）。"""
    nodes, _edges = glg._build_graph_data()
    routes = [n for n in nodes if str(n["id"]).startswith("roadmap/")]
    assert routes and all(n.get("_route_items") for n in routes)
