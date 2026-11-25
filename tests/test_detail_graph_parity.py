"""Regression tests for 1:1 parity between published detail pages and link-graph nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import check_export_quality as ceq
import export_minimal as em
import generate_link_graph as glg
import pytest
from utils.paths import path_to_id

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")


def _sample_reference_md(title: str = "Sample Reference Index") -> str:
    return f"""---
type: reference_page
status: complete
topic: systems
summary: "Sample reference index for parity testing."
sources:
  - https://example.com/ref
updated: 2026-11-24
---

# {title}

| Section | Focus |
| --- | --- |
| Core | Overview |

```mermaid
flowchart LR
  A[Input] --> B[Reference]
```

### Section One

Details one.

### Section Two

Details two.

### Section Three

Details three.

## 参考来源与收录边界

- Source notes.

## 关联页面

- [Tech Overview](../tech-map/overview.md)
"""


def _sample_tech_map_md(title: str = "Sample Tech Node") -> str:
    return f"""---
type: tech_map_node
status: complete
topic: systems
summary: "Sample tech map node for parity testing."
sources:
  - https://example.com/tech
updated: 2026-11-24
---

# {title}

## 在全栈中的位置

```mermaid
flowchart LR
  A[Input] --> B[Tech Node]
```

## 核心内容与站内入口

- Details.

## 上下游模块

- Upstream and downstream.

## 参考来源与收录边界

- Source notes.
"""


def _build_temp_repo(tmp_path: Path) -> Path:
    _write_md(
        tmp_path / "wiki" / "concepts" / "sample-concept.md",
        """---
type: concept
status: complete
topic: systems
summary: "Sample concept."
sources:
  - sources/papers/sample.md
updated: 2026-11-24
---

# Sample Concept

一句话定义。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| RL | Reinforcement Learning | 强化学习 |
| IL | Imitation Learning | 模仿学习 |
| MPC | Model Predictive Control | 模型预测控制 |

## 核心内容

Linked to [Tech Overview](../../tech-map/overview.md).

## 参考来源

- [Sample](../../sources/papers/sample.md)
""",
    )
    _write_md(
        tmp_path / "sources" / "papers" / "sample.md",
        "# Sample Source\n",
    )
    _write_md(
        tmp_path / "roadmap" / "sample-roadmap.md",
        """---
type: roadmap_page
status: complete
topic: systems
summary: "Sample roadmap."
sources:
  - https://example.com/roadmap
updated: 2026-11-24
---

# Sample Roadmap

- [Sample Concept](../wiki/concepts/sample-concept.md)
""",
    )
    _write_md(
        tmp_path / "tech-map" / "overview.md",
        _sample_tech_map_md("Tech Map Overview"),
    )
    _write_md(
        tmp_path / "references" / "papers" / "isolated-ref.md",
        """---
type: reference_page
status: complete
topic: systems
summary: "Isolated reference page with zero internal links."
sources:
  - https://example.com/isolated
updated: 2026-11-24
---

# Isolated Reference

No internal links here.
""",
    )
    return tmp_path


def _run_parity_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    monkeypatch.setattr(em, "ROOT", tmp_path)
    monkeypatch.setattr(glg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(glg, "SOURCES_DIR", tmp_path / "sources")

    paths = em.collect_paths(tmp_path)
    items = [em.build_item(p) for p in paths]
    site_payload = em.build_site_data(items)
    nodes, edges = glg._build_graph_data()
    return site_payload, nodes, edges


def test_collect_paths_and_graph_stay_in_sync_for_add_isolated_rename_and_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_temp_repo(tmp_path)

    # Initial state includes an isolated reference page with zero internal edges.
    site_payload, nodes, _edges = _run_parity_sync(root, monkeypatch)
    detail_ids = set(site_payload["pages"]["detail_pages"].keys())
    graph_detail_ids = {n["detail_id"] for n in nodes}
    assert detail_ids == graph_detail_ids
    assert "reference-papers-isolated-ref" in graph_detail_ids
    assert "tech-node-overview" in graph_detail_ids

    # Add an isolated tech-map module with 0 links -> must appear in both detail_pages and graph.
    added_tech = root / "tech-map" / "modules" / "system" / "isolated-mod.md"
    _write_md(added_tech, "# Isolated Module\n\nZero edges.\n")
    site_payload, nodes, _ = _run_parity_sync(root, monkeypatch)
    detail_ids = set(site_payload["pages"]["detail_pages"].keys())
    graph_detail_ids = {n["detail_id"] for n in nodes}
    assert detail_ids == graph_detail_ids
    assert "tech-node-system-isolated-mod" in graph_detail_ids
    added_node = next(n for n in nodes if n["detail_id"] == "tech-node-system-isolated-mod")
    assert added_node["type"] == "tech_map_node"

    # Rename the isolated tech-map module -> old ID disappears, new ID appears in both.
    renamed_tech = root / "tech-map" / "modules" / "system" / "renamed-mod.md"
    added_tech.rename(renamed_tech)
    site_payload, nodes, _ = _run_parity_sync(root, monkeypatch)
    detail_ids = set(site_payload["pages"]["detail_pages"].keys())
    graph_detail_ids = {n["detail_id"] for n in nodes}
    assert detail_ids == graph_detail_ids
    assert "tech-node-system-isolated-mod" not in graph_detail_ids
    assert "tech-node-system-renamed-mod" in graph_detail_ids

    # Delete the isolated reference page -> removed cleanly from both.
    (root / "references" / "papers" / "isolated-ref.md").unlink()
    site_payload, nodes, _ = _run_parity_sync(root, monkeypatch)
    detail_ids = set(site_payload["pages"]["detail_pages"].keys())
    graph_detail_ids = {n["detail_id"] for n in nodes}
    assert detail_ids == graph_detail_ids
    assert "reference-papers-isolated-ref" not in graph_detail_ids


def _write_exports_fixture(
    exports_dir: Path,
    detail_pages: dict[str, dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_count: int | None = None,
) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    (exports_dir / "site-catalog-v1.json").write_text(
        json.dumps({"pages": {"detail_pages": detail_pages}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (exports_dir / "link-graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False),
        encoding="utf-8",
    )
    stats_count = len(nodes) if node_count is None else node_count
    (exports_dir / "graph-stats.json").write_text(
        json.dumps({"node_count": stats_count, "edge_count": len(edges)}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_validate_detail_graph_parity_accepts_consistent_exports(tmp_path: Path) -> None:
    exports_dir = tmp_path / "exports"
    _write_exports_fixture(
        exports_dir,
        detail_pages={
            "reference-papers-a": {"id": "reference-papers-a", "path": "references/papers/a.md"},
            "tech-node-overview": {"id": "tech-node-overview", "path": "tech-map/overview.md"},
        },
        nodes=[
            {
                "id": "references/papers/a.md",
                "detail_id": "reference-papers-a",
                "type": "reference_page",
            },
            {
                "id": "tech-map/overview.md",
                "detail_id": "tech-node-overview",
                "type": "tech_map_node",
            },
        ],
        edges=[[0, 1]],
    )
    detail_count, edge_count = ceq.validate_detail_graph_parity(exports_dir)
    assert (detail_count, edge_count) == (2, 1)


def test_validate_detail_graph_parity_rejects_missing_extra_duplicate_and_bad_edges(
    tmp_path: Path,
) -> None:
    exports_dir = tmp_path / "exports"

    # Missing in graph
    _write_exports_fixture(
        exports_dir,
        detail_pages={
            "reference-papers-a": {"id": "reference-papers-a", "path": "references/papers/a.md"},
            "tech-node-overview": {"id": "tech-node-overview", "path": "tech-map/overview.md"},
        },
        nodes=[{"id": "references/papers/a.md", "detail_id": "reference-papers-a"}],
        edges=[],
    )
    with pytest.raises(ValueError, match="未入图"):
        ceq.validate_detail_graph_parity(exports_dir)

    # Extra in graph
    _write_exports_fixture(
        exports_dir,
        detail_pages={
            "reference-papers-a": {"id": "reference-papers-a", "path": "references/papers/a.md"}
        },
        nodes=[
            {"id": "references/papers/a.md", "detail_id": "reference-papers-a"},
            {"id": "tech-map/overview.md", "detail_id": "tech-node-overview"},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="孤立图谱节点"):
        ceq.validate_detail_graph_parity(exports_dir)

    # Duplicate detail_id
    _write_exports_fixture(
        exports_dir,
        detail_pages={
            "reference-papers-a": {"id": "reference-papers-a", "path": "references/papers/a.md"}
        },
        nodes=[
            {"id": "references/papers/a.md", "detail_id": "reference-papers-a"},
            {"id": "references/papers/b.md", "detail_id": "reference-papers-a"},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="重复 detail_id"):
        ceq.validate_detail_graph_parity(exports_dir)

    # Out-of-bounds edge index
    _write_exports_fixture(
        exports_dir,
        detail_pages={
            "reference-papers-a": {"id": "reference-papers-a", "path": "references/papers/a.md"}
        },
        nodes=[{"id": "references/papers/a.md", "detail_id": "reference-papers-a"}],
        edges=[[0, 99]],
    )
    with pytest.raises(ValueError, match="下标越界"):
        ceq.validate_detail_graph_parity(exports_dir)

    # graph-stats.json node_count mismatch
    _write_exports_fixture(
        exports_dir,
        detail_pages={
            "reference-papers-a": {"id": "reference-papers-a", "path": "references/papers/a.md"}
        },
        nodes=[{"id": "references/papers/a.md", "detail_id": "reference-papers-a"}],
        edges=[],
        node_count=99,
    )
    with pytest.raises(ValueError, match="node_count"):
        ceq.validate_detail_graph_parity(exports_dir)


@pytest.mark.full_only
def test_repo_exports_detail_graph_parity_and_33_governed_nodes() -> None:
    exports_dir = REPO_ROOT / "exports"
    site_catalog_path = exports_dir / "site-catalog-v1.json"
    link_graph_path = exports_dir / "link-graph.json"
    graph_stats_path = exports_dir / "graph-stats.json"
    if not (site_catalog_path.exists() and link_graph_path.exists() and graph_stats_path.exists()):
        pytest.skip("Derived exports not generated; run `make ci-preflight` first.")

    detail_count, edge_count = ceq.validate_detail_graph_parity(exports_dir)
    assert detail_count > 0 and edge_count > 0

    expected_paths = em.collect_paths(REPO_ROOT)
    expected_ids = {path_to_id(p, REPO_ROOT) for p in expected_paths}
    assert len(expected_ids) == len(expected_paths)

    site_data = json.loads(site_catalog_path.read_text(encoding="utf-8"))
    detail_pages = site_data["pages"]["detail_pages"]
    assert set(detail_pages.keys()) == expected_ids

    link_graph = json.loads(link_graph_path.read_text(encoding="utf-8"))
    nodes_by_detail_id = {n["detail_id"]: n for n in link_graph["nodes"]}
    assert set(nodes_by_detail_id.keys()) == expected_ids

    graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
    assert graph_stats.get("orphan_nodes", []) == []

    ref_paths = sorted(
        p for p in expected_paths if p.relative_to(REPO_ROOT).parts[0] == "references"
    )
    tech_paths = sorted(
        p for p in expected_paths if p.relative_to(REPO_ROOT).parts[0] == "tech-map"
    )
    assert len(ref_paths) == 16
    assert len(tech_paths) == 17
    assert len(ref_paths) + len(tech_paths) == 33

    for p in ref_paths:
        did = path_to_id(p, REPO_ROOT)
        node = nodes_by_detail_id[did]
        assert node["type"] == "reference_page", p
        assert node["health_score"] == 3, p
        assert node["health_gaps"] == 0, p
        assert node.get("community"), p
        assert detail_pages[did]["health_score"] == 3, p

    for p in tech_paths:
        did = path_to_id(p, REPO_ROOT)
        node = nodes_by_detail_id[did]
        assert node["type"] == "tech_map_node", p
        assert node["health_score"] == 3, p
        assert node["health_gaps"] == 0, p
        assert node.get("community"), p
        assert detail_pages[did]["health_score"] == 3, p
