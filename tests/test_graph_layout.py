"""Tests for offline graph layout export."""

from __future__ import annotations

from graph_layout import LAYOUT_VERSION, compute_force_layout


def test_compute_force_layout_returns_positions_for_all_nodes() -> None:
    nodes = [
        {"id": "wiki/a.md", "label": "A"},
        {"id": "wiki/b.md", "label": "B"},
        {"id": "wiki/c.md", "label": "C"},
    ]
    edges = [
        {"source": "wiki/a.md", "target": "wiki/b.md"},
        {"source": "wiki/b.md", "target": "wiki/c.md"},
    ]
    degree_map = {"wiki/a.md": 1, "wiki/b.md": 2, "wiki/c.md": 1}
    layout = compute_force_layout(nodes, edges, degree_map, width=800, height=600)

    assert layout["version"] == LAYOUT_VERSION
    assert len(layout["positions"]) == 3
    for node in nodes:
        pos = layout["positions"][node["id"]]
        assert "x" in pos and "y" in pos
        assert 0 <= pos["x"] <= 800
        assert 0 <= pos["y"] <= 600
