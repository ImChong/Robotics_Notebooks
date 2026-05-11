"""Sanity checks for graph export sync constants."""

from __future__ import annotations

from graph_exports_sync import GRAPH_EXPORT_FILES, repo_root


def test_graph_export_filenames_include_core_graph_json() -> None:
    assert "link-graph.json" in GRAPH_EXPORT_FILES
    assert "home-stats.json" in GRAPH_EXPORT_FILES


def test_repo_root_points_at_workspace() -> None:
    root = repo_root()
    assert (root / "scripts" / "graph_exports_sync.py").exists()
    assert (root / "wiki").is_dir()
