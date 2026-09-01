"""Sanity checks for graph export sync constants and the git-driven guard."""

from __future__ import annotations

import json
from pathlib import Path

import graph_exports_sync as ges
import pytest
from graph_exports_sync import GRAPH_EXPORT_FILES, repo_root


def test_graph_export_filenames_include_core_graph_json() -> None:
    assert "link-graph.json" in GRAPH_EXPORT_FILES
    assert "home-stats.json" in GRAPH_EXPORT_FILES
    assert "hub-rankings.json" in GRAPH_EXPORT_FILES


def test_repo_root_points_at_workspace() -> None:
    root = repo_root()
    assert (root / "scripts" / "graph_exports_sync.py").exists()
    assert (root / "wiki").is_dir()


def _write_activity(tmp_path: Path, payload: object) -> Path:
    exports = tmp_path / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    (exports / ges.ACTIVITY_FILE).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_git_driven_activity_passes(tmp_path) -> None:
    root = _write_activity(tmp_path, {"source": "git", "days": []})
    ges.assert_git_driven_activity(root)  # 不抛错即通过


def test_log_md_fallback_is_refused(tmp_path) -> None:
    # 浅克隆回退产物会丢失仅存在于 git 历史的条目，不得同步进 docs/exports/。
    root = _write_activity(tmp_path, {"source": "log.md", "days": []})
    with pytest.raises(SystemExit) as excinfo:
        ges.assert_git_driven_activity(root)
    message = str(excinfo.value)
    assert "log.md" in message
    assert "git fetch --unshallow" in message


def test_missing_activity_file_does_not_block(tmp_path) -> None:
    # 本轮没生成图谱产物（分步流程）时无从判断降级与否，不阻塞同步。
    ges.assert_git_driven_activity(tmp_path)


def test_unparsable_activity_file_does_not_block(tmp_path) -> None:
    root = tmp_path
    (root / "exports").mkdir(parents=True)
    (root / "exports" / ges.ACTIVITY_FILE).write_text("not json", encoding="utf-8")
    ges.assert_git_driven_activity(root)


def test_copy_refuses_degraded_activity(tmp_path, monkeypatch) -> None:
    # 守卫接在 copy 入口上：拒绝时不应写出任何 docs/exports/ 文件。
    root = _write_activity(tmp_path, {"source": "log.md", "days": []})
    (root / "exports" / "home-stats.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ges, "repo_root", lambda: root)
    with pytest.raises(SystemExit):
        ges.copy_graph_exports_to_docs()
    assert not (root / "docs" / "exports" / "home-stats.json").exists()


def test_copy_syncs_when_git_driven(tmp_path, monkeypatch) -> None:
    root = _write_activity(tmp_path, {"source": "git", "days": []})
    (root / "exports" / "home-stats.json").write_text('{"node_count":1}', encoding="utf-8")
    monkeypatch.setattr(ges, "repo_root", lambda: root)
    ges.copy_graph_exports_to_docs()
    assert (root / "docs" / "exports" / "home-stats.json").read_text() == '{"node_count":1}'
