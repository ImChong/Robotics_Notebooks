"""pr_derived_guard：PR 改机器人专属文件应被拦下；sync 合入 main 时自动消解这些冲突。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pr_derived_guard as guard
import pytest


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """origin（裸仓）+ 工作仓；main 上有 wiki 页、catalog.md、log.md 与一个入库统计文件。"""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "t")
    _git(work, "checkout", "-q", "-b", "main")
    (work / "wiki").mkdir()
    (work / "wiki" / "a.md").write_text("a\n", encoding="utf-8")
    (work / "catalog.md").write_text("catalog v1\n", encoding="utf-8")
    (work / "log.md").write_text("## [2026-01-01] ingest | old\n", encoding="utf-8")
    (work / "exports").mkdir()
    (work / "exports" / "graph-stats.json").write_text('{"n":1}\n', encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "-u", "origin", "main")
    monkeypatch.setattr(guard, "REPO_ROOT", work)
    return work


def _branch_touching_derived(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "wiki" / "b.md").write_text("b\n", encoding="utf-8")
    (repo / "catalog.md").write_text("catalog feature\n", encoding="utf-8")
    (repo / "log.md").write_text(
        "## [2026-02-02] ingest | feature entry\n\n## [2026-01-01] ingest | old\n",
        encoding="utf-8",
    )
    (repo / "exports" / "graph-stats.json").write_text('{"n":2}\n', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature")


def _main_moves_on(repo: Path) -> None:
    """main 上：另一个 PR 合入 + 机器人重写 catalog/log，并删除（gitignore）统计文件。"""
    _git(repo, "checkout", "-q", "main")
    (repo / "wiki" / "c.md").write_text("c\n", encoding="utf-8")
    (repo / "catalog.md").write_text("catalog main\n", encoding="utf-8")
    (repo / "log.md").write_text(
        "## [2026-02-03] ingest | main entry\n\n## [2026-01-01] ingest | old\n",
        encoding="utf-8",
    )
    _git(repo, "rm", "-q", "exports/graph-stats.json")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "feature")


def test_check_flags_bot_owned_changes(repo: Path) -> None:
    _branch_touching_derived(repo)
    assert guard.changed_guarded_paths("origin/main") == [
        "catalog.md",
        "exports/graph-stats.json",
        "log.md",
    ]
    assert guard.cmd_check("origin/main") == 1


def test_check_ignores_deletions_and_source_changes(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "clean")
    (repo / "wiki" / "b.md").write_text("b\n", encoding="utf-8")
    _git(repo, "rm", "-q", "--cached", "exports/graph-stats.json")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "clean")
    assert guard.cmd_check("origin/main") == 0


def test_sync_resolves_derived_conflicts_and_keeps_log_entry(repo: Path) -> None:
    _branch_touching_derived(repo)
    _main_moves_on(repo)

    assert guard.cmd_sync("origin", "main") == 0

    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert (repo / "catalog.md").read_text(encoding="utf-8") == "catalog main\n"
    assert "main entry" in (repo / "log.md").read_text(encoding="utf-8")
    assert "exports/graph-stats.json" not in _git(repo, "ls-files")
    assert (repo / "wiki" / "b.md").exists() and (repo / "wiki" / "c.md").exists()
    fragments = list((repo / "log.d").glob("*.md"))
    assert len(fragments) == 1
    assert "feature entry" in fragments[0].read_text(encoding="utf-8")
    # 同步后 guard 通过
    assert guard.cmd_check("origin/main") == 0


def test_sync_reports_real_source_conflicts(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "wiki" / "a.md").write_text("feature edit\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "feature")
    _git(repo, "checkout", "-q", "main")
    (repo / "wiki" / "a.md").write_text("main edit\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "main")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "feature")

    assert guard.cmd_sync("origin", "main") == 1
    assert guard.unmerged_paths() == ["wiki/a.md"]


def test_sync_refuses_dirty_worktree(repo: Path) -> None:
    (repo / "wiki" / "a.md").write_text("dirty\n", encoding="utf-8")
    assert guard.cmd_sync("origin", "main") == 1


def test_sync_resolves_stat_line_conflicts_keeping_other_edits(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text("title\n\nbadge 1 nodes\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "readme")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "-b", "feature")
    readme.write_text("title edited\n\nbadge 2 nodes\n\nbody\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "feature")
    _git(repo, "checkout", "-q", "main")
    readme.write_text("title\n\nbadge endpoint\n\nbody\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "main")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "feature")

    assert guard.cmd_sync("origin", "main") == 0
    assert readme.read_text(encoding="utf-8") == "title edited\n\nbadge endpoint\n\nbody\n"
