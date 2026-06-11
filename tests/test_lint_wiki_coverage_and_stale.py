"""coverage_stats、frontmatter updated 豁免、已跟踪文件的本地改动 mtime。"""

from __future__ import annotations

import subprocess
import time
from datetime import date
from pathlib import Path

import lint_wiki as lw
import pytest


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


def _init_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "sources" / "papers").mkdir(parents=True)
    (tmp_path / "wiki" / "methods").mkdir(parents=True)
    return tmp_path


def _commit_at(repo: Path, paths: list[Path], when: int, msg: str) -> None:
    _git(repo, "add", *[str(p) for p in paths])
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        msg,
        "--date",
        f"@{when} +0000",
    )


def test_coverage_stats_from_results() -> None:
    results = lw._empty_results()
    results["_ingest_covered"] = 3
    results["_ingest_total"] = 4
    stats = lw.coverage_stats(results)
    assert stats == {"covered": 3, "total": 4, "percent": 75}


def test_frontmatter_updated_exempts_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path, monkeypatch)
    wiki = repo / "wiki" / "methods" / "foo.md"
    wiki.write_text(
        "---\ntype: method\nupdated: 2026-06-11\n---\n\n# foo\n",
        encoding="utf-8",
    )
    old = int(time.time()) - 10 * 86400
    _commit_at(repo, [wiki], old, "old wiki")

    src = repo / "sources" / "papers" / "foo.md"
    src.write_text("see [foo](../../wiki/methods/foo.md)\n", encoding="utf-8")
    _commit_at(repo, [src], int(time.time()) - 1 * 86400, "newer source")

    results = lw._empty_results()
    lw._check_sources_health(results)
    assert results["stale_pages"] == []


def test_committed_wiki_local_edit_uses_fresh_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, monkeypatch)
    wiki = repo / "wiki" / "methods" / "foo.md"
    wiki.write_text("# foo\n", encoding="utf-8")
    src = repo / "sources" / "papers" / "foo.md"
    src.write_text("see [foo](../../wiki/methods/foo.md)\n", encoding="utf-8")
    old = int(time.time()) - 10 * 86400
    _commit_at(repo, [wiki, src], old, "both old")

    src.write_text("see [foo](../../wiki/methods/foo.md)\n# newer source body\n", encoding="utf-8")
    _commit_at(repo, [src], int(time.time()) - 1 * 86400, "newer source only")

    wiki.write_text("# foo\nedited locally\n", encoding="utf-8")
    import os as _os

    _os.utime(wiki, (time.time(), time.time()))

    results = lw._empty_results()
    lw._check_sources_health(results)
    assert results["stale_pages"] == []


def test_frontmatter_updated_date_parses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path, monkeypatch)
    wiki = repo / "wiki" / "methods" / "foo.md"
    wiki.write_text("---\nupdated: 2026-06-11\n---\n", encoding="utf-8")
    assert lw._frontmatter_updated_date(wiki) == date(2026, 6, 11)
