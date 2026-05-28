"""Tests for the stale-page detection in `_check_sources_health`.

The check used to compare filesystem `mtime`, which produced spurious
"stale" warnings in fresh clones (e.g. cloud Agent containers where every
checked-out file has the same `mtime`). It now prefers git commit time and
only falls back to filesystem `mtime` for files that have no git history
(genuinely uncommitted local edits).
"""

from __future__ import annotations

import subprocess
import time
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
    # Override committer date too so `git log %ct` reflects the intended ts
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "commit",
            "--amend",
            "--no-edit",
            "-q",
            "--date",
            f"@{when} +0000",
        ],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_DATE": f"@{when} +0000",
            "GIT_COMMITTER_DATE": f"@{when} +0000",
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


def test_fresh_clone_mtime_is_ignored_when_git_says_same_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source and wiki committed on the same day → no stale warning, even
    if their filesystem mtimes diverge wildly (simulating a fresh clone)."""
    repo = _init_repo(tmp_path, monkeypatch)
    wiki = repo / "wiki" / "methods" / "foo.md"
    wiki.write_text("# foo\nsome content\n", encoding="utf-8")
    src = repo / "sources" / "papers" / "foo.md"
    src.write_text("see [foo](../../wiki/methods/foo.md)\n", encoding="utf-8")

    same_day = int(time.time()) - 7 * 86400
    _commit_at(repo, [wiki, src], same_day, "ingest foo")

    # Simulate fresh-clone scenario: source mtime is *much* newer than wiki
    # mtime on disk. Pre-fix this would produce a false stale warning.
    now = time.time()
    import os as _os

    _os.utime(src, (now, now))
    _os.utime(wiki, (now - 3 * 86400, now - 3 * 86400))

    results = lw._empty_results()
    lw._check_sources_health(results)
    assert results["stale_pages"] == []


def test_genuine_stale_when_source_committed_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source committed > 1 day after wiki → real stale warning fires."""
    repo = _init_repo(tmp_path, monkeypatch)
    wiki = repo / "wiki" / "methods" / "foo.md"
    wiki.write_text("# foo\n", encoding="utf-8")
    _commit_at(repo, [wiki], int(time.time()) - 10 * 86400, "wiki")

    src = repo / "sources" / "papers" / "foo.md"
    src.write_text("see [foo](../../wiki/methods/foo.md)\n", encoding="utf-8")
    _commit_at(repo, [src], int(time.time()) - 1 * 86400, "newer source")

    results = lw._empty_results()
    lw._check_sources_health(results)
    assert len(results["stale_pages"]) == 1
    assert "wiki/methods/foo.md" in results["stale_pages"][0].replace("\\", "/")


def test_uncommitted_wiki_edit_falls_back_to_fs_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If wiki has uncommitted local edits, we trust filesystem mtime so
    the user's fresh edit isn't reported as stale."""
    repo = _init_repo(tmp_path, monkeypatch)
    wiki = repo / "wiki" / "methods" / "foo.md"
    wiki.write_text("# foo\n", encoding="utf-8")
    src = repo / "sources" / "papers" / "foo.md"
    src.write_text("see [foo](../../wiki/methods/foo.md)\n", encoding="utf-8")
    _commit_at(repo, [src], int(time.time()) - 1 * 86400, "src")

    # Wiki is never committed; its fs mtime is `now`, so should not be stale.
    import os as _os

    _os.utime(wiki, (time.time(), time.time()))

    results = lw._empty_results()
    lw._check_sources_health(results)
    assert results["stale_pages"] == []


def test_non_git_environment_falls_back_to_fs_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If git isn't available / repo isn't initialised, behaviour falls back
    to filesystem mtime (pre-fix behaviour) so this isn't a hard regression."""
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    (tmp_path / "sources" / "papers").mkdir(parents=True)
    (tmp_path / "wiki" / "methods").mkdir(parents=True)

    wiki = tmp_path / "wiki" / "methods" / "foo.md"
    wiki.write_text("# foo\n", encoding="utf-8")
    src = tmp_path / "sources" / "papers" / "foo.md"
    src.write_text("see [foo](../../wiki/methods/foo.md)\n", encoding="utf-8")

    import os as _os

    # Source 5 days newer than wiki on disk, no git → expect stale warning.
    now = time.time()
    _os.utime(wiki, (now - 10 * 86400, now - 10 * 86400))
    _os.utime(src, (now - 5 * 86400, now - 5 * 86400))

    results = lw._empty_results()
    lw._check_sources_health(results)
    assert len(results["stale_pages"]) == 1
