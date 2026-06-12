"""Tests for V24 lint check: 陈旧声明（stale claim）巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki" / "concepts"
    wiki.mkdir(parents=True)
    return wiki


def _page(wiki: Path, name: str, updated: str, tags: list[str], body: str) -> Path:
    tag_block = "".join(f"  - {t}\n" for t in tags)
    page = wiki / name
    page.write_text(
        f"---\ntype: concept\ntags:\n{tag_block}updated: {updated}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return page


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_stale_claims(pages, results)
    return results


def test_stale_claim_flagged_when_newer_same_tag_page_exists(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    old = _page(wiki, "old.md", "2025-01-01", ["vision"], "本方法是当前最强的视觉骨干。")
    newer = _page(wiki, "newer.md", "2026-01-01", ["vision"], "更新的视觉骨干综述。")
    results = _run([old, newer])
    assert len(results["stale_claims"]) == 1
    record = results["stale_claims"][0]
    assert "old.md" in record and "当前最强" in record and "newer.md" in record


def test_no_flag_when_claim_page_is_newest(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    newest = _page(wiki, "a.md", "2026-06-01", ["vision"], "这是最新的 SOTA 结果。")
    older = _page(wiki, "b.md", "2025-01-01", ["vision"], "较早的综述。")
    results = _run([newest, older])
    assert results["stale_claims"] == []


def test_no_flag_without_shared_tag(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(wiki, "a.md", "2025-01-01", ["vision"], "当前最强的视觉骨干。")
    other = _page(wiki, "b.md", "2026-01-01", ["control"], "更晚但不同主题。")
    results = _run([claim, other])
    assert results["stale_claims"] == []


def test_no_flag_without_absolute_phrasing(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    plain = _page(wiki, "a.md", "2025-01-01", ["vision"], "一种常规的视觉骨干，无绝对化措辞。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vision"], "更晚的同主题页。")
    results = _run([plain, newer])
    assert results["stale_claims"] == []


def test_claim_inside_code_block_is_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(wiki, "a.md", "2025-01-01", ["vision"], "```\nSOTA\n```\n普通正文。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vision"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert results["stale_claims"] == []


def test_stale_claims_is_info_only(tmp_path, monkeypatch) -> None:
    results = lw._empty_results()
    results["stale_claims"].append("wiki/concepts/old.md（含绝对化措辞「SOTA」...）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
