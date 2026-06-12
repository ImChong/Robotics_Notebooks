"""Tests for V24 lint check: 缺页概念巡检（missing concept page，信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki" / "concepts"
    wiki.mkdir(parents=True)
    return wiki


def _page(wiki: Path, name: str, body: str) -> Path:
    page = wiki / name
    page.write_text(f"---\ntype: concept\n---\n\n{body}\n", encoding="utf-8")
    return page


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_missing_concept_pages(pages, results)
    return results


def test_term_flagged_when_referenced_by_enough_pages(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", f"本页讨论 **GRPO** 优化方法第 {i} 段。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    assert any("GRPO" in rec for rec in results["missing_concept_pages"])


def test_term_not_flagged_below_threshold(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "提及 **GRPO** 一次。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES - 1)
    ]
    results = _run(pages)
    assert results["missing_concept_pages"] == []


def test_term_with_existing_page_not_flagged(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    # 已有独立页 grpo.md → 即便多页引用也不应建议新建
    pages = [_page(wiki, "grpo.md", "GRPO 概念页。")]
    pages += [
        _page(wiki, f"p{i}.md", "引用 **GRPO**。") for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    assert results["missing_concept_pages"] == []


def test_stopwords_and_paths_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "frontmatter 字段 **type** 与路径 `wiki/concepts/x.md`。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # type 是停用词；含 '/' 与 '.md' 的路径不符合单 token 词形 → 均不入候选
    assert results["missing_concept_pages"] == []


def test_case_insensitive_merge(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    half = lw.MISSING_CONCEPT_PAGE_MIN_PAGES // 2 + 1
    pages = [_page(wiki, f"a{i}.md", "讨论 **ViT** 骨干。") for i in range(half)]
    pages += [_page(wiki, f"b{i}.md", "讨论 `vit` 骨干。") for i in range(half)]
    results = _run(pages)
    # 大小写应归并到同一术语计数，达到阈值后被标记一次
    hits = [r for r in results["missing_concept_pages"] if "vit" in r.lower()]
    assert len(hits) == 1


def test_missing_concept_pages_is_info_only(tmp_path, monkeypatch) -> None:
    results = lw._empty_results()
    results["missing_concept_pages"].append("GRPO（被 6 个页面引用...）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
