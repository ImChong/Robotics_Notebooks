"""Tests for V22 lint check: methods_without_practitioner_query."""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "methods").mkdir(parents=True)
    (wiki / "concepts").mkdir()
    (wiki / "queries").mkdir()
    (wiki / "comparisons").mkdir()
    return wiki


def _run(pages: list[Path]) -> dict:
    page_set = {p.resolve() for p in pages}
    inbound, _ = lw._build_link_index(pages, page_set)
    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    return results


def test_warns_when_high_inbound_method_has_no_practitioner(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    method = wiki / "methods" / "foo.md"
    method.write_text("# foo\n", encoding="utf-8")
    refs = []
    for i in range(4):
        p = wiki / "concepts" / f"c{i}.md"
        p.write_text("see [foo](../methods/foo.md)\n", encoding="utf-8")
        refs.append(p)
    pages = [method, *refs]

    results = _run(pages)
    hits = results["methods_without_practitioner_query"]
    assert len(hits) == 1
    assert "wiki/methods/foo.md" in hits[0].replace("\\", "/")
    assert "被 4 个页面引用" in hits[0]


def test_silent_when_query_page_links_method(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    method = wiki / "methods" / "foo.md"
    method.write_text("# foo\n", encoding="utf-8")
    refs = [
        wiki / "concepts" / "c0.md",
        wiki / "concepts" / "c1.md",
        wiki / "concepts" / "c2.md",
        wiki / "queries" / "q.md",
    ]
    for p in refs:
        p.write_text("see [foo](../methods/foo.md)\n", encoding="utf-8")
    pages = [method, *refs]

    results = _run(pages)
    assert results["methods_without_practitioner_query"] == []


def test_silent_when_comparison_page_links_method(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    method = wiki / "methods" / "foo.md"
    method.write_text("# foo\n", encoding="utf-8")
    refs = [
        wiki / "concepts" / "c0.md",
        wiki / "concepts" / "c1.md",
        wiki / "concepts" / "c2.md",
        wiki / "comparisons" / "cmp.md",
    ]
    for p in refs:
        p.write_text("see [foo](../methods/foo.md)\n", encoding="utf-8")
    pages = [method, *refs]

    results = _run(pages)
    assert results["methods_without_practitioner_query"] == []


def test_silent_when_inbound_at_or_below_threshold(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    method = wiki / "methods" / "foo.md"
    method.write_text("# foo\n", encoding="utf-8")
    refs = []
    for i in range(3):
        p = wiki / "concepts" / f"c{i}.md"
        p.write_text("see [foo](../methods/foo.md)\n", encoding="utf-8")
        refs.append(p)
    pages = [method, *refs]

    results = _run(pages)
    assert results["methods_without_practitioner_query"] == []


def test_self_links_are_excluded_from_inbound_count(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    method = wiki / "methods" / "foo.md"
    method.write_text("self [foo](foo.md)\n", encoding="utf-8")
    refs = []
    for i in range(3):
        p = wiki / "concepts" / f"c{i}.md"
        p.write_text("see [foo](../methods/foo.md)\n", encoding="utf-8")
        refs.append(p)
    pages = [method, *refs]

    results = _run(pages)
    assert results["methods_without_practitioner_query"] == []


def test_info_only_keys_do_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["methods_without_practitioner_query"].append("wiki/methods/foo.md（被 5 个页面引用）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
