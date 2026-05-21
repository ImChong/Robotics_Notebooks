"""Tests for lint_wiki._check_methods_without_practitioner_query (V22)."""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _make_method_with_refs(
    tmp_path: Path,
    method_name: str,
    referrer_dirs: list[str],
) -> tuple[list[Path], dict[Path, list[Path]]]:
    """Build a tiny wiki tree with a single method and N referrers under given dirs."""
    wiki = tmp_path / "wiki"
    method_dir = wiki / "methods"
    method_dir.mkdir(parents=True, exist_ok=True)
    method_page = method_dir / f"{method_name}.md"
    method_page.write_text("# method body\n", encoding="utf-8")

    referrers: list[Path] = []
    for idx, sub in enumerate(referrer_dirs):
        d = wiki / sub
        d.mkdir(parents=True, exist_ok=True)
        ref = d / f"ref-{idx}.md"
        ref.write_text(f"[m](../methods/{method_name}.md)\n", encoding="utf-8")
        referrers.append(ref)

    pages = [method_page, *referrers]
    inbound: dict[Path, list[Path]] = {p.resolve(): [] for p in pages}
    inbound[method_page.resolve()] = [r.resolve() for r in referrers]
    return pages, inbound


def test_method_below_threshold_is_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    pages, inbound = _make_method_with_refs(
        tmp_path, "foo", referrer_dirs=["concepts", "concepts", "concepts"]
    )
    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    assert results["methods_without_practitioner_query"] == []


def test_method_above_threshold_without_query_is_flagged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    pages, inbound = _make_method_with_refs(
        tmp_path,
        "foo",
        referrer_dirs=["concepts", "concepts", "entities", "tasks"],
    )
    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    assert len(results["methods_without_practitioner_query"]) == 1
    entry = results["methods_without_practitioner_query"][0]
    assert "wiki/methods/foo.md" in entry.replace("\\", "/")
    assert "4" in entry


def test_method_with_query_referrer_is_not_flagged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    pages, inbound = _make_method_with_refs(
        tmp_path,
        "foo",
        referrer_dirs=["concepts", "concepts", "entities", "queries"],
    )
    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    assert results["methods_without_practitioner_query"] == []


def test_method_with_comparison_referrer_is_not_flagged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    pages, inbound = _make_method_with_refs(
        tmp_path,
        "foo",
        referrer_dirs=["concepts", "concepts", "tasks", "comparisons"],
    )
    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    assert results["methods_without_practitioner_query"] == []


def test_duplicate_referrer_does_not_inflate_count(tmp_path: Path, monkeypatch) -> None:
    """A single page linking multiple times must count as one referrer."""
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    method_dir = wiki / "methods"
    concept_dir = wiki / "concepts"
    method_dir.mkdir(parents=True)
    concept_dir.mkdir(parents=True)
    method_page = method_dir / "foo.md"
    method_page.write_text("body\n", encoding="utf-8")
    ref = concept_dir / "a.md"
    ref.write_text("[1](../methods/foo.md) [2](../methods/foo.md)\n", encoding="utf-8")

    pages = [method_page, ref]
    # Simulate inbound where the same referrer appears twice (list, not set):
    inbound = {p.resolve(): [] for p in pages}
    inbound[method_page.resolve()] = [ref.resolve(), ref.resolve()]

    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    # 1 distinct referrer ≤ threshold (3) → must not be flagged
    assert results["methods_without_practitioner_query"] == []


def test_method_readme_is_ignored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    method_dir = wiki / "methods"
    concepts_dir = wiki / "concepts"
    method_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)
    readme = method_dir / "README.md"
    readme.write_text("index\n", encoding="utf-8")
    referrers = []
    for i in range(4):
        r = concepts_dir / f"r-{i}.md"
        r.write_text("[x](../methods/README.md)\n", encoding="utf-8")
        referrers.append(r)

    pages = [readme, *referrers]
    inbound = {p.resolve(): [] for p in pages}
    inbound[readme.resolve()] = [r.resolve() for r in referrers]

    results = lw._empty_results()
    lw._check_methods_without_practitioner_query(pages, inbound, results)
    assert results["methods_without_practitioner_query"] == []
