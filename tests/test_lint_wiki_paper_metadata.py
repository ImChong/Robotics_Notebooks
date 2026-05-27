"""Tests for V23 lint check: paper-* 实体页元数据基线（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    return wiki


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_paper_entity_metadata(pages, results)
    return results


def test_complete_paper_passes_both_checks(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "paper-foo.md"
    page.write_text(
        "---\n"
        "type: entity\n"
        "arxiv: 2401.00000\n"
        "---\n"
        "\n"
        "## 方法栈\n正文\n\n## 评测\n正文\n\n## 与其他工作对比\n正文\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["paper_missing_source_meta"] == []
    assert results["paper_missing_three_sections"] == []


def test_venue_or_code_satisfies_source_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    for key in ("venue", "code"):
        page = wiki / "entities" / f"paper-{key}.md"
        page.write_text(
            f"---\ntype: entity\n{key}: https://example.org\n---\n"
            "## 方法\n## 实验\n## 对比\n",
            encoding="utf-8",
        )
        results = _run([page])
        assert results["paper_missing_source_meta"] == [], f"{key} should satisfy"


def test_missing_source_key_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "paper-bar.md"
    page.write_text(
        "---\ntype: entity\nsources:\n  - ../../sources/papers/bar.md\n---\n"
        "## 方法栈\n## 评测\n## 对比\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["paper_missing_source_meta"] == ["wiki/entities/paper-bar.md"]
    assert results["paper_missing_three_sections"] == []


def test_partial_three_sections_records_missing(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "paper-baz.md"
    page.write_text(
        "---\narxiv: 2402.99999\n---\n## 方法栈\n仅有方法段\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["paper_missing_source_meta"] == []
    assert len(results["paper_missing_three_sections"]) == 1
    record = results["paper_missing_three_sections"][0]
    assert "评测" in record and "对比" in record and "方法" not in record


def test_non_paper_entity_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "unitree-g1.md"
    page.write_text("---\ntype: entity\n---\n# 普通实体\n", encoding="utf-8")
    results = _run([page])
    assert results["paper_missing_source_meta"] == []
    assert results["paper_missing_three_sections"] == []


def test_info_only_keys_do_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["paper_missing_source_meta"].append("wiki/entities/paper-x.md")
    results["paper_missing_three_sections"].append("wiki/entities/paper-x.md（缺 评测）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 2
