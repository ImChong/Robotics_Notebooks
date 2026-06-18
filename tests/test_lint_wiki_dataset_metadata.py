"""Tests for V25 lint check: dataset 实体页元数据巡检（信息型）。"""

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
    lw._check_dataset_entity_metadata(pages, results)
    return results


def test_complete_dataset_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "foo-dataset.md"
    page.write_text(
        "---\n"
        "type: entity\n"
        "tags:\n"
        "  - dataset\n"
        "  - mocap\n"
        "---\n"
        "规模约 40 小时动捕序列；模态为 SMPL 参数；采用 CC-BY 许可证；"
        "需经重定向适配目标机器人形态后作为训练输入。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["dataset_missing_metadata"] == []


def test_inline_tags_are_detected(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "bar-dataset.md"
    page.write_text(
        "---\ntype: entity\ntags: [dataset, human-motion]\n---\n正文仅描述规模 1000 条。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert len(results["dataset_missing_metadata"]) == 1
    record = results["dataset_missing_metadata"][0]
    assert "模态" in record and "许可证" in record and "重定向就绪度" in record
    assert "规模" not in record


def test_non_dataset_entity_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "unitree-g1.md"
    page.write_text(
        "---\ntype: entity\ntags: [hardware, humanoid]\n---\n# 普通实体\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["dataset_missing_metadata"] == []


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["dataset_missing_metadata"].append("wiki/entities/x-dataset.md（缺 许可证）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
