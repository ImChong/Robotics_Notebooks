"""原文通读结构化判据：sources 归档「通读」字段 + wiki frontmatter original。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw
import pytest
from original_reading import original_entries, original_fully_read, reading_status


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("- **通读：** 已完成\n", "已完成"),
        ("- **通读：** 未完成\n", "未完成"),
        ("- **通读：**已完成  \n", "已完成"),
        ("- **核对范围：** 已通读全文。\n", None),  # 无字段：不从自由文本推断
        ("- **通读:** 已完成\n", None),  # 半角冒号不合法
        ("- **通读：** 读完了\n", None),  # 取值不合法
        ("- **通读：** 已完成\n- **通读：** 未完成\n", None),  # 重复字段
    ],
)
def test_reading_status(tmp_path: Path, text: str, expected: str | None) -> None:
    path = tmp_path / "s.md"
    path.write_text(text, encoding="utf-8")
    assert reading_status(path) == expected


def test_reading_status_missing_file(tmp_path: Path) -> None:
    assert reading_status(tmp_path / "missing.md") is None


def test_original_entries_scalar_and_list() -> None:
    assert original_entries("---\noriginal: ../../sources/a.md\n---\n") == ["../../sources/a.md"]
    content = "---\noriginal:\n  - ../../sources/a.md\n  - ../../sources/b.md\n---\n"
    assert original_entries(content) == ["../../sources/a.md", "../../sources/b.md"]
    assert original_entries("---\nsources: [a]\n---\n") == []


def test_original_must_stay_inside_sources(tmp_path: Path) -> None:
    """original 指向 sources/ 之外的文件，即使带「通读：已完成」也不算。"""
    (tmp_path / "sources").mkdir()
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "wiki" / "fake.md").write_text("- **通读：** 已完成\n", encoding="utf-8")
    page = tmp_path / "wiki" / "concepts" / "p.md"
    content = "---\noriginal: ../fake.md\n---\n"
    assert not original_fully_read(content, page, tmp_path / "sources")


def test_lint_flags_invalid_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """lint 阻塞项：original 指向缺失文件、非 sources/ 文件或无合法通读字段的归档。"""
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "ok.md").write_text("- **通读：** 未完成\n", encoding="utf-8")
    (tmp_path / "sources" / "bare.md").write_text("正文\n", encoding="utf-8")
    pages_dir = tmp_path / "wiki" / "concepts"
    pages_dir.mkdir(parents=True)
    page = pages_dir / "p.md"
    page.write_text(
        "---\noriginal:\n  - ../../sources/ok.md\n  - ../../sources/bare.md\n"
        "  - ../../sources/missing.md\n  - ../p.md\n---\n",
        encoding="utf-8",
    )
    results: dict = {"invalid_original": []}
    lw._check_original_reading([page], results)
    flagged = results["invalid_original"]
    assert len(flagged) == 3
    assert not any("ok.md" in item for item in flagged)
