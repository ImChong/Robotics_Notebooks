"""log.md 写入应与「新记录在上」约定一致；PR 写 log.d/ 碎片，main 上并入 log.md。"""

from datetime import date
from pathlib import Path

from log_md import (
    fold_log_fragments,
    prepend_log_entry,
    rotate_old_log_entries,
    write_log_fragment,
    write_log_prepend,
)


def test_prepend_log_entry_inserts_before_first_section() -> None:
    text = "> preamble\n\n## [2026-01-01] ingest | old\n\nbody\n"
    entry = "## [2026-02-02] lint | new\n\n"
    out = prepend_log_entry(text, entry)
    assert out.index("## [2026-02-02]") < out.index("## [2026-01-01]")
    assert out.startswith("> preamble\n\n## [2026-02-02]")


def test_prepend_log_entry_empty_log_gets_entry_at_end() -> None:
    text = "> only preamble\n\n"
    entry = "## [2026-03-03] structural | first\n\n"
    out = prepend_log_entry(text, entry)
    assert "## [2026-03-03]" in out
    assert "> only preamble" in out


def test_write_log_prepend(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    log.write_text("> preamble\n\n## [2026-01-01] ingest | old\n\n", encoding="utf-8")
    write_log_prepend("## [2026-06-06] lint | health-check | test\n\n", log)
    text = log.read_text(encoding="utf-8")
    assert text.index("## [2026-06-06]") < text.index("## [2026-01-01]")


def test_write_log_fragment_unique_names(tmp_path: Path) -> None:
    a = write_log_fragment("## [2026-06-06] ingest | a\n", "ingest", tmp_path, today="2026-06-06")
    b = write_log_fragment("## [2026-06-06] ingest | b\n", "ingest", tmp_path, today="2026-06-06")
    assert a != b
    assert a.name.startswith("2026-06-06-") and a.name.endswith(".md")
    assert a.read_text(encoding="utf-8") == "## [2026-06-06] ingest | a\n"


def test_fold_log_fragments_newest_on_top_and_removed(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    log.write_text("## [2026-01-01] ingest | old\n\nbody\n", encoding="utf-8")
    frag_dir = tmp_path / "log.d"
    frag_dir.mkdir()
    (frag_dir / "README.md").write_text("说明", encoding="utf-8")
    (frag_dir / "2026-02-01-x.md").write_text("## [2026-02-01] ingest | feb\n", encoding="utf-8")
    (frag_dir / "2026-03-01-y.md").write_text("## [2026-03-01] query | mar\n", encoding="utf-8")

    folded = fold_log_fragments(log, frag_dir)

    assert [p.name for p in folded] == ["2026-02-01-x.md", "2026-03-01-y.md"]
    text = log.read_text(encoding="utf-8")
    assert (
        text.index("## [2026-03-01]")
        < text.index("## [2026-02-01]")
        < text.index("## [2026-01-01]")
    )
    assert sorted(p.name for p in frag_dir.iterdir()) == ["README.md"]
    assert fold_log_fragments(log, frag_dir) == []


def test_rotate_old_log_entries_groups_by_quarter_and_keeps_recent(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    archive_dir = tmp_path / "docs" / "plans" / "archive"
    log.write_text(
        "> 顶部说明\n\n"
        "## [2026-11-10] ingest | new\n\n- newest\n\n"
        "## [2026-10-05] structural | recent\n\n- oct\n\n"
        "## [2026-09-20] ingest | q3\n\n- sep\n\n"
        "## [2026-06-15] ingest | q2\n\n- jun\n",
        encoding="utf-8",
    )

    written = rotate_old_log_entries(
        log_path=log,
        archive_dir=archive_dir,
        keep_recent_months=2,
        reference_date=date(2026, 11, 17),
    )

    assert sorted(p.name for p in written) == ["log-2026-Q2.md", "log-2026-Q3.md"]
    remaining = log.read_text(encoding="utf-8")
    assert "> 顶部说明" in remaining
    assert "## [2026-11-10] ingest | new" in remaining
    assert "## [2026-10-05] structural | recent" in remaining
    assert "2026-09-20" not in remaining
    assert "2026-06-15" not in remaining
    assert "## [2026-09-20] ingest | q3" in (archive_dir / "log-2026-Q3.md").read_text(
        encoding="utf-8"
    )
    assert "## [2026-06-15] ingest | q2" in (archive_dir / "log-2026-Q2.md").read_text(
        encoding="utf-8"
    )

    # 幂等性：再次执行不会重复归档
    assert (
        rotate_old_log_entries(
            log_path=log,
            archive_dir=archive_dir,
            keep_recent_months=2,
            reference_date=date(2026, 11, 17),
        )
        == []
    )


def test_rotate_old_log_entries_prepends_to_existing_archive(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    existing_q3 = archive_dir / "log-2026-Q3.md"
    existing_q3.write_text(
        "# 知识库维护日志归档（2026 Q3）\n\n## [2026-08-01] ingest | earlier\n\n- aug\n",
        encoding="utf-8",
    )
    log.write_text(
        "> 顶部说明\n\n"
        "## [2026-11-01] ingest | now\n\n- nov\n\n"
        "## [2026-09-12] ingest | sep\n\n- sep\n",
        encoding="utf-8",
    )

    rotate_old_log_entries(
        log_path=log,
        archive_dir=archive_dir,
        keep_recent_months=2,
        reference_date=date(2026, 11, 17),
    )

    q3_text = existing_q3.read_text(encoding="utf-8")
    assert q3_text.count("# 知识库维护日志归档（2026 Q3）") == 1
    assert q3_text.index("2026-09-12") < q3_text.index("2026-08-01")
