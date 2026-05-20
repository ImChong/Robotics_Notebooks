"""log.md 写入应与「新记录在上」约定一致（append_log / lint --write-log）。"""

from pathlib import Path

from scripts.log_md import prepend_log_entry, write_log_prepend


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
