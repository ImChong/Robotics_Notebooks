"""append_log.py 应与 log.md 新记录在上的约定一致。"""

from pathlib import Path

from scripts.append_log import prepend_log_entry


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
