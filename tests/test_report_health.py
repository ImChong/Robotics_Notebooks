"""report_health --check-pr-body：ingest PR 正文的节点健康度汇报须与 CI 实测一致。"""

from __future__ import annotations

from report_health import check_pr_body, format_table

FULL = ("wiki/entities/paper-full.md", 3, [])
PARTIAL = ("wiki/entities/paper-partial.md", 2, ["原文未通读"])


def test_pasted_script_table_passes_when_full_score() -> None:
    assert check_pr_body(format_table([FULL]), [FULL]) == []


def test_missing_page_is_reported() -> None:
    problems = check_pr_body(format_table([FULL]), [FULL, PARTIAL])
    assert len(problems) == 1 and PARTIAL[0] in problems[0] and "缺这一页" in problems[0]


def test_stale_or_guessed_score_is_rejected() -> None:
    stale = "| wiki/entities/paper-full.md | 2/3 | — |"
    guessed = "| `wiki/entities/paper-full.md` | **按规则静态检查预计 3/3** | — |"
    unrun = "| wiki/entities/paper-full.md | 未运行脚本 | — |"
    for body in (stale, guessed, unrun):
        assert len(check_pr_body(body, [FULL])) == 1, body


def test_partial_score_needs_reason_beyond_auto_gap() -> None:
    assert "须写明原因" in check_pr_body(format_table([PARTIAL]), [PARTIAL])[0]
    explained = "| wiki/entities/paper-partial.md | 2/3 | 原文未通读：官方全文付费，无法获取 |"
    assert check_pr_body(explained, [PARTIAL]) == []


def test_backticks_links_and_bold_scores_are_accepted() -> None:
    body = (
        "| [`wiki/entities/paper-full.md`](https://github.com/o/r/blob/b/wiki/entities/paper-full.md)"
        " | **3/3** | — |"
    )
    assert check_pr_body(body, [FULL]) == []
