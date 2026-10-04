"""Regression coverage for Mermaid placeholders left in ingested Markdown."""

from pathlib import Path

import lint_wiki as lw
import pytest


def test_placeholder_blocks_fail_lint_with_original_line_numbers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    page = tmp_path / "wiki" / "entities" / "paper-example.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# Example\n\n```text\na sample\n```\n\n§§§mermaid\nflowchart TD\nA --> B\n§§§\n",
        encoding="utf-8",
    )
    results = lw._empty_results()
    lw._check_per_page([page], {page.resolve(): [page.resolve()]}, {}, results)
    assert results["mermaid_placeholder_fences"] == [
        "wiki/entities/paper-example.md:7: Mermaid 占位围栏须替换为 ```mermaid"
    ]
    only_placeholder = lw._empty_results()
    only_placeholder["mermaid_placeholder_fences"] = results["mermaid_placeholder_fences"]
    assert lw._failing_total(only_placeholder) == 1
    assert "图表不会渲染" in lw.format_report(results)


@pytest.mark.parametrize("fence", ["```", "````", "~~~"])
def test_placeholder_examples_inside_real_fences_are_ignored(fence):
    content = f"{fence}text\n§§§mermaid\nflowchart TD\nA --> B\n§§§\n{fence}\n"
    assert lw.find_mermaid_placeholder_lines(content) == []


def test_standard_diagram_and_inline_placeholder_explanation_are_valid():
    content = '```mermaid\nflowchart TD\nA["§§§mermaid"] --> B\n```\n见 `§§§mermaid`。\n'
    assert lw.find_mermaid_placeholder_lines(content) == []


def test_shorter_fence_does_not_end_a_nested_code_example():
    content = "````markdown\n```text\n```\n§§§mermaid\n````\n§§§mermaid\n"
    assert lw.find_mermaid_placeholder_lines(content) == [6]


def test_list_indented_code_example_does_not_trigger_lint():
    content = "- 示例：\n\n    ```text\n    §§§mermaid\n    §§§\n    ```\n"
    assert lw.find_mermaid_placeholder_lines(content) == []


def test_invalid_backtick_info_string_cannot_hide_a_placeholder():
    content = "```example`text\n§§§mermaid\nflowchart TD\nA --> B\n§§§\n"
    assert lw.find_mermaid_placeholder_lines(content) == [2]


def test_indented_code_does_not_hide_a_later_placeholder():
    content = "    ```text\n\n§§§mermaid\nflowchart TD\n§§§\n"
    assert lw.find_mermaid_placeholder_lines(content) == [3]


def test_indented_backticks_cannot_close_a_top_level_fence():
    content = "```text\n    ```\n§§§mermaid\n```\n"
    assert lw.find_mermaid_placeholder_lines(content) == []


def test_placeholder_text_in_indented_code_is_ignored():
    assert lw.find_mermaid_placeholder_lines("    §§§mermaid\n    flowchart TD\n") == []
