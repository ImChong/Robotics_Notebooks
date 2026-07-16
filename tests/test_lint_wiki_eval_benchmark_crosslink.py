"""Tests for V29 lint check: 具身大模型评测基准页交叉链路巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "comparisons").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    return wiki


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_eval_benchmark_crosslink(pages, results)
    return results


def test_entity_with_hub_backlink_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "robo-bench.md"
    page.write_text(
        "---\ntype: entity\ntags: [benchmark, mllm, evaluation]\n---\n"
        "RoboBench 处于评测选型闭环①认知层，见 "
        "[端到端 Query](../queries/embodied-eval-benchmark-selection-loop.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == []


def test_inline_tag_entity_without_hub_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "ewmbench.md"
    page.write_text(
        "---\ntype: entity\ntags: [benchmark, world-model, evaluation]\n---\n"
        "EWMBench 正文，未回链任何评测基准枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == ["wiki/entities/ewmbench.md"]


def test_list_style_derived_tag_entity_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "paper-gigaworld-1-policy-evaluation.md"
    page.write_text(
        "---\ntype: entity\ntags:\n  - policy\n  - policy-evaluation\n---\n策略评估器正文。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == [
        "wiki/entities/paper-gigaworld-1-policy-evaluation.md"
    ]


def test_comparison_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "comparisons" / "embodied-benchmark-family-compare.md"
    page.write_text(
        "---\ntype: comparison\ntags: [benchmark, evaluation]\n---\n"
        "评测基准家族对比正文，未回链专题枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == [
        "wiki/comparisons/embodied-benchmark-family-compare.md"
    ]


def test_concept_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "simulation-evaluation-infrastructure.md"
    page.write_text(
        "---\ntype: concept\ntags: [evaluation, simulation]\n---\n评测基建概念正文，未回链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == [
        "wiki/concepts/simulation-evaluation-infrastructure.md"
    ]


def test_topic_hub_backlink_also_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "esi-bench.md"
    page.write_text(
        "---\ntype: entity\ntags: [benchmark, evaluation]\n---\n"
        "ESI-Bench，见专题 "
        "[具身评测基准](../overview/topic-embodied-eval-benchmark.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == []


def test_untagged_entity_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "diffusion-policy.md"
    page.write_text(
        "---\ntype: entity\ntags: [imitation-learning, generative]\n---\n与评测基准无关。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == []


def test_hub_pages_exempt_from_self_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "embodied-eval-benchmark-selection-loop.md"
    page.write_text(
        "---\ntype: concept\ntags: [benchmark, evaluation]\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["eval_benchmark_crosslink"] == []


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["eval_benchmark_crosslink"].append("wiki/entities/ewmbench.md")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
