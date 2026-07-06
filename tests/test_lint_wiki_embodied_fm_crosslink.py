"""Tests for V28 lint check: 具身大模型家族概念/对比页交叉链路巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "comparisons").mkdir(parents=True)
    return wiki


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_embodied_fm_crosslink(pages, results)
    return results


def test_concept_with_hub_backlink_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "embodied-fm-latency-generalization-tradeoff.md"
    page.write_text(
        "---\ntype: concept\ntags: [vla, world-model, control]\n---\n"
        "实时性↔泛化取舍在具身大模型选型闭环中的定位，见 "
        "[端到端 Query](../queries/embodied-fm-taxonomy-loop.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == []


def test_inline_tag_concept_without_hub_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "world-action-models.md"
    page.write_text(
        "---\ntype: concept\ntags: [world-model, vla, generative]\n---\n"
        "世界-动作模型正文，未回链任何具身大模型枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == ["wiki/concepts/world-action-models.md"]


def test_list_style_derived_tag_concept_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "vision-language-navigation.md"
    page.write_text(
        "---\ntype: concept\ntags:\n  - navigation\n  - vln-policy\n---\n视觉语言导航正文。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == ["wiki/concepts/vision-language-navigation.md"]


def test_comparison_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "comparisons" / "vlm-vln-vla-vlx-world-model-taxonomy.md"
    page.write_text(
        "---\ntype: comparison\ntags: [vlm, vln, vla, vlx, world-model]\n---\n"
        "五大具身模型家族对比正文，未回链专题枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == [
        "wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md"
    ]


def test_topic_hub_backlink_also_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "unified-multimodal-tokens.md"
    page.write_text(
        "---\ntype: concept\ntags: [vlx, multimodal]\n---\n"
        "统一多模态 token，见专题 "
        "[具身大模型](../overview/topic-embodied-foundation-model.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == []


def test_untagged_concept_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "diffusion-policy.md"
    page.write_text(
        "---\ntype: concept\ntags: [imitation-learning, generative]\n---\n与具身大模型分类无关。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == []


def test_hub_pages_exempt_from_self_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "embodied-fm-taxonomy-loop.md"
    page.write_text(
        "---\ntype: concept\ntags: [vla, world-model]\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["embodied_fm_crosslink"] == []


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["embodied_fm_crosslink"].append("wiki/concepts/world-action-models.md")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
