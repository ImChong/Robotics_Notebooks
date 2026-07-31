"""Tests for V31 lint check: 机器人视觉感知栈交叉链路巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "comparisons").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "methods").mkdir(parents=True)
    return wiki


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_perception_stack_crosslink(pages, results)
    return results


def test_entity_with_query_hub_backlink_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "ultralytics.md"
    page.write_text(
        "---\ntype: entity\ntags: [tooling, object-detection, perception]\n---\n"
        "Ultralytics 处于 2D 检测层，见 "
        "[端到端 Query](../queries/robot-perception-stack-selection-loop.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == []


def test_inline_tag_entity_without_hub_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "rf-detr.md"
    page.write_text(
        "---\ntype: entity\ntags: [object-detection, instance-segmentation, real-time]\n---\n"
        "RF-DETR 正文，未回链任何感知栈枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == ["wiki/entities/rf-detr.md"]


def test_list_style_derived_tag_entity_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "findanything.md"
    page.write_text(
        "---\ntype: entity\ntags:\n  - project\n  - semantic-mapping\n  - open-vocabulary\n---\n"
        "开放词汇 3D 语义建图正文。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == ["wiki/entities/findanything.md"]


def test_comparison_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "comparisons" / "yolo-vs-detr.md"
    page.write_text(
        "---\ntype: comparison\ntags: [object-detection, real-time]\n---\n"
        "单阶段 vs DETR 对比正文，未回链纵深枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == ["wiki/comparisons/yolo-vs-detr.md"]


def test_concept_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "2d-to-3d-semantic-lifting-gap.md"
    page.write_text(
        "---\ntype: concept\ntags: [perception, semantic-mapping]\n---\n"
        "2D→3D 语义提升 gap 概念正文，未回链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == [
        "wiki/concepts/2d-to-3d-semantic-lifting-gap.md"
    ]


def test_method_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "methods" / "object-detection.md"
    page.write_text(
        "---\ntype: method\ntags: [object-detection, computer-vision]\n---\n"
        "目标检测方法页正文，未回链纵深枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == ["wiki/methods/object-detection.md"]


def test_topic_hub_backlink_also_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "paper-segment-anything.md"
    page.write_text(
        "---\ntype: entity\ntags: [foundation-model, segmentation]\n---\n"
        "Segment Anything，见纵深 "
        "[机器人感知栈](../overview/hub-perception-stack.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == []


def test_untagged_entity_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "diffusion-policy.md"
    page.write_text(
        "---\ntype: entity\ntags: [imitation-learning, generative]\n---\n与感知栈无关。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == []


def test_both_hubs_present_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "paper-sam2.md"
    page.write_text(
        "---\ntype: entity\ntags: [segmentation, video-segmentation]\n---\n"
        "SAM2 可提示分割层，见 "
        "[Query](../queries/robot-perception-stack-selection-loop.md) 与 "
        "[纵深](../overview/hub-perception-stack.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == []


def test_hub_pages_exempt_from_self_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "robot-perception-stack-selection-loop.md"
    page.write_text(
        "---\ntype: concept\ntags: [perception, detection]\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["perception_stack_crosslink"] == []


def test_substring_lookalike_tags_are_not_flagged(tmp_path: Path, monkeypatch) -> None:
    """裸子串会把无关标签误判为含感知栈关键词，token 前缀匹配后这些页不应进入清单。"""
    wiki = _setup_wiki(tmp_path, monkeypatch)
    lookalikes = {
        "imitation-learning.md": "[imitation-learning, behavior-cloning]",
        "reception-desk.md": "[reception, ui]",
    }
    pages = []
    for name, tags in lookalikes.items():
        page = wiki / "concepts" / name
        page.write_text(
            f"---\ntype: concept\ntags: {tags}\n---\n未回链感知栈枢纽。\n", encoding="utf-8"
        )
        pages.append(page)
    results = _run(pages)
    assert results["perception_stack_crosslink"] == []


def test_derived_and_plural_perception_tags_still_flagged(tmp_path: Path, monkeypatch) -> None:
    """'promptable-segmentation'（派生）与 'detections'（复数）仍应被 token 前缀捕获。"""
    wiki = _setup_wiki(tmp_path, monkeypatch)
    p1 = wiki / "entities" / "sam.md"
    p1.write_text(
        "---\ntype: entity\ntags: [promptable-segmentation, sam]\n---\n未回链。\n",
        encoding="utf-8",
    )
    p2 = wiki / "entities" / "some-detector.md"
    p2.write_text("---\ntype: entity\ntags: [detections]\n---\n未回链。\n", encoding="utf-8")
    results = _run([p1, p2])
    assert sorted(results["perception_stack_crosslink"]) == [
        "wiki/entities/sam.md",
        "wiki/entities/some-detector.md",
    ]


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["perception_stack_crosslink"].append("wiki/entities/rf-detr.md")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
