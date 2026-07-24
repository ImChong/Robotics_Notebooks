"""Tests for V30 lint check: 执行器驱动链页交叉链路巡检（信息型）。"""

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
    lw._check_actuator_drive_chain_crosslink(pages, results)
    return results


def test_entity_with_query_hub_backlink_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "simplefoc.md"
    page.write_text(
        "---\ntype: entity\ntags: [foc, driver, open-source]\n---\n"
        "SimpleFOC 处于驱动固件层，见 "
        "[端到端 Query](../queries/actuator-drive-chain-selection-loop.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == []


def test_inline_tag_entity_without_hub_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "kicad.md"
    page.write_text(
        "---\ntype: entity\ntags: [eda, pcb, open-source]\n---\n"
        "KiCad 正文，未回链任何驱动链枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == ["wiki/entities/kicad.md"]


def test_list_style_derived_tag_entity_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "bam-better-actuator-models.md"
    page.write_text(
        "---\ntype: entity\ntags:\n  - modeling\n  - actuator-network\n---\n执行器建模正文。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == [
        "wiki/entities/bam-better-actuator-models.md"
    ]


def test_comparison_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "comparisons" / "kicad-vs-altium.md"
    page.write_text(
        "---\ntype: comparison\ntags: [eda, pcb]\n---\nEDA 工具对比正文，未回链专题枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == ["wiki/comparisons/kicad-vs-altium.md"]


def test_concept_page_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "torque-source-abstraction-gap.md"
    page.write_text(
        "---\ntype: concept\ntags: [actuator, torque-source]\n---\n"
        "理想力矩源抽象概念正文，未回链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == [
        "wiki/concepts/torque-source-abstraction-gap.md"
    ]


def test_topic_hub_backlink_also_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "altium-designer.md"
    page.write_text(
        "---\ntype: entity\ntags: [eda, pcb]\n---\n"
        "Altium Designer，见专题 "
        "[执行器驱动链](../overview/topic-actuator-drive-chain.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == []


def test_untagged_entity_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "entities" / "diffusion-policy.md"
    page.write_text(
        "---\ntype: entity\ntags: [imitation-learning, generative]\n---\n与驱动链无关。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == []


def test_both_hubs_present_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "ethercat-protocol.md"
    page.write_text(
        "---\ntype: concept\ntags: [actuator, realtime-bus, foc]\n---\n"
        "EtherCAT 实时总线层，见 "
        "[Query](../queries/actuator-drive-chain-selection-loop.md) 与 "
        "[专题](../overview/topic-actuator-drive-chain.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == []


def test_hub_pages_exempt_from_self_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "actuator-drive-chain-selection-loop.md"
    page.write_text(
        "---\ntype: concept\ntags: [actuator, foc]\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["actuator_drive_chain_crosslink"] == []


def test_substring_lookalike_tags_are_not_flagged(tmp_path: Path, monkeypatch) -> None:
    """裸子串会把 'impedance'/'bipedal'/'pedagogy'/'bytedance' 误判为含 'eda'，
    token 前缀匹配后这些非执行器页不应进入驱动链巡检清单。
    """
    wiki = _setup_wiki(tmp_path, monkeypatch)
    lookalikes = {
        "impedance-control.md": "[impedance-control, force-control]",
        "lip-zmp.md": "[locomotion, bipedal]",
        "humanoid-rubber-man-analogy.md": "[humanoid, pedagogy]",
        "paper-xrobotoolkit.md": "[teleop, bytedance]",
    }
    pages = []
    for name, tags in lookalikes.items():
        sub = "entities" if name.startswith("paper-") else "concepts"
        page = wiki / sub / name
        page.write_text(
            f"---\ntype: concept\ntags: {tags}\n---\n未回链驱动链枢纽。\n", encoding="utf-8"
        )
        pages.append(page)
    results = _run(pages)
    assert results["actuator_drive_chain_crosslink"] == []


def test_derived_and_plural_actuator_tags_still_flagged(tmp_path: Path, monkeypatch) -> None:
    """'actuators'（复数）与 'foc-driver'（派生）仍应被 token 前缀匹配捕获。"""
    wiki = _setup_wiki(tmp_path, monkeypatch)
    p1 = wiki / "concepts" / "field-oriented-control.md"
    p1.write_text("---\ntype: concept\ntags: [actuators, motor]\n---\n未回链。\n", encoding="utf-8")
    p2 = wiki / "entities" / "some-driver.md"
    p2.write_text("---\ntype: entity\ntags: [foc-driver]\n---\n未回链。\n", encoding="utf-8")
    results = _run([p1, p2])
    assert sorted(results["actuator_drive_chain_crosslink"]) == [
        "wiki/concepts/field-oriented-control.md",
        "wiki/entities/some-driver.md",
    ]


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["actuator_drive_chain_crosslink"].append("wiki/entities/kicad.md")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
