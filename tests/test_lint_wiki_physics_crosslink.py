"""Tests for V26 lint check: 动力学/仿真概念页交叉链路巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "formalizations").mkdir(parents=True)
    return wiki


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_physics_concept_crosslink(pages, results)
    return results


def test_concept_with_hub_backlink_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "contact-dynamics.md"
    page.write_text(
        "---\ntype: concept\ntags: [dynamics, contact]\n---\n"
        "接触动力学在物理保真度链路中的定位，见 "
        "[端到端 Query](../queries/simulation-physics-fidelity.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["physics_concept_crosslink"] == []


def test_inline_tag_concept_without_hub_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "centroidal-dynamics.md"
    page.write_text(
        "---\ntype: concept\ntags: [dynamics, locomotion, mpc]\n---\n"
        "质心动力学正文，未回链任何保真度枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["physics_concept_crosslink"] == ["wiki/concepts/centroidal-dynamics.md"]


def test_list_style_tag_formalization_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "formalizations" / "articulated-body-algorithms.md"
    page.write_text(
        "---\ntype: formalization\ntags:\n  - dynamics\n  - aba\n---\n刚体动力学算法正文。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["physics_concept_crosslink"] == [
        "wiki/formalizations/articulated-body-algorithms.md"
    ]


def test_untagged_concept_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "diffusion-policy.md"
    page.write_text(
        "---\ntype: concept\ntags: [imitation-learning, generative]\n---\n与物理保真度无关。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["physics_concept_crosslink"] == []


def test_hub_pages_exempt_from_self_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "physics-fidelity-sim2real-gap.md"
    page.write_text(
        "---\ntype: concept\ntags: [simulation, sim2real]\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["physics_concept_crosslink"] == []


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["physics_concept_crosslink"].append("wiki/concepts/centroidal-dynamics.md")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
