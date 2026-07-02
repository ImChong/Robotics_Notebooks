"""Tests for V27 lint check: 接触/力控/操作概念页交叉链路巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    return wiki


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_contact_control_crosslink(pages, results)
    return results


def test_concept_with_hub_backlink_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "impedance-control.md"
    page.write_text(
        "---\ntype: concept\ntags: [control, impedance-control, force-control]\n---\n"
        "阻抗控制在接触力旋量闭环链路中的定位，见 "
        "[端到端 Query](../queries/contact-wrench-closed-loop.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["contact_control_crosslink"] == []


def test_inline_tag_concept_without_hub_is_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "contact-rich-manipulation.md"
    page.write_text(
        "---\ntype: concept\ntags: [manipulation, contact, force-control]\n---\n"
        "接触丰富操作正文，未回链任何力控枢纽。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["contact_control_crosslink"] == ["wiki/concepts/contact-rich-manipulation.md"]


def test_list_style_tag_concept_flagged(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "tactile-sensing.md"
    page.write_text(
        "---\ntype: concept\ntags:\n  - perception\n  - tactile-sensing\n---\n触觉感知正文。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["contact_control_crosslink"] == ["wiki/concepts/tactile-sensing.md"]


def test_topic_hub_backlink_also_passes(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "force-control-basics.md"
    page.write_text(
        "---\ntype: concept\ntags: [control, manipulation, force-control]\n---\n"
        "力控基础，见专题 [接触力控](../overview/topic-contact-force-control.md)。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["contact_control_crosslink"] == []


def test_untagged_concept_is_ignored(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "diffusion-policy.md"
    page.write_text(
        "---\ntype: concept\ntags: [imitation-learning, generative]\n---\n与接触力控无关。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["contact_control_crosslink"] == []


def test_hub_pages_exempt_from_self_check(tmp_path: Path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    page = wiki / "concepts" / "contact-wrench-closed-loop.md"
    page.write_text(
        "---\ntype: concept\ntags: [contact, force-control]\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    results = _run([page])
    assert results["contact_control_crosslink"] == []


def test_info_only_does_not_count_toward_failing_total() -> None:
    results = lw._empty_results()
    results["contact_control_crosslink"].append("wiki/concepts/contact-rich-manipulation.md")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
