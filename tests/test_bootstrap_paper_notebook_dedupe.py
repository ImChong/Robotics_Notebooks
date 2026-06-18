"""bootstrap_paper_notebook_knowledge 去重逻辑单元测试。"""

from __future__ import annotations

import bootstrap_paper_notebook_knowledge as bootstrap


def _completed_paper(**overrides: object) -> dict:
    base = {
        "folder": "papers/01_Foundational_RL/ADD_Adversarial_Differential_Discriminators",
        "dir": "ADD_Adversarial_Differential_Discriminators",
        "title": "ADD: Adversarial Differential Discriminators",
        "arxiv": "2505.04961",
        "url": "https://example.test/add.html",
        "category": "01_Foundational_RL",
    }
    base.update(overrides)
    return base


def _planned_alias(**overrides: object) -> dict:
    base = {
        "folder": "papers/01_Foundational_RL/add-adversarial-disentanglement-and-distillation",
        "dir": "add-adversarial-disentanglement-and-distillation",
        "title": "ADD",
        "arxiv": None,
        "url": "https://example.test/add-alias.html",
        "category": "01_Foundational_RL",
        "planned": True,
        "from_progress_md": True,
    }
    base.update(overrides)
    return base


def test_merge_paper_catalog_drops_planned_alias_when_note_exists() -> None:
    completed = [_completed_paper()]
    planned = [_planned_alias()]
    merged = bootstrap.merge_paper_catalog(completed, planned)
    assert len(merged) == 1
    assert merged[0]["dir"] == "ADD_Adversarial_Differential_Discriminators"
    assert not merged[0].get("planned")


def test_merge_paper_catalog_drops_planned_alias_without_arxiv_when_label_has_arxiv() -> None:
    canonical = _planned_alias(
        dir="child-a-whole-body-humanoid-teleoperation-system",
        title="CHILD: a Whole-Body Humanoid Teleoperation System",
        arxiv="2508.00162",
        folder="papers/07_Teleoperation/child-a-whole-body-humanoid-teleoperation-system",
        category="07_Teleoperation",
        from_progress_md=False,
    )
    alias = _planned_alias(
        dir="child-controller-for-humanoid-imitation-and-live",
        title=(
            "CHILD: Controller for Humanoid Imitation and Live Demonstration "
            "a Whole-Body Humanoid Teleoperation System"
        ),
        arxiv=None,
        folder="papers/07_Teleoperation/child-controller-for-humanoid-imitation-and-live",
        category="07_Teleoperation",
    )
    merged = bootstrap.merge_paper_catalog([canonical], [alias])
    assert len(merged) == 1
    assert merged[0]["dir"] == "child-a-whole-body-humanoid-teleoperation-system"


def test_dedupe_category_entries_keeps_one_row_per_wiki_target() -> None:
    completed = _completed_paper()
    planned = _planned_alias()
    wiki = "wiki/methods/add.md"
    papers_in_cat = [
        (completed, wiki, {}),
        (planned, wiki, {}),
    ]
    deduped = bootstrap.dedupe_category_entries(papers_in_cat)
    assert len(deduped) == 1
    assert deduped[0][0]["dir"] == "ADD_Adversarial_Differential_Discriminators"
