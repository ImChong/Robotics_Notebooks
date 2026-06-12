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
