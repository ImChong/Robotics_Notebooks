"""Tests for Paper Notebooks title cleanup and entity matching."""

from __future__ import annotations

from sync_paper_notebook_links import (
    auto_match,
    clean_display_title,
    collect_wiki_index,
    pick_entity_or_single,
    short_label,
)


def test_clean_display_title_strips_markdown_link() -> None:
    title = "[MuJoCo Playground](https://playground.mujoco.org/)"
    assert clean_display_title(title) == "MuJoCo Playground"


def test_short_label_does_not_keep_brackets_before_colon() -> None:
    title = "[EgoPoser: Robust Real-Time Egocentric Pose Estimation]"
    assert short_label(title) == "EgoPoser"


def test_short_label_does_not_keep_brackets_from_link_title() -> None:
    title = "[Zeroth Bot](https://github.com/zeroth-robotics/zeroth-bot)"
    assert short_label(title) == "Zeroth Bot"


def test_pick_entity_or_single_prefers_non_paper_entity() -> None:
    candidates = {
        "wiki/entities/paper-notebook-mujoco-playground-https-playground-mujoco-org.md",
        "wiki/entities/mujoco-playground.md",
    }
    picked = pick_entity_or_single(candidates)
    assert picked == ["wiki/entities/mujoco-playground.md"]


def test_auto_match_maps_mujoco_playground_to_repo_entity(tmp_path, monkeypatch) -> None:
  # Use live wiki index from repo; only assert mapping exists.
    wiki_index = collect_wiki_index()
    paper = {"title": "[MuJoCo Playground](https://playground.mujoco.org/)", "arxiv": None}
    picked = auto_match(paper, wiki_index)
    assert picked == ["wiki/entities/mujoco-playground.md"]


def test_category_entry_suffix_for_existing_entity() -> None:
    import bootstrap_paper_notebook_knowledge as bootstrap

    paper = {"planned": True, "url": "https://example.test"}
    assert (
        bootstrap.category_entry_suffix(paper, "wiki/entities/mujoco-playground.md")
        == "见 wiki 实体页"
    )
    assert bootstrap.category_entry_suffix(paper, "wiki/entities/paper-notebook-demo.md") == "待深读"
