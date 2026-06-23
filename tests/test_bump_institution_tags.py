"""bump_institution_tags：工具实体判定与机构推断。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "bump_institution_tags",
    _REPO / "scripts" / "bump_institution_tags.py",
)
assert _SPEC and _SPEC.loader
bump = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bump)


def test_is_tool_entity_excludes_hardware_platform_concept() -> None:
    assert not bump.is_tool_entity(
        "wiki/entities/humanoid-robot.md",
        ["humanoid", "hardware", "platform", "actuator"],
    )


def test_is_tool_entity_includes_repo_software() -> None:
    assert bump.is_tool_entity(
        "wiki/entities/isaac-lab.md",
        ["entity", "simulator", "isaac", "isaac-sim"],
    )


def test_is_tool_entity_excludes_paper_notebook_stub() -> None:
    assert not bump.is_tool_entity(
        "wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md",
        ["paper", "humanoid-paper-notebooks", "paper-notebook-stub"],
    )


def test_keyword_in_text_rejects_mit_license_false_positive() -> None:
    text = "manim uses the mit license for distribution."
    assert not bump._keyword_in_text("mit", text)


def test_keyword_in_text_rejects_meta_inside_metalhead() -> None:
    assert not bump._keyword_in_text("meta", "metalhead quadruped project")


def test_infer_from_summary_nvidia() -> None:
    registry = bump._load_registry(bump.INSTITUTIONS_PATH)
    alias_map = bump._build_alias_map(registry)
    content = (
        "---\n"
        "tags: [entity, simulator, isaac]\n"
        "summary: NVIDIA 官方 robot learning 框架\n"
        "---\n\n"
        "# Isaac Lab\n\n"
        "正文。\n"
    )
    ids = bump.infer_institution_ids(
        "wiki/entities/isaac-lab.md",
        content,
        registry,
        alias_map,
        bump._label_keywords(registry),
    )
    assert "nvidia" in ids
