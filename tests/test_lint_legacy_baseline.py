"""Tests for schema/lint-legacy-baseline.json and blocking checks on new wiki pages."""

from __future__ import annotations

import json
from pathlib import Path

import lint_wiki as lw
import pytest

REAL_REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_BASELINE = REAL_REPO_ROOT / "schema" / "lint-legacy-baseline.json"


def test_real_legacy_baseline_is_sorted_unique_and_existing() -> None:
    data = json.loads(REAL_BASELINE.read_text(encoding="utf-8"))
    for key in ("missing_abbrev_glossary", "entity_missing_outgoing"):
        items = data[key]
        assert items == sorted(set(items)), f"{key} 须有序且无重复"
        for rel in items:
            assert (REAL_REPO_ROOT / rel).is_file(), f"基线中不存在的路径: {rel}"


def test_baseline_grandfathers_legacy_and_blocks_new_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    (tmp_path / "schema").mkdir()
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "entities").mkdir(parents=True)
    (wiki / "methods").mkdir(parents=True)
    (wiki / "tasks").mkdir(parents=True)

    # 被引用目标方法/任务页
    m1 = wiki / "methods" / "ppo.md"
    m1.write_text(
        "---\ntype: method\nstatus: complete\nsummary: PPO\n---\n# PPO\n\n## 英文缩写速查\n\n[概念](../concepts/ok.md)\n",
        encoding="utf-8",
    )
    t1 = wiki / "tasks" / "walk.md"
    t1.write_text(
        "---\ntype: task\nstatus: complete\nsummary: Walk\n---\n# Walk\n\n## 英文缩写速查\n\n[PPO](../methods/ppo.md)\n",
        encoding="utf-8",
    )
    ok_concept = wiki / "concepts" / "ok.md"
    ok_concept.write_text(
        "---\ntype: concept\nstatus: complete\nsummary: OK\n---\n# OK\n\n一句话。\n\n## 英文缩写速查\n\n| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n\n"
        "## 为什么重要\n重要。\n\n## 关联页面\n- [PPO](../methods/ppo.md)\n\n## 参考来源\n- [s](../../sources/a.md)\n",
        encoding="utf-8",
    )

    # 历史豁免页：缺英文缩写速查 + entity 缺 methods/tasks 出边
    legacy_concept = wiki / "concepts" / "legacy-no-abbrev.md"
    legacy_concept.write_text(
        "---\ntype: concept\nstatus: stub\nsummary: Legacy\n---\n# Legacy\n\n正文。\n",
        encoding="utf-8",
    )
    legacy_entity = wiki / "entities" / "legacy-tool.md"
    legacy_entity.write_text(
        "---\ntype: entity\nstatus: stub\n---\n# Tool\n\n## 英文缩写速查\n\n正文。\n",
        encoding="utf-8",
    )

    (tmp_path / "schema" / "lint-legacy-baseline.json").write_text(
        json.dumps(
            {
                "missing_abbrev_glossary": ["wiki/concepts/legacy-no-abbrev.md"],
                "entity_missing_outgoing": ["wiki/entities/legacy-tool.md"],
            }
        ),
        encoding="utf-8",
    )

    # 1) 仅含豁免历史页：不应触发新增项阻塞错误
    pages = [m1, t1, ok_concept, legacy_concept, legacy_entity]
    res = lw._empty_results()
    baseline = lw._load_legacy_baseline()
    assert baseline is not None
    lw._check_per_page(pages, {}, {}, res, baseline=baseline)
    lw._check_frontmatter(pages, res, require_status=True)
    lw._check_entities_outgoing(pages, res, baseline["entity_missing_outgoing"])
    assert res["missing_abbrev_glossary_new"] == []
    assert res["entity_missing_outgoing"] == []
    assert res["invalid_status"] == []
    assert res["missing_abbrev_glossary"] == ["wiki/concepts/legacy-no-abbrev.md"]
    assert len(res["entity_missing_outgoing_legacy"]) == 1

    # 2) 合规新实体不报错；新增页面缺 status、缺英文缩写速查、或新 entity 缺出边 → 进入阻塞列表
    new_ok = wiki / "entities" / "new-good-entity.md"
    new_ok.write_text(
        "---\ntype: entity\nstatus: complete\n---\n# Good\n\n## 英文缩写速查\n\n- [PPO](../methods/ppo.md)\n- [Walk](../tasks/walk.md)\n",
        encoding="utf-8",
    )
    new_bad = wiki / "entities" / "new-bad-entity.md"
    new_bad.write_text("---\ntype: entity\n---\n# Bad\n\n无出边也无缩写。\n", encoding="utf-8")
    pages.extend([new_ok, new_bad])
    res2 = lw._empty_results()
    lw._check_per_page(pages, {}, {}, res2, baseline=baseline)
    lw._check_frontmatter(pages, res2, require_status=True)
    lw._check_entities_outgoing(pages, res2, baseline["entity_missing_outgoing"])
    assert res2["missing_abbrev_glossary_new"] == ["wiki/entities/new-bad-entity.md"]
    assert res2["entity_missing_outgoing"] == ["wiki/entities/new-bad-entity.md (当前出边: 0)"]
    assert res2["invalid_status"] == ["wiki/entities/new-bad-entity.md（缺少 status 字段）"]
