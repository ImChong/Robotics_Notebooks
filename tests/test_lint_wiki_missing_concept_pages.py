"""Tests for V24 lint check: 缺页概念巡检（missing concept page，信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki" / "concepts"
    wiki.mkdir(parents=True)
    return wiki


def _page(wiki: Path, name: str, body: str) -> Path:
    page = wiki / name
    page.write_text(f"---\ntype: concept\n---\n\n{body}\n", encoding="utf-8")
    return page


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_missing_concept_pages(pages, results)
    return results


def test_term_flagged_when_referenced_by_enough_pages(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", f"本页讨论 **GRPO** 优化方法第 {i} 段。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    assert any("GRPO" in rec for rec in results["missing_concept_pages"])


def test_term_not_flagged_below_threshold(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "提及 **GRPO** 一次。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES - 1)
    ]
    results = _run(pages)
    assert results["missing_concept_pages"] == []


def test_term_with_existing_page_not_flagged(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    # 已有独立页 grpo.md → 即便多页引用也不应建议新建
    pages = [_page(wiki, "grpo.md", "GRPO 概念页。")]
    pages += [
        _page(wiki, f"p{i}.md", "引用 **GRPO**。") for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    assert results["missing_concept_pages"] == []


def test_stopwords_and_paths_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "frontmatter 字段 **type** 与路径 `wiki/concepts/x.md`。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # type 是停用词；含 '/' 与 '.md' 的路径不符合单 token 词形 → 均不入候选
    assert results["missing_concept_pages"] == []


def test_toolchain_stopword_printf_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "控制环里禁止 `printf` 阻塞 I/O。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # printf 是 C 标准库打印调用（实时环反例），非可成页的机器人概念
    assert results["missing_concept_pages"] == []


def test_runtime_command_stopword_stop_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "行为树里 `STOP` 后插入复位动作。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # STOP 是运行时命令名 / 动作枚举值，非单一可成页概念
    assert results["missing_concept_pages"] == []


def test_frontmatter_key_stopword_code_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "官方入口见 frontmatter `code` / 项目页。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # code 是 frontmatter 来源键（仓库入口指针），非可成页的机器人概念
    assert results["missing_concept_pages"] == []


def test_frontmatter_key_stopword_venue_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "| **Venue** | ECCV 2026 |")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # Venue 是核心信息表的发表信息行标签（同 frontmatter venue 来源键），非机器人概念
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_libero_plus_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "扰动增强 **LIBERO-Plus** 零样本 85.1%。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # LIBERO-Plus 是 LIBERO 的扰动增强套件，已由 entities/libero-benchmark.md 专节覆盖
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_joint_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "MuJoCo 在 **joint** 上提供 `armature` 属性。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # joint 的关节属性 / WAM Joint 族两义已由 URDF、world-action-models 等页覆盖
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_reset_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "Gym 风格 `reset` / `step` 接口。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # reset 是环境/策略 API 的方法名（episode 复位），已由 entities/gymnasium.md 释义
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_rgbd_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "输入为 **RGB-D** 观测。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # RGB-D 是传感模态标签，已由六种空间表征 / 三维坐标变换等页覆盖
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_env_ids_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "从 `CartPole-v1` 跨到 `Isaac-Cartpole-v0`。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # 两者都是环境注册 id，同一 cart-pole 概念已由 concepts/cartpole.md 覆盖
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_onpolicyrunner_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "rsl_rl 的循环入口叫 `OnPolicyRunner`。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # 具体实现的类名，本体是 concepts/rl-runner.md 的 on-policy Runner 抽象
    assert results["missing_concept_pages"] == []


def test_covered_elsewhere_base_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    pages = [
        _page(wiki, f"p{i}.md", "TITA 对 `base` 接触直接 terminate。")
        for i in range(lw.MISSING_CONCEPT_PAGE_MIN_PAGES)
    ]
    results = _run(pages)
    # base 的基座连杆 / 权重档名 / 消融条件名三义已由 URDF、浮动基座等页覆盖
    assert results["missing_concept_pages"] == []


def test_case_insensitive_merge(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    half = lw.MISSING_CONCEPT_PAGE_MIN_PAGES // 2 + 1
    pages = [_page(wiki, f"a{i}.md", "讨论 **ViT** 骨干。") for i in range(half)]
    pages += [_page(wiki, f"b{i}.md", "讨论 `vit` 骨干。") for i in range(half)]
    results = _run(pages)
    # 大小写应归并到同一术语计数，达到阈值后被标记一次
    hits = [r for r in results["missing_concept_pages"] if "vit" in r.lower()]
    assert len(hits) == 1


def test_missing_concept_pages_is_info_only(tmp_path, monkeypatch) -> None:
    results = lw._empty_results()
    results["missing_concept_pages"].append("GRPO（被 6 个页面引用...）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
