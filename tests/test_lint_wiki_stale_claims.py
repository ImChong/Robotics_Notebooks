"""Tests for V24 lint check: 陈旧声明（stale claim）巡检（信息型）。"""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def _setup_wiki(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki" / "concepts"
    wiki.mkdir(parents=True)
    return wiki


def _page(wiki: Path, name: str, updated: str, tags: list[str], body: str) -> Path:
    tag_block = "".join(f"  - {t}\n" for t in tags)
    page = wiki / name
    page.write_text(
        f"---\ntype: concept\ntags:\n{tag_block}updated: {updated}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return page


def _run(pages: list[Path]) -> dict:
    results = lw._empty_results()
    lw._check_stale_claims(pages, results)
    return results


def test_stale_claim_flagged_when_newer_same_tag_page_exists(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    old = _page(wiki, "old.md", "2025-01-01", ["vision"], "本方法是当前最强的视觉骨干。")
    newer = _page(wiki, "newer.md", "2026-01-01", ["vision"], "更新的视觉骨干综述。")
    results = _run([old, newer])
    assert len(results["stale_claims"]) == 1
    record = results["stale_claims"][0]
    assert "old.md" in record and "当前最强" in record and "newer.md" in record


def test_no_flag_when_claim_page_is_newest(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    newest = _page(wiki, "a.md", "2026-06-01", ["vision"], "这是最新的 SOTA 结果。")
    older = _page(wiki, "b.md", "2025-01-01", ["vision"], "较早的综述。")
    results = _run([newest, older])
    assert results["stale_claims"] == []


def test_no_flag_without_shared_tag(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(wiki, "a.md", "2025-01-01", ["vision"], "当前最强的视觉骨干。")
    other = _page(wiki, "b.md", "2026-01-01", ["control"], "更晚但不同主题。")
    results = _run([claim, other])
    assert results["stale_claims"] == []


def test_no_flag_without_absolute_phrasing(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    plain = _page(wiki, "a.md", "2025-01-01", ["vision"], "一种常规的视觉骨干，无绝对化措辞。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vision"], "更晚的同主题页。")
    results = _run([plain, newer])
    assert results["stale_claims"] == []


def test_claim_inside_code_block_is_ignored(tmp_path, monkeypatch) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(wiki, "a.md", "2025-01-01", ["vision"], "```\nSOTA\n```\n普通正文。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vision"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert results["stale_claims"] == []


def test_sota_substring_in_word_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「SOTA」出现在 Minnesota 等词内不应误报（需词边界匹配）。
    claim = _page(
        wiki := _setup_wiki(tmp_path, monkeypatch),
        "a.md",
        "2025-01-01",
        ["vla"],
        "作者来自 University of Minnesota。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["vla"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert results["stale_claims"] == []


def test_negated_claim_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「不是 SoTA」是辟谣式写法，本身在否认断言，不应报陈旧声明。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["vla"],
        "报告 96% 可用会话率，这是部署证据，不是策略 SoTA。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["vla"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_cannot_be_read_as_claim_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「不可直接当 SOTA 通才」与「不是 SoTA」同属辟谣式写法，在否认该断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["humanoid"],
        "BeyondMimic 行不可直接当 SOTA 通才。论文自己把它标成 specialist。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["humanoid"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_insufficient_evidence_negation_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「不足以支撑『全面 SOTA』叙事」把否认落在谓词上，与「不是 SoTA」同为辟谣。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["wam"],
        "饱和榜易误读：LIBERO / RoboTwin 的 0.1–0.2 pp 不足以支撑「全面 SOTA」叙事。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["wam"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_not_responsible_for_negation_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「不负责单独变成 SOTA VLA」同为「不 + 谓词」的辟谣，在否认该断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["world-models"],
        "WM 负责在语义脆弱任务上选更一致的块，不负责单独变成 SOTA VLA。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["world-models"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_postpositioned_overread_cue_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「读『SOTA 碾压』会过读」是后置辟谣：落笔顺序与「不是 SoTA」相反，语义同为否认。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["world-model"],
        "门控误差领先很窄：0.066 vs 0.067，读「SOTA 碾压」会过读。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["world-model"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_overread_cue_in_next_sentence_does_not_exempt(tmp_path, monkeypatch) -> None:
    # 后置线索同样只在命中词同句内生效：下一句的「过读」不应放过本句的真实断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["world-model"],
        "本方法仍是该任务的 SOTA。别的指标才容易过读。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["world-model"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert len(results["stale_claims"]) == 1
    assert "SOTA" in results["stale_claims"][0]


def test_negation_cue_in_previous_sentence_does_not_exempt(tmp_path, monkeypatch) -> None:
    # 否定线索只在命中词同句内生效：上一句的「不可」不应放过本句的真实断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["humanoid"],
        "该权重不可商用。本方法仍是该任务的 SOTA。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["humanoid"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert len(results["stale_claims"]) == 1
    assert "SOTA" in results["stale_claims"][0]


def test_library_page_title_reference_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「VLA SOTA Leaderboard」是库内页面标题，引用它属导航而非本页断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["vla"],
        "对照 [VLA SOTA Leaderboard](./vla-sota-leaderboard.md) 即可，本工作不重跑。",
    )
    leaderboard = _page(wiki, "vla-sota-leaderboard.md", "2025-01-01", ["vla"], "摘录榜。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vla"], "更晚的同主题页。")
    assert _run([claim, leaderboard, newer])["stale_claims"] == []


def test_runtime_object_latest_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「最新 pending 帧」描述系统行为，不会随领域进展过时。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(wiki, "a.md", "2025-01-01", ["vla"], "服务端只保留最新 pending 帧。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vla"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_runtime_candidate_latest_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「只在最新候选过门后刷新」描述预览服务的运行时行为，与「最新 pending 帧」同类。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["diagrams"],
        "本机监视单个 JSON，只在最新候选过门后刷新；失败保留 last-good。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["diagrams"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_sota_as_baseline_label_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「相对经典/SOTA 基线」里的 SOTA 是对照组的类别名，不是本页对自身的断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["sim2real"],
        "相对经典/SOTA 基线，证书平均收窄 51.6% ± 16%。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["sim2real"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_sota_not_followed_by_baseline_is_still_flagged(tmp_path, monkeypatch) -> None:
    # 豁免只在「基线」紧跟命中词时生效：隔着断言正文的「基线」不构成对照组标签。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["sim2real"],
        "本方法仍是该任务的 SOTA，全面超过论文里的四条基线。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["sim2real"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert len(results["stale_claims"]) == 1
    assert "SOTA" in results["stale_claims"][0]


def test_runtime_math_quantity_latest_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「最新 \((\mathbf{q},\mathbf{e})\)」是写成行内公式的运行时量，与「最新状态」同类。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["teleoperation"],
        r"内层根据最新 \((\mathbf{q},\mathbf{e})\) 生成接触相关手指协调。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["teleoperation"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_latest_before_unrelated_math_line_is_still_flagged(tmp_path, monkeypatch) -> None:
    # 豁免只在公式紧跟命中词时生效：隔着断言正文的公式不构成运行时对象。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "2025.md",
        "2025-01-01",
        ["vla"],
        r"这是最新的开源实现，其目标为 \(\min_\theta L(\theta)\)。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["vla"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert len(results["stale_claims"]) == 1
    assert "最新" in results["stale_claims"][0]


def test_abbrev_glossary_entry_is_not_flagged(tmp_path, monkeypatch) -> None:
    # 「英文缩写速查」区块是词条释义表（写作规范固定区块），不是本页断言。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["vision"],
        "## 一句话定义\n\n常规语义分割基准。\n\n"
        "## 英文缩写速查\n\n"
        "| 缩写 | 英文全称 | 简要说明 |\n"
        "|------|----------|----------|\n"
        "| SOTA | State of the Art | 排行榜对照参考 |\n\n"
        "## 为什么重要\n\n教学与历史对照常用。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["vision"], "更晚的同主题页。")
    assert _run([claim, newer])["stale_claims"] == []


def test_claim_outside_abbrev_glossary_is_still_flagged(tmp_path, monkeypatch) -> None:
    # 豁免只剥离速查区块本身：区块外的正文断言仍须命中。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(
        wiki,
        "a.md",
        "2025-01-01",
        ["vision"],
        "## 英文缩写速查\n\n| SOTA | State of the Art | 排行榜对照参考 |\n\n"
        "## 为什么重要\n\n本方法仍是该任务的 SOTA。",
    )
    newer = _page(wiki, "b.md", "2026-01-01", ["vision"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert len(results["stale_claims"]) == 1
    assert "SOTA" in results["stale_claims"][0]


def test_plain_latest_claim_is_still_flagged(tmp_path, monkeypatch) -> None:
    # 豁免只针对结构性误报：无否定/无页面名/非运行时对象的断言仍须命中。
    wiki = _setup_wiki(tmp_path, monkeypatch)
    claim = _page(wiki, "a.md", "2025-01-01", ["vla"], "这是最新的开源实现。")
    newer = _page(wiki, "b.md", "2026-01-01", ["vla"], "更晚的同主题页。")
    results = _run([claim, newer])
    assert len(results["stale_claims"]) == 1
    assert "最新" in results["stale_claims"][0]


def test_stale_claims_is_info_only(tmp_path, monkeypatch) -> None:
    results = lw._empty_results()
    results["stale_claims"].append("wiki/concepts/old.md（含绝对化措辞「SOTA」...）")
    assert lw._failing_total(results) == 0
    assert lw._info_total(results) == 1
