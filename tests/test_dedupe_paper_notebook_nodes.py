"""dedupe_paper_notebook_nodes stub-stub merge tests."""

from __future__ import annotations

from pathlib import Path

import dedupe_paper_notebook_nodes as dedupe


def _write_stub(
    path: Path,
    *,
    h1: str,
    title: str,
    arxiv: str | None = None,
) -> None:
    arxiv_line = f'arxiv: "{arxiv}"\n' if arxiv else ""
    path.write_text(
        f"""---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-06-11
{arxiv_line}sources:
  - ../../sources/papers/{path.stem.replace("paper-notebook-", "humanoid_pnb_")}.md
summary: "{h1}：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# {h1}

**{title}** 已列入 Paper Notebooks 的 **PROGRESS.md 待深读** 清单。

## 一句话定义

{h1} 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 参考来源

- [Humanoid Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
""",
        encoding="utf-8",
    )


def test_find_stub_stub_merge_pairs_prefers_arxiv_holder(tmp_path, monkeypatch) -> None:
    wiki = tmp_path / "wiki" / "entities"
    wiki.mkdir(parents=True)
    keeper = wiki / "paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md"
    duplicate = wiki / "paper-notebook-child-controller-for-humanoid-imitation-and-live.md"
    _write_stub(
        keeper,
        h1="CHILD",
        title="CHILD: a Whole-Body Humanoid Teleoperation System",
        arxiv="2508.00162",
    )
    _write_stub(
        duplicate,
        h1="CHILD",
        title=(
            "CHILD: Controller for Humanoid Imitation and Live Demonstration "
            "a Whole-Body Humanoid Teleoperation System"
        ),
    )

    monkeypatch.setattr(dedupe, "ROOT", tmp_path)
    monkeypatch.setattr(dedupe, "WIKI", tmp_path / "wiki")

    pairs = dedupe.find_stub_stub_merge_pairs()
    assert pairs == [
        (
            "wiki/entities/paper-notebook-child-controller-for-humanoid-imitation-and-live.md",
            "wiki/entities/paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md",
        )
    ]
