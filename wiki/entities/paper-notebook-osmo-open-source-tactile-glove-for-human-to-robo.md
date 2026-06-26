---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2512.08920"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_osmo-open-source-tactile-glove-for-human-to-robo.md
summary: "OSMO 的核心思路是：让人和机器人戴上「同一只」触觉手套——人戴它采数据、机器手也装它来执行，从而同时抹平视觉外观差异和触觉模态差异，把视频学不到的连续法向力/剪切力反馈，原封不动地从人手迁移到机器手，最终仅靠人类演示就训出能持续保压擦拭的策略。"
---

# OSMO

**OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

OSMO 的核心思路是：让人和机器人戴上「同一只」触觉手套——人戴它采数据、机器手也装它来执行，从而同时抹平视觉外观差异和触觉模态差异，把视频学不到的连续法向力/剪切力反馈，原封不动地从人手迁移到机器手，最终仅靠人类演示就训出能持续保压擦拭的策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer.html> |
| arXiv | <https://arxiv.org/abs/2512.08920> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_osmo-open-source-tactile-glove-for-human-to-robo.md](../../sources/papers/humanoid_pnb_osmo-open-source-tactile-glove-for-human-to-robo.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer.html>
- 论文：<https://arxiv.org/abs/2512.08920>

## 推荐继续阅读

- [机器人论文阅读笔记：OSMO](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer.html)
