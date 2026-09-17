---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2504.13165"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./ruka-v2-hand.md
sources:
  - ../../sources/papers/humanoid_pnb_ruka-rethinking-the-design-of-humanoid-hands-wit.md
summary: "RUKA：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# RUKA

**RUKA: Rethinking the Design of Humanoid Hands with Learning** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

RUKA 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/ruka-rethinking-the-design-of-humanoid-hands-wit` |
| arXiv | <https://arxiv.org/abs/2504.13165> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 RUKA v1 的待读条目：本库真正已 ingest 的内容在后继硬件 RUKA-v2，v1 自身的设计取舍仍待深读笔记补齐。**

- 现阶段可确认的只有策展元数据：分类 12_Hardware_Design、arXiv 2504.13165、深读笔记待撰写。
- 想看具体设计取舍，应先读本库已 ingest 的 [RUKA-v2 Hand](./ruka-v2-hand.md)——它在 v1 基础上增加 2-DoF 腕与指根外展/内收，并全栈开源。
- 保留本页是为了不漏掉 v1 这个前置工作；在笔记完成前不宜把它当作可引用的技术结论来源。

## 与其他页面的关系

- 后继硬件（本库已 ingest）：[RUKA-v2 Hand](./ruka-v2-hand.md) — 在 v1 基础上增加 **2-DoF 腕** 与 **指根外展/内收**，全栈开源（[ruka-hand-v2.github.io](https://ruka-hand-v2.github.io/)）
- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_ruka-rethinking-the-design-of-humanoid-hands-wit.md](../../sources/papers/humanoid_pnb_ruka-rethinking-the-design-of-humanoid-hands-wit.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2504.13165>

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
