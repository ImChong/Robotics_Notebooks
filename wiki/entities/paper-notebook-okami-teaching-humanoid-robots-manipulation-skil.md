---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2410.11792"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_okami.md
summary: "研究从单段视频演示模仿来教人形机器人操作技能。OKAMI 从单段 RGB-D 视频生成操作计划并导出可执行策略。其核心是物体感知重定向（object-aware retargeting）：让人形复现视频中的人类动作，同时在部署时适应不同物体位置。OKAMI 用开放世界视觉模型识别任务相关物体，并分别重定向身体动作与手部姿态。实验表明 OKAMI 在多变视觉与空间条件下强泛化，在开放世界从观察模仿（imitation from observation）上超越 SOTA 基线。进一步地，用 OKAMI 的 rollout 轨迹训练闭环视觉运动策略，在无需费力遥操作的情况下达平均 79.2% 成功率。"
---

# OKAMI

**OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

研究从单段视频演示模仿来教人形机器人操作技能。OKAMI 从单段 RGB-D 视频生成操作计划并导出可执行策略。其核心是物体感知重定向（object-aware retargeting）：让人形复现视频中的人类动作，同时在部署时适应不同物体位置。OKAMI 用开放世界视觉模型识别任务相关物体，并分别重定向身体动作与手部姿态。实验表明 OKAMI 在多变视觉与空间条件下强泛化，在开放世界从观察模仿（imitation from observation）上超越 SOTA 基线。进一步地，用 OKAMI 的 rollout 轨迹训练闭环视觉运动策略，在无需费力遥操作的情况下达平均 79.2% 成功率。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| OKAMI | 本文方法名 |
| Object-Aware Retargeting | 物体感知重定向 |
| Single Video Imitation | 单段视频模仿 |
| Open-world Vision | 开放世界视觉模型 |
| Imitation from Observation | 从观察模仿（无动作标签） |
| Closed-loop Visuomotor | 闭环视觉运动策略 |

## 为什么重要

- **"单视频 + 物体感知重定向"极大降低教学成本**：看一遍就会、还能适应物体位置；
- **身体/手部分别重定向**是处理具身差异的实用拆分；
- **用 rollout 自举训练闭环策略**免遥操作，是数据飞轮的一环；
- 与 MimicDroid、Masquerade 等"从人类视频学操作"路线互补（同 UT/Yuke Zhu 系）。

## 解决什么问题

教人形操作通常需大量演示/遥操作。能否**从单段视频**就学会？ - 单视频缺动作标签，且**物体位置会变**； - 人-机具身差异需重定向。

OKAMI 要：从**单段 RGB-D 视频**生成计划 + 策略，并能**适应不同物体位置**。

## 核心机制

1. **单视频模仿教人形操作**：从一段 RGB-D 视频生成计划 + 策略；
2. **物体感知重定向**：开放世界识别物体、分别重定向身体/手部、适应物体位置；
3. **超 SOTA 泛化**：开放世界 imitation-from-observation；
4. **闭环策略 79.2%**：用 rollout 训练、无需遥操作。

方法拆解（深读笔记小节）：单 RGB-D 视频 → 操作计划；物体感知重定向（核心）；rollout 训练闭环策略；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation.html> |
| arXiv | <https://arxiv.org/abs/2410.11792> |
| 作者 | Jinhan Li、Yifeng Zhu、Yuqi Xie、Zhenyu Jiang、Mingyo Seo、Georgios Pavlakos、Yuke Zhu（UT Austin） |
| 发表 | 2024 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_okami.md](../../sources/papers/humanoid_pnb_okami.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation.html>
- 论文：<https://arxiv.org/abs/2410.11792>

## 推荐继续阅读

- [机器人论文阅读笔记：OKAMI](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation.html)
