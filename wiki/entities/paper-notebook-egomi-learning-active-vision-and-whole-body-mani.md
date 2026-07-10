---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.00153"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_egomi.md
summary: "机器人从人类视频学操作，要跨越具身差距。人在做任务时会主动协调头与手，用动态视角变化与视觉搜索策略。EgoMI 捕捉同步的末端执行器与头部轨迹，可迁移到半人形机器人；并引入一个记忆增强策略（memory-augmented policy），选择性纳入历史观测以应对视角切换。在带可动相机头的双臂机器人上测试：显式建模头部运动的策略持续优于基线，说明协调的手眼学习能有效弥合人-机具身差距（针对半人形）。"
---

# EgoMI

**EgoMI: Learning Active Vision and Whole-Body Manipulation from Egocentric Human Demonstrations** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

机器人从人类视频学操作，要跨越具身差距。人在做任务时会主动协调头与手，用动态视角变化与视觉搜索策略。EgoMI 捕捉同步的末端执行器与头部轨迹，可迁移到半人形机器人；并引入一个记忆增强策略（memory-augmented policy），选择性纳入历史观测以应对视角切换。在带可动相机头的双臂机器人上测试：显式建模头部运动的策略持续优于基线，说明协调的手眼学习能有效弥合人-机具身差距（针对半人形）。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Active Vision | 主动视觉，主动调整视角/搜索 |
| Head-Hand Coordination | 头手协调 |
| Memory-Augmented | 记忆增强，选择性用历史观测 |
| Embodiment Gap | 具身差距，人与机器人形态差异 |
| Actuated Camera Head | 可动相机头 |
| Semi-humanoid | 半人形（双臂 + 可动头） |

## 为什么重要

- **主动视觉（头手协调）是人类操作的隐藏要素**，忽略它会限制从人类视频学习的上限；
- **记忆增强**对视角动态变化的任务很关键；
- **半人形（双臂 + 可动头）**是连接人类数据与人形的实用载体；
- 与 Vision in Action、Learning to Look 等主动感知工作呼应。

## 解决什么问题

从人类视频学操作有**具身差距**，且： - 人会**主动协调头与手**（动态视角、视觉搜索），机器人若忽略则学不好； - **视角快速切换**让策略难以利用历史观测。

EgoMI 要：把**头部主动运动**显式建模，并用**记忆**应对视角切换，迁移到半人形。

## 核心机制

1. **同步末端 + 头部轨迹捕捉**：把主动视觉纳入学习；
2. **记忆增强策略**：选择性用历史观测应对视角切换；
3. **显式头部运动建模**：弥合人-机具身差距；
4. **半人形验证**：可动相机头双臂机器人上优于基线。

方法拆解（深读笔记小节）：同步捕捉末端 + 头部轨迹；记忆增强策略（应对视角切换）；显式头部运动建模；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos.html> |
| arXiv | <https://arxiv.org/abs/2511.00153> |
| 作者 | Justin Yu、Yide Shentu、Di Wu、Pieter Abbeel、Ken Goldberg、Philipp Wu（UC Berkeley） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_egomi.md](../../sources/papers/humanoid_pnb_egomi.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos.html>
- 论文：<https://arxiv.org/abs/2511.00153>

## 推荐继续阅读

- [机器人论文阅读笔记：EgoMI](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos.html)
