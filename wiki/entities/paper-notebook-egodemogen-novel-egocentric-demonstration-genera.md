---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2509.22578"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_egodemogen.md
summary: "基于模仿学习的视觉运动策略表现强，但常对第一视角视角变化（egocentric viewpoint shifts）敏感。EgoDemoGen 是一个框架，在无需多视角数据的前提下，生成新第一视角下的「观测-动作」配对演示。它由两部分组成：① EgoTrajTransfer——用运动技能分割 + 几何感知变换 + 逆运动学滤波，把机器人轨迹迁移到新第一视角帧；② EgoViewTransfer——一个条件视频生成模型，把新视角重投影的场景与渲染的机器人运动融合，合成逼真观测。实验：仿真策略成功率绝对提升 +24.6% 与 +16.9%；真机在不同视角条件下提升 +16.0% 与 +23.0%。"
---

# EgoDemoGen

**EgoDemoGen: Egocentric Demonstration Generation for Viewpoint Generalization in Robotic Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

基于模仿学习的视觉运动策略表现强，但常对第一视角视角变化（egocentric viewpoint shifts）敏感。EgoDemoGen 是一个框架，在无需多视角数据的前提下，生成新第一视角下的「观测-动作」配对演示。它由两部分组成：① EgoTrajTransfer——用运动技能分割 + 几何感知变换 + 逆运动学滤波，把机器人轨迹迁移到新第一视角帧；② EgoViewTransfer——一个条件视频生成模型，把新视角重投影的场景与渲染的机器人运动融合，合成逼真观测。实验：仿真策略成功率绝对提升 +24.6% 与 +16.9%；真机在不同视角条件下提升 +16.0% 与 +23.0%。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Viewpoint Generalization | 视角泛化，应对视角变化 |
| EgoTrajTransfer | 轨迹迁移到新第一视角 |
| EgoViewTransfer | 条件视频生成新视角观测 |
| IK Filtering | 逆运动学滤波 |
| Geometry-aware | 几何感知变换 |
| Paired Demo | 观测-动作配对演示 |

## 为什么重要

- **视角敏感是视觉运动策略的通病**，尤其第一视角人形；
- **"生成数据补视角"**比采集多视角更省；
- **轨迹迁移 + 视频生成**组合是合成配对演示的有效范式；
- 与 EgoMI（主动视觉）从不同角度解决视角问题。

## 解决什么问题

视觉运动策略**对第一视角视角变化敏感**： - 头部/相机视角一变，策略就退化； - 采集**多视角数据**昂贵。

EgoDemoGen 要：**无需多视角数据**，**生成**新视角下的配对演示来提升视角泛化。

## 核心机制

1. **无需多视角数据的视角泛化**：生成新第一视角配对演示；
2. **EgoTrajTransfer**：技能分割 + 几何变换 + IK 滤波迁移轨迹；
3. **EgoViewTransfer**：条件视频生成逼真新视角观测；
4. **显著提升**：仿真 +24.6/16.9%、真机 +16/23%。

方法拆解（深读笔记小节）：EgoTrajTransfer：轨迹迁到新视角；EgoViewTransfer：合成逼真新视角观测；结果（无需多视角数据）；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization.html> |
| arXiv | <https://arxiv.org/abs/2509.22578> |
| 作者 | Yuan Xu、Jiabing Yang、Xiaofeng Wang、Zheng Zhu、Yan Huang、Liang Wang 等 |
| 发表 | 2025 年 9 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_egodemogen.md](../../sources/papers/humanoid_pnb_egodemogen.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization.html>
- 论文：<https://arxiv.org/abs/2509.22578>

## 推荐继续阅读

- [机器人论文阅读笔记：EgoDemoGen](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization.html)
