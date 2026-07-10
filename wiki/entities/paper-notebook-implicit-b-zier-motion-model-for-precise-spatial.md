---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_implicit-bezier-motion-model-for-precise-spatial.md
summary: "隐式贝塞尔运动模型（Implicit Bézier Motion Model, IBMM）提供对生成运动的细粒度空间与时间控制。它针对此前贝塞尔运动模型（BMM）的关键局限——BMM 只能在均匀时间间隔预测一组固定控制点，使艺术家无法做细粒度时间控制（如在时间上移动控制点、或在需要更多细节的区域增加控制点）。IBMM 在训练时隐式学习贝塞尔拟合，支持任意时间控制点，无需对数据预先拟合，并彻底取消「步幅（stride）」概念，使艺术家可在任意帧约束任意末端关节。此外，IBMM 还为用户引入一项新的全局控制：对运动全局缓入/缓出（ease-in/out）的直接手柄——这是首个在生成自然运动时无需人工标注即可全局控制时间的方法。"
---

# Implicit Bézier Motion Model for Precise Spatial and Temporal Control

**Implicit Bézier Motion Model for Precise Spatial and Temporal Control** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

隐式贝塞尔运动模型（Implicit Bézier Motion Model, IBMM）提供对生成运动的细粒度空间与时间控制。它针对此前贝塞尔运动模型（BMM）的关键局限——BMM 只能在均匀时间间隔预测一组固定控制点，使艺术家无法做细粒度时间控制（如在时间上移动控制点、或在需要更多细节的区域增加控制点）。IBMM 在训练时隐式学习贝塞尔拟合，支持任意时间控制点，无需对数据预先拟合，并彻底取消「步幅（stride）」概念，使艺术家可在任意帧约束任意末端关节。此外，IBMM 还为用户引入一项新的全局控制：对运动全局缓入/缓出（ease-in/out）的直接手柄——这是首个在生成自然运动时无需人工标注即可全局控制时间的方法。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| IBMM | Implicit Bézier Motion Model |
| BMM | （前作）Bézier Motion Model |
| Control Point | 贝塞尔控制点 |
| Stride | 步幅（固定时间间隔，本文取消） |
| Ease-in/out | 缓入/缓出（全局时间控制） |
| End-effector | 末端关节 |

## 为什么重要

- **"任意帧约束任意末端 + 全局控时"是强可控运动表示**，对人形动作编辑/关键帧规划有借鉴；
- **隐式拟合取消步幅**比固定时间网格更灵活，可迁移到机器人轨迹参数化；
- **全局缓入/缓出**对应运动的加减速塑形，与人形动作自然性相关；
- 贝塞尔等紧凑参数化适合作机器人参考轨迹的可控表示。

## 解决什么问题

前作 **BMM** 的时间控制太僵： - 只在**均匀间隔**预测**固定控制点**； - 艺术家**无法**在时间上移动/增加控制点； - 缺**全局时间（缓入/缓出）**控制。

IBMM 要：**任意时间控制点**、**任意帧约束任意末端**、且可**全局控时**，无需人工标注。

## 核心机制

1. **隐式贝塞尔拟合**：任意时间控制点、无需预拟合、取消步幅；
2. **任意帧约束任意末端**：细粒度空间控制；
3. **全局缓入/缓出控制**：首个无需人工标注的全局控时；
4. **面向艺术家工作流**：精确时空控制的自然运动生成。

方法拆解（深读笔记小节）：隐式贝塞尔拟合（取消步幅）；任意帧约束任意末端；全局缓入/缓出控制；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control.html> |
| 作者 | Disney Research Studios（详见项目页） |
| 发表 | 2025 年 12 月（SIGGRAPH MIG 2025） |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_implicit-bezier-motion-model-for-precise-spatial.md](../../sources/papers/humanoid_pnb_implicit-bezier-motion-model-for-precise-spatial.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control.html>

## 推荐继续阅读

- [机器人论文阅读笔记：Implicit Bézier Motion Model for Precise Spatial and Temporal Control](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control.html)
