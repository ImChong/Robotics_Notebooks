---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_spatial-relationship-preserving-character-motion.md
summary: "本文提出一种编辑与重定向运动的新方法，专门针对涉及身体部位之间紧密交互的运动——可以是单个或多个关节化角色之间（如跳舞、摔跤、剑斗），也可以是角色与受限环境之间（如钻进车里）。核心是引入一个结构——交互网格（interaction mesh）——来表示空间关系。通过在动画各帧上最小化交互网格的局部形变（local deformation），这些紧密空间关系在运动编辑/重定向时被保持，同时减少不当的相互穿插（interpenetration）。交互网格表示通用，对单/多角色的交互身体部位以及环境中的物体提供统一处理，适用于跳舞（单角色不同部位紧密交互）、摔跤/格斗游戏（多角色交互）等多种场景。"
---

# Spatial Relationship Preserving Character Motion Adaptation

**Spatial Relationship Preserving Character Motion Adaptation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：13_Physics-Based_Animation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文提出一种编辑与重定向运动的新方法，专门针对涉及身体部位之间紧密交互的运动——可以是单个或多个关节化角色之间（如跳舞、摔跤、剑斗），也可以是角色与受限环境之间（如钻进车里）。核心是引入一个结构——交互网格（interaction mesh）——来表示空间关系。通过在动画各帧上最小化交互网格的局部形变（local deformation），这些紧密空间关系在运动编辑/重定向时被保持，同时减少不当的相互穿插（interpenetration）。交互网格表示通用，对单/多角色的交互身体部位以及环境中的物体提供统一处理，适用于跳舞（单角色不同部位紧密交互）、摔跤/格斗游戏（多角色交互）等多种场景。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Interaction Mesh | 交互网格，表示部位/角色/物体间空间关系 |
| Motion Adaptation | 运动适配（编辑 + 重定向） |
| Spatial Relationship | 空间关系（紧密交互） |
| Local Deformation | 局部形变（最小化以保关系） |
| Interpenetration | 相互穿插（要减少） |
| Retargeting | 重定向（到不同体型/环境） |

## 为什么重要

- **"交互网格 + 保空间关系"对人-人/人-物交互重定向有直接借鉴**：呼应本仓 04 的 PAIR（接触/关系保持的交互重定向）；
- **减少穿插**正是人形动作重定向/接触任务要解决的；
- **统一处理部位/角色/物体**的思想，对多接触全身任务的关系建模有价值；
- 作为经典图形学工作，为当代"接触/关系保持"的数据生成提供思想源头。

## 解决什么问题

编辑/重定向**紧密交互**的运动很难： - 跳舞、摔跤、剑斗、钻车等涉及**部位/角色/环境**间**紧密空间关系**； - 朴素编辑会**破坏关系**或产生**穿插**； - 需要一种**通用**表示统一处理单/多角色与物体。

本文要：在编辑/重定向时**保持空间关系**、**减少穿插**。

## 核心机制

1. **交互网格（interaction mesh）**：统一表示部位/角色/物体间空间关系；
2. **最小化局部形变保关系**：编辑/重定向时保持紧密交互、减少穿插；
3. **通用统一**：单/多角色与环境物体一体处理；
4. **经典基础**：紧密交互运动编辑的奠基性工作（SIGGRAPH 2010）。

方法拆解（深读笔记小节）：交互网格表示空间关系；最小化局部形变以保关系；统一、通用；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Spatial_Relationship_Preserving_Character_Motion_Adaptation/Spatial_Relationship_Preserving_Character_Motion_Adaptation.html> |
| 作者 | Edmond S. L. Ho、Taku Komura、Chiew-Lan Tai |
| 发表 | 2010 年（SIGGRAPH 2010 / ACM TOG 29(4)） |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_spatial-relationship-preserving-character-motion.md](../../sources/papers/humanoid_pnb_spatial-relationship-preserving-character-motion.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Spatial_Relationship_Preserving_Character_Motion_Adaptation/Spatial_Relationship_Preserving_Character_Motion_Adaptation.html>

## 推荐继续阅读

- [机器人论文阅读笔记：Spatial Relationship Preserving Character Motion Adaptation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Spatial_Relationship_Preserving_Character_Motion_Adaptation/Spatial_Relationship_Preserving_Character_Motion_Adaptation.html)
