---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2410.03311"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_scaling-large-motion-models-with-million-level-h.md
summary: "本文构建 MotionLib ——首个百万级动作生成数据集，至少比现有同类大 15×，并配分层文本描述（hierarchical text）。用它训练一个大型动作模型，在多样人类活动（含未见类别）上表现强劲，强调数据与模型规模一起放大的重要性。还提出 Motionbook 动作编码：一种紧凑无损的动作表示，以及一个新颖的「2D 无查找（lookup-free）」token 化方法——在保留细粒度细节的同时扩大码本容量。"
---

# Scaling Large Motion Models with Million-Level Human Motions

**Scaling Large Motion Models with Million-Level Human Motions** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文构建 MotionLib ——首个百万级动作生成数据集，至少比现有同类大 15×，并配分层文本描述（hierarchical text）。用它训练一个大型动作模型，在多样人类活动（含未见类别）上表现强劲，强调数据与模型规模一起放大的重要性。还提出 Motionbook 动作编码：一种紧凑无损的动作表示，以及一个新颖的「2D 无查找（lookup-free）」token 化方法——在保留细粒度细节的同时扩大码本容量。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| MotionLib | 百万级动作生成数据集 |
| Motionbook | 本文动作编码/token 化方法 |
| Hierarchical Text | 分层文本描述 |
| Lookup-free Tokenizer | 无查找 token 化（2D） |
| Codebook | 码本（量化字典） |
| Scaling | 数据/模型规模化 |

## 为什么重要

- **"数据 + 模型一起放大"是动作生成的 scaling law**，对人形动作生成（Being-M0.5、UniAct）有指导；
- **无查找 token 化**缓解码本容量瓶颈，是动作离散化的改进方向；
- **分层文本**有助于细粒度可控生成；
- 大规模动作模型可作人形"语言→动作"的上游先验。

## 解决什么问题

动作生成缺**大规模数据**与**好的动作 token 化**： - 现有数据集**小**，难支撑大模型； - 传统码本 token 化在**容量与细节**间难两全。

论文要：建**百万级**数据集、训**大型动作模型**、并设计**更好的动作编码**。

## 核心机制

1. **MotionLib**：首个百万级动作数据集（≥15×、分层文本）；
2. **大型动作模型 + 规模化研究**：数据与模型一起放大；
3. **Motionbook 编码**：紧凑无损 + 2D 无查找 token 化；
4. **强泛化**：多样人类活动含未见类别。

方法拆解（深读笔记小节）：MotionLib：百万级 + 分层文本；大型动作模型 + 规模化；Motionbook：紧凑无损 + 2D 无查找 token 化；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Scaling_Large_Motion_Models_with_Million-Level_Human_Motions/Scaling_Large_Motion_Models_with_Million-Level_Human_Motions.html> |
| arXiv | <https://arxiv.org/abs/2410.03311> |
| 作者 | Ye Wang、Sipeng Zheng、Bin Cao、Qianshan Wei、Qin Jin、Zongqing Lu（BAAI / 人大等） |
| 发表 | 2024 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_scaling-large-motion-models-with-million-level-h.md](../../sources/papers/humanoid_pnb_scaling-large-motion-models-with-million-level-h.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Scaling_Large_Motion_Models_with_Million-Level_Human_Motions/Scaling_Large_Motion_Models_with_Million-Level_Human_Motions.html>
- 论文：<https://arxiv.org/abs/2410.03311>

## 推荐继续阅读

- [机器人论文阅读笔记：Scaling Large Motion Models with Million-Level Human Motions](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Scaling_Large_Motion_Models_with_Million-Level_Human_Motions/Scaling_Large_Motion_Models_with_Million-Level_Human_Motions.html)
