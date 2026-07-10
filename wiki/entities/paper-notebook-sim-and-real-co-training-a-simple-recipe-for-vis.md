---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2503.24361"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_sim-and-real-co-training.md
summary: "大规模真实机器人数据集潜力大，但真实人类数据采集费时费力。本文主张：与其只做 sim-to-real 迁移，不如在训练时直接把「仿真」与「真实」数据集混合协同训练（co-training）。通过在机械臂与人形系统、多样操作任务上的系统实验，作者证明：即便仿真与真实数据有明显差异，仿真数据也能把真实任务表现平均提升 38%。这给出一个简单有效的视觉操作训练配方。"
---

# Sim-and-Real Co-Training

**Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

大规模真实机器人数据集潜力大，但真实人类数据采集费时费力。本文主张：与其只做 sim-to-real 迁移，不如在训练时直接把「仿真」与「真实」数据集混合协同训练（co-training）。通过在机械臂与人形系统、多样操作任务上的系统实验，作者证明：即便仿真与真实数据有明显差异，仿真数据也能把真实任务表现平均提升 38%。这给出一个简单有效的视觉操作训练配方。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Co-Training | 协同训练，sim 与 real 数据混合训练 |
| Sim-to-Real | 仿真到真机迁移（被对比对象） |
| Vision-Based | 基于视觉的策略 |
| Domain Gap | 域差距（sim 与 real 差异） |
| Recipe | 配方，可复用的训练做法 |
| Generalist | 通才机器人模型 |

## 为什么重要

- **"混合训练 > 两段迁移"是反直觉但实用的洞见**：让模型同时见两域更稳；
- **仿真数据即便不完美也有用**，降低对昂贵真实数据的依赖；
- 对人形（真实采集更难）尤其有价值；
- 与 DreamGen、DexMimicGen 等"用仿真/合成数据扩规模"思路互补。

## 解决什么问题

真实数据采集贵，仿真数据多但有域差距： - 纯 **sim-to-real 迁移**常需精心对齐； - 想更**简单**地用上仿真数据提升真实表现。

论文要：一个**简单配方**——直接 **sim + real 协同训练**，看仿真数据能否稳定帮真实任务。

## 核心机制

1. **sim+real 协同训练配方**：训练时混合，而非两段迁移；
2. **系统实验（臂 + 人形）**：研究最优配方；
3. **+38% 真实表现**：即便域差异明显；
4. **简单可复用**：易嫁接到现有视觉操作流程。

方法拆解（深读笔记小节）：协同训练（sim + real 混合）；系统实验（臂 + 人形）；结论；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2503.24361> |
| 作者 | Abhiram Maddukuri、Zhenyu Jiang、Soroush Nasiriany、Ken Goldberg、Ajay Mandlekar、Linxi Fan、Yuke Zhu 等（NVIDIA / Berkeley / UT） |
| 发表 | 2025 年 3 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_sim-and-real-co-training.md](../../sources/papers/humanoid_pnb_sim-and-real-co-training.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation.html>
- 论文：<https://arxiv.org/abs/2503.24361>

## 推荐继续阅读

- [机器人论文阅读笔记：Sim-and-Real Co-Training](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation.html)
