---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.12705"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dreamgen.md
summary: "DreamGen 是一个简单而高效的四阶段流水线，通过神经轨迹（neural trajectories）——由视频世界模型生成的合成机器人数据——训练能跨行为、跨环境泛化的机器人策略。流程：① 用图像到视频生成模型；② 把模型适配到目标机器人本体，生成逼真合成视频；③ 用潜动作模型（latent action model）或逆动力学模型（inverse-dynamics model）从视频中恢复伪动作序列；④ 用这些数据训练策略。还提出 DreamGen Bench 评测视频生成质量。实验中，仅用单一取放任务、单一环境的遥操作数据，DreamGen 就让人形在已见与未见环境完成 22 种新行为，展示强行为与环境泛化，为超越人工采集地扩展机器人学习开辟新路径。"
---

# DreamGen

**DreamGen: Unlocking Generalization in Robot Learning through Video World Models** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

DreamGen 是一个简单而高效的四阶段流水线，通过神经轨迹（neural trajectories）——由视频世界模型生成的合成机器人数据——训练能跨行为、跨环境泛化的机器人策略。流程：① 用图像到视频生成模型；② 把模型适配到目标机器人本体，生成逼真合成视频；③ 用潜动作模型（latent action model）或逆动力学模型（inverse-dynamics model）从视频中恢复伪动作序列；④ 用这些数据训练策略。还提出 DreamGen Bench 评测视频生成质量。实验中，仅用单一取放任务、单一环境的遥操作数据，DreamGen 就让人形在已见与未见环境完成 22 种新行为，展示强行为与环境泛化，为超越人工采集地扩展机器人学习开辟新路径。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Video World Model | 视频世界模型，生成未来视频 |
| Neural Trajectory | 神经轨迹，生成的合成机器人数据 |
| Latent Action Model | 潜动作模型，从视频推动作 |
| Inverse-Dynamics | 逆动力学模型，从状态变化推动作 |
| Pseudo-action | 伪动作，恢复出的动作标签 |
| DreamGen Bench | 本文视频生成评测基准 |

## 为什么重要

- **"视频世界模型 + 伪动作恢复"是扩数据的新范式**：把生成视频变成可训练的动作数据；
- **跨行为/跨环境泛化**直击机器人学习的核心痛点；
- **最小真实数据 → 大量合成行为**性价比极高；
- 与 Humanoid World Models、DexMimicGen 等生成/世界模型工作呼应（同 NVIDIA 系）。

## 解决什么问题

机器人策略泛化差、数据采集贵： - 真实采集**少行为、少环境**； - 想**跨行为/跨环境泛化**，但缺数据。

DreamGen 要：用**视频世界模型**生成**带动作标签**的合成数据，**最小真实采集**就解锁泛化。

## 核心机制

1. **视频世界模型生成神经轨迹**：合成机器人数据训练策略；
2. **四阶段流水线**：生成→适配本体→恢复伪动作→训练；
3. **DreamGen Bench**：评测视频生成质量；
4. **强泛化**：单任务单环境数据 → 人形 22 种新行为。

方法拆解（深读笔记小节）：四阶段流水线；神经轨迹 = 合成机器人数据；DreamGen Bench；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models.html> |
| arXiv | <https://arxiv.org/abs/2505.12705> |
| 作者 | Joel Jang、Seonghyeon Ye、Ajay Mandlekar、Yuke Zhu、Linxi Fan、Dieter Fox、Jan Kautz 等（NVIDIA） |
| 发表 | 2025 年 5 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dreamgen.md](../../sources/papers/humanoid_pnb_dreamgen.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models.html>
- 论文：<https://arxiv.org/abs/2505.12705>

## 推荐继续阅读

- [机器人论文阅读笔记：DreamGen](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models.html)
