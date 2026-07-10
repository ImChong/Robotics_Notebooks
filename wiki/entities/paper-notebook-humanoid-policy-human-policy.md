---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2503.13441"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoid-policy.md
summary: "用多样数据训练人形操作策略能增强鲁棒与跨任务/跨平台泛化。但只从机器人演示学很费力——需昂贵遥操作、难规模化。本文研究一种更可扩展的数据源：第一视角人类演示，作为机器人学习的跨本体训练数据。工作从数据采集与建模两方面弥合具身差距：① 引入与人形任务对齐的第一视角人类数据集 PH2D；② 提出 Human Action Transformer（HAT），统一人类与人形的状态-动作表示，并具备可微重定向能力；再与机器人数据协同训练。相比只用机器人数据，人类数据显著提升泛化与鲁棒，且大幅提高数据采集效率。"
---

# Humanoid Policy ~ Human Policy

**Humanoid Policy ~ Human Policy** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

用多样数据训练人形操作策略能增强鲁棒与跨任务/跨平台泛化。但只从机器人演示学很费力——需昂贵遥操作、难规模化。本文研究一种更可扩展的数据源：第一视角人类演示，作为机器人学习的跨本体训练数据。工作从数据采集与建模两方面弥合具身差距：① 引入与人形任务对齐的第一视角人类数据集 PH2D；② 提出 Human Action Transformer（HAT），统一人类与人形的状态-动作表示，并具备可微重定向能力；再与机器人数据协同训练。相比只用机器人数据，人类数据显著提升泛化与鲁棒，且大幅提高数据采集效率。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Cross-Embodiment | 跨本体（人↔人形） |
| PH2D | 与人形任务对齐的第一视角人类数据集 |
| HAT | Human Action Transformer |
| Unified State-Action | 统一状态-动作表示 |
| Differentiable Retargeting | 可微重定向 |
| Co-training | 协同训练（人类 + 机器人数据） |

## 为什么重要

- **"人形策略≈人类策略"是有力的跨本体假设**：统一表示让人类数据直接可用；
- **可微重定向**把"人→人形"做成可学模块，优于手工重定向；
- **协同训练**兼得人类规模与机器人对齐；
- 与 H-RDT、Being-H0、In-N-On 共同构成"人类数据驱动人形操作"的方法簇（作者群高度重叠）。

## 解决什么问题

只用机器人演示训练人形操作**费力难扩展**： - 遥操作贵； - 想用**第一视角人类数据**，但有**具身差距**（状态/动作空间不同）。

论文要：用**第一视角人类演示**作跨本体数据，弥合具身差距、提升人形操作。

## 核心机制

1. **第一视角人类数据作跨本体训练源**：可扩展、采集高效；
2. **PH2D 数据集**：与人形任务对齐；
3. **HAT 统一状态-动作 + 可微重定向**：端到端弥合具身差距；
4. **协同训练显著增益**：泛化与鲁棒提升。

方法拆解（深读笔记小节）：PH2D：与人形任务对齐的人类数据集；HAT：统一状态-动作 + 可微重定向；与机器人数据协同训练；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoid_Policy__Human_Policy/Humanoid_Policy__Human_Policy.html> |
| arXiv | <https://arxiv.org/abs/2503.13441> |
| 作者 | Ri-Zhao Qiu、Shiqi Yang、Xuxin Cheng、Tairan He、Ryan Hoque、Guanya Shi、Xiaolong Wang 等（UCSD / CMU 等） |
| 发表 | 2025 年 3 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoid-policy.md](../../sources/papers/humanoid_pnb_humanoid-policy.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoid_Policy__Human_Policy/Humanoid_Policy__Human_Policy.html>
- 论文：<https://arxiv.org/abs/2503.13441>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoid Policy ~ Human Policy](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoid_Policy__Human_Policy/Humanoid_Policy__Human_Policy.html)
