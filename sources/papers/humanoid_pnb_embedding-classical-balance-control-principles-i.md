# Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2603.08619>
- **入库日期：** 2026-06-07
- **一句话说明：** 这篇论文的核心不是“再堆一个更大的网络”，而是把经典平衡控制里可解释的物理量（Capture Point、CoM 状态、质心动量）嵌入到 RL 训练中： - 在训练时作为特权 critic 输入和奖励塑形信号； - 在部署时 actor 仍然只依赖本体感觉，保证可落地。 结果是在一个统一策略中实现了从小扰动到大跌倒后的恢复行为链，并报告 93.4% 的恢复成功率。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-embedding-classical-balance-control-principles-i](../../wiki/entities/paper-notebook-embedding-classical-balance-control-principles-i.md).

## 对 wiki 的映射

- [paper-notebook-embedding-classical-balance-control-principles-i](../../wiki/entities/paper-notebook-embedding-classical-balance-control-principles-i.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery.html>
- 论文：<https://arxiv.org/abs/2603.08619>
