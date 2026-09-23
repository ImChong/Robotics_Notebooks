---
type: entity
tags: [paper, scaling-laws, nlp, foundation-model, openai, machine-learning]
status: complete
updated: 2026-09-21
arxiv: "2001.08361"
venue: "arXiv 2020"
related:
  - ../concepts/embodied-scaling-laws.md
  - ./light-o1.md
  - ./dyna-2.md
  - ../methods/egoscale.md
sources:
  - ../../sources/papers/kaplan_scaling_laws_arxiv_2001_08361.md
summary: "Kaplan et al.（OpenAI，arXiv:2001.08361）：LM 损失对模型规模、数据量、算力呈幂律；给出 compute-optimal 分配与过拟合规律——具身 Transfer Scaling 的方法论原典。"
---

# Scaling Laws for Neural Language Models

**Scaling Laws for Neural Language Models**（Kaplan et al.，[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)，OpenAI 2020）系统测量 **自回归语言模型** 的 cross-entropy 如何随 **参数量 N、数据集大小 D、训练算力 C** 变化，发现 **跨 7+ 数量级的幂律**，并推导 **固定算力下最优 N:D 分配**。

## 一句话定义

**用幂律 $L(N,D,C)$ 把「堆模型 / 堆数据 / 堆算力」从玄学变成可外推的工程曲线，并解释何时更大模型更 compute-efficient。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LM | Language Model | 自回归下一 token 预测模型 |
| CE | Cross-Entropy | 语言建模主损失 |
| FLOPs | Floating Point Operations | 训练算力度量 |
| SOTA | State of the Art | 当时最优基准 |
| RLHF | Reinforcement Learning from Human Feedback | 后续对齐阶段（非本文核心） |

## 为什么重要

- **Scaling 研究范式：** 后续 Chinchilla、GPT-4、具身 **EgoScale / Dyna-2 / Light-O1 Transfer Law** 均借用 **幂律拟合 + optimal allocation** 话语。
- **Light-O1 直接引用：** Tech Blog [10] 用 $L(D)=L_0+\alpha D^{-\eta}$ 拟合 **human action pretraining token 预算** 与适配后误差——协议对齐 Kaplan 式分析。
- **与 Bitter Lesson 互补：** 提供 **可量化** 的「scale helps」证据，而非仅原则性论述。

## 核心方法与发现

| 轴 | 规律（直觉） |
|----|--------------|
| **模型规模 N** | 损失随 N **幂律下降**；宽/深等细节在宽范围内 **次要** |
| **数据 D** | 损失随 D **幂律下降**；数据不足 → **可预测过拟合** |
| **算力 C** | 损失随 C **幂律下降**；存在 **compute-optimal** 模型尺寸 |
| **分配** | 固定 C 时，**更大模型 + 更少步数** 常优于小模型长跑 |

## 实验与评测

- 实验横跨 **多个数量级** 的模型与数据（OpenAI 内部 LM 训练）。
- 报告 **test loss** 为主指标；给出 **early stopping / 过拟合** 与 D 的关系。
- **不覆盖** 下游 task fine-tune 或 RLHF——仅 **pretraining cross-entropy** 缩放。

## 结论

**Kaplan scaling laws 把 LM 预训练变成「可预算、可外推、可分配算力」的工程问题。**

1. **幂律在宽范围内稳定** — 使「加 10× 数据能换多少 loss」可谈。
2. **架构细节次要** — 在固定范式内，**N 与 D** 是第一性变量（与后来 DiT/VLA 讨论同源）。
3. **Compute-optimal 训练** — 指导 Chinchilla 等「大模型 + 足够数据」路线。
4. **具身迁移读法：** Light-O1 把 **D** 换成 **multimodal human-action tokens**，检验 **跨本体适配误差** 是否仍幂律——**类比而非恒等**。
5. **边界：** 原文是 **text CE**；机器人 **success rate** 可能非平滑幂律，外推需谨慎。

## 与其他工作对比

| 维度 | Kaplan LM (2020) | Chinchilla (2022) | Light-O1 Transfer (2026) |
|------|------------------|-------------------|--------------------------|
| 对象 | Text LM CE | 修正 compute-optimal 数据量 | Human action prior → 多本体适配 |
| 指标 | Cross-entropy | CE + downstream | Next-action loss + MPJPE |
| 开源 | 无统一官方仓 | 无 | Preview 部分开源 |

## 关联页面

- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
- [Light-O1](./light-o1.md)
- [Dyna-2](./dyna-2.md)

## 参考来源

- [kaplan_scaling_laws_arxiv_2001_08361.md](../../sources/papers/kaplan_scaling_laws_arxiv_2001_08361.md)
- 论文：<https://arxiv.org/abs/2001.08361>

## 推荐继续阅读

- [Embodied Scaling Laws 概念页](../concepts/embodied-scaling-laws.md)
