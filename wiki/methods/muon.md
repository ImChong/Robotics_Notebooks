---
type: method
tags: [deep-learning, optimization, muon, orthogonalization, llm-training, transformer, moonshot]
status: complete
updated: 2026-07-14
summary: "Muon 对隐藏层 2D 权重先做 SGD-momentum，再用 Newton–Schulz 迭代近似正交化更新方向；原始提出为博客+代码，Moonshot 在 arXiv:2502.16982 证明其可扩展至 Billion-scale LLM 并约 2× 计算效率。"
related:
  - ./adamw.md
  - ./sgd-momentum.md
  - ./lion.md
  - ../comparisons/deep-learning-optimizers.md
  - ../entities/paper-muon-scalable-llm-training.md
  - ../entities/karpathy-autoresearch.md
  - ../concepts/transformer.md
  - ../concepts/deep-learning-foundations.md
sources:
  - ../../sources/blogs/muon_keller_jordan_2024.md
  - ../../sources/papers/muon_optimizer_primary_refs.md
  - ../../sources/repos/kellerjordan-muon.md
---

# Muon（MomentUm Orthogonalized by Newton–Schulz）

**Muon** 是面向神经网络 **隐藏层 2D 权重矩阵** 的优化器：对 SGD-momentum 产生的更新矩阵 $G$，用 **Newton–Schulz 迭代**（`newtonschulz5`）做**近似正交化**后再施加到参数。标量/向量参数、输入输出层仍用 [AdamW](./adamw.md)。Muon **最初由 Keller Jordan 等技术博客提出（2024-12）**，而非 arXiv 论文；规模化验证见 Moonshot AI 的 [Muon is Scalable for LLM Training](../entities/paper-muon-scalable-llm-training.md)（arXiv:2502.16982），在 Kimi / Moonlight 训练中广泛使用。

## 一句话定义

> 隐藏层矩阵更新先积动量、再正交化——用 Newton–Schulz 把梯度方向拉成近似正交矩阵，放大稀有学习方向，在 LLM 预训练中可比 AdamW 更省算力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Muon | MomentUm Orthogonalized by Newton–Schulz | Keller Jordan 等 2024 博客提出 |
| NS | Newton–Schulz iteration | 矩阵正交化的快速迭代，替代完整 SVD |
| SVD | Singular Value Decomposition | 奇异值分解；Muon 用 NS 近似 $UV^\top$ |
| WD | Weight Decay | 解耦权重衰减；LLM 规模化训练的关键配套 |
| LLM | Large Language Model | Muon 在 Billion-scale 预训练上验证 |
| MoE | Mixture of Experts | Moonlight 16B 采用 MoE 架构 |
| AdamW | Adam with decoupled Weight decay | 与 Muon 混用的标准优化器（非矩阵参数） |

## 为什么重要

- **首发形态特殊：** 社区引用与 GitHub Citation 指向 [Keller Jordan 博客](https://kellerjordan.github.io/posts/muon/)，说明 Muon 是 **博客 + 开源实现** 驱动的优化器，而非传统论文首发路线。
- **LLM 训练新选项：** Moonshot 在 arXiv:2502.16982 用 scaling law 证明相对 [AdamW](./adamw.md) 约 **2× 计算效率**；Moonlight 3B/16B MoE 成为公开标杆。
- **Speedrun 生态：** NanoGPT / Modded-NanoGPT、[karpathy/autoresearch](../entities/karpathy-autoresearch.md) 等将 Muon+AdamW 作为默认配方。
- **理论逐渐清晰：** 谱范数约束（arXiv:2506.15054）、Momentum 作谱滤波（arXiv:2606.03899）、Newton 推导（arXiv:2604.01472）解释「为何先 momentum 再正交化」。

## 主要技术路线

### 1. 更新流程

```mermaid
flowchart LR
  g[梯度 g_t] --> mom[SGD Momentum]
  mom --> G[更新矩阵 G]
  G --> ns[Newton–Schulz 正交化]
  ns --> upd[参数更新 ΔW]
  upd --> W[隐藏层权重 W]
```

对 2D 权重 $W$：

1. 累积 momentum 得矩阵 $G$（同 SGD-momentum）；
2. `newtonschulz5(G)`：bfloat16 下迭代约 5 步，近似 $\mathrm{Ortho}(G) \approx UV^\top$（$G=USV^\top$）；
3. $W \leftarrow W - \eta \cdot \mathrm{Ortho}(G)$（配合 WD 与 per-parameter scale）。

**4D 卷积权重** 可将后三维展平为矩阵后套用 Muon。

### 2. 参数分工（混合优化）

| 参数类型 | 推荐优化器 |
|----------|-----------|
| 隐藏层 2D 权重（Linear、Attention QKV 等） | **Muon** |
| 标量、偏置、LayerNorm、Embedding | **AdamW** |
| 输入/输出层 | **AdamW** |

[karpathy/autoresearch](../entities/karpathy-autoresearch.md) 的 `train.py` 即采用此 **Muon + AdamW** 双优化器模式。

### 3. LLM 规模化两技巧（arXiv:2502.16982）

- **解耦 Weight Decay：** 与 AdamW 类似，WD 须正确施加，否则大模型不稳定。
- **Per-parameter Update Scale：** 按参数形状精细调节有效步长，使 Muon **开箱可用**、减少大规模超参搜索。

### 4. 理论视角（简表）

| 论文 | 核心洞见 |
|------|----------|
| arXiv:2506.15054 | Muon 隐式在 **谱范数约束** 下优化，限制权重最大奇异值增长 |
| arXiv:2606.03899 | Momentum = **去噪谱滤波**；须 **先 momentum、后正交化** |
| arXiv:2604.01472 | Muon ≈ 矩阵空间 **Newton 方法** 近似；Newton-Muon 为显式推广 |

## 工程实践

| 场景 | 建议 |
|------|------|
| LLM 预训练（≥1B） | 优先读 arXiv:2502.16982；Moonshot 开源分布式实现 |
| NanoGPT / speedrun | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) + AdamW 混用 |
| 机器人 VLA 微调 | 仍以 AdamW 为主；Muon 证据集中在 **预训练** 而非小数据 IL |
| 从零复现 | 博客 `newtonschulz5` + 论文 WD/scale 细节缺一不可 |

### 与 AdamW 选型对照

| 维度 | AdamW | Muon |
|------|-------|------|
| 适用参数 | 全部 | 主要 **2D 隐藏层** |
| 二阶矩 | 有（per-parameter 自适应） | 无；靠正交化几何 |
| LLM scaling law | 基线 | 报告 ~2× 计算效率（Moonshot） |
| 实现成熟度 | PyTorch 内置 | 第三方 / 论文开源实现 |
| 理论完备度 | 成熟 | 快速发展中（2025–2026 多篇理论文） |

## 局限与风险

- **非万能替代 AdamW：** 仅针对矩阵块；混合优化增加实现与调试复杂度。
- **正交化开销：** Newton–Schulz 虽快于 SVD，仍比纯 AdamW 多算；需分布式实现（MuonBP 等）优化吞吐。
- **机器人栈证据少：** 当前强力结果在 **LLM 预训练** 与 speedrun；仿真 RL / 小 MLP 策略尚未大规模验证。
- **变体众多：** MONA、MuonEq、MiMuon、Newton-Muon 等并存，选型需对照具体任务与实现。

## 关联页面

- [AdamW](./adamw.md) · [SGD Momentum](./sgd-momentum.md) · [Lion](./lion.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)
- [Muon is Scalable for LLM Training（论文实体）](../entities/paper-muon-scalable-llm-training.md)
- [karpathy/autoresearch](../entities/karpathy-autoresearch.md)
- [Transformer](../concepts/transformer.md)

## 参考来源

- [Muon 原始博客（Keller Jordan, 2024-12）](../../sources/blogs/muon_keller_jordan_2024.md)
- [Muon Optimizer 论文与理论文献摘录](../../sources/papers/muon_optimizer_primary_refs.md)
- [KellerJordan/Muon 仓库](../../sources/repos/kellerjordan-muon.md)

## 推荐继续阅读

- [Keller Jordan, Muon blog](https://kellerjordan.github.io/posts/muon/)
- [Muon is Scalable for LLM Training (arXiv:2502.16982)](https://arxiv.org/abs/2502.16982)
- [Muon Optimizes Under Spectral Norm Constraints (arXiv:2506.15054)](https://arxiv.org/abs/2506.15054)
- [Denoise First, Orthogonalize Later (arXiv:2606.03899)](https://arxiv.org/abs/2606.03899)
- [The Newton-Muon Optimizer (arXiv:2604.01472)](https://arxiv.org/abs/2604.01472)
