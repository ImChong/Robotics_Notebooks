# Muon Optimizer — 论文与理论文献摘录

> 来源归档（ingest）

- **标题：** Muon Optimizer — 原始博客、规模化验证、理论与变体论文索引
- **类型：** paper + blog 索引
- **入库日期：** 2026-07-14
- **一句话说明：** 汇总 Muon 从博客提出到 LLM 规模化训练、谱范数理论与后续变体的主要一手文献，支撑方法页与选型对比。

## 核心文献摘录

### 1) Muon: An optimizer for hidden layers in neural networks（Keller Jordan et al., 2024-12）
- **形式：** 技术博客（**非 arXiv**）
- **链接：** https://kellerjordan.github.io/posts/muon/
- **代码：** https://github.com/KellerJordan/Muon
- **核心贡献：** 提出 Muon = Momentum + Newton–Schulz 正交化；限定隐藏层 2D 权重；其余参数用 AdamW。
- **对 wiki 的映射：**
  - [Muon](../../wiki/methods/muon.md)
  - [sources/blogs/muon_keller_jordan_2024.md](../blogs/muon_keller_jordan_2024.md)

### 2) Muon is Scalable for LLM Training（Moonshot AI, arXiv:2502.16982）
- **链接：** https://arxiv.org/abs/2502.16982
- **核心贡献：** 证明 Muon 可扩展至 **Billion-scale LLM**；两大工程技巧：**Weight Decay** + **Per-parameter Update Scale**；用 Muon 训练 **Moonlight 3B/16B MoE**（5.7T tokens）；Scaling law 显示相对 AdamW 约 **2× 计算效率**；开源分布式 Muon 实现。
- **对 wiki 的映射：**
  - [Muon](../../wiki/methods/muon.md)
  - [paper-muon-scalable-llm-training](../../wiki/entities/paper-muon-scalable-llm-training.md)

### 3) Muon Optimizes Under Spectral Norm Constraints（Chen, Li, Liu, arXiv:2506.15054）
- **链接：** https://arxiv.org/abs/2506.15054
- **核心贡献：** 将 Muon 置于 Lion-$\mathcal{K}$ 族；证明 Muon（解耦 WD）隐式在 **谱范数（spectral norm）** 约束下优化，解释其隐式正则化。
- **对 wiki 的映射：**
  - [Muon](../../wiki/methods/muon.md)

### 4) Denoise First, Orthogonalize Later（arXiv:2606.03899）
- **链接：** https://arxiv.org/abs/2606.03899
- **核心贡献：** Momentum 在 Muon 中充当 **谱滤波器**：先去梯度噪声、再正交化；**Momentum Before Orthogonalization** 优于反序或去掉 momentum。
- **对 wiki 的映射：**
  - [Muon](../../wiki/methods/muon.md)

### 5) The Newton-Muon Optimizer（Du & Su, arXiv:2604.01472）
- **链接：** https://arxiv.org/abs/2604.01472
- **核心贡献：** 从 **Newton 方法** 在矩阵空间的 surrogate 推导 Muon；标准 Muon 可视为忽略输入二阶矩右预条件的隐式 Newton；提出 **Newton-Muon** 作为自然推广（Modded-NanoGPT 上约 **6% 更少步数**）。
- **对 wiki 的映射：**
  - [Muon](../../wiki/methods/muon.md)

### 6) 后续变体（索引，待单篇深读）

| 名称 | 主要贡献 | 备注 |
|------|----------|------|
| MONA | Muon + Nesterov Acceleration | 加速变体 |
| MuonBP | Block Orthogonalization | 多 GPU 吞吐 |
| MuonEq | Orthogonalization 前 Row/Column Balance | 平衡预处理 |
| MiMuon | Muon + SGD | 泛化增强 |
| Newton-Muon | Newton 理论推导 | 见 arXiv:2604.01472 |
| OLion | 谱范数 + $\ell_\infty$ 交 | arXiv:2602.01105 |

## 推荐阅读顺序（理解难度递增）

1. [Muon 博客](../blogs/muon_keller_jordan_2024.md) — 算法动机与设计
2. arXiv:2502.16982 — LLM 规模化与工程细节
3. arXiv:2604.01472 — Newton 视角
4. arXiv:2506.15054 — 谱范数隐式约束
5. arXiv:2606.03899 — Momentum 为何在正交化之前

## 对 wiki 的映射

- [Muon（方法页）](../../wiki/methods/muon.md)
- [Deep Learning Optimizers 对比](../../wiki/comparisons/deep-learning-optimizers.md)
