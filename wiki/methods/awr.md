---
type: method
tags: [rl, off-policy, regression]
status: complete
updated: 2026-04-28
related:
  - ./reinforcement-learning.md
sources:
  - ../../sources/papers/awr.md
summary: "AWR 是一种简单的离策 RL 算法，通过优势加权回归将策略优化转化为监督学习。"
---

# AWR: 优势加权回归

**Advantage-Weighted Regression (AWR)** 旨在解决传统 RL 算法（如 PPO/SAC）在处理静态离线数据或大规模并行训练时的复杂性。

## 核心公式
AWR 通过最大化以下目标来更新策略：
280870\mathbb{E}_{s,a \sim \mathcal{D}} \left[ \exp\left( \frac{1}{\beta} A(s,a) \right) \log \pi(a|s) \right]280870
其中 (s,a)$ 是由基准值网络估计的优势。

## 优势
- **稳定性**：将 RL 转化为带权重的监督回归任务，避免了复杂的策略梯度更新。
- **离策支持**：天然支持从非当前策略产生的数据中学习。

## 参考来源
- [sources/papers/awr.md](../../sources/papers/awr.md)
