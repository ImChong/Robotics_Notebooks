---
type: method
tags: [rl, off-policy, regression, xbpeng]
status: complete
updated: 2026-04-28
related:
  - ../entities/mimickit.md
  - ../entities/protomotions.md
  - ./reinforcement-learning.md
  - ./policy-optimization.md
sources:
  - ../../sources/papers/awr.md
summary: "AWR 是一种简单的离策 RL 算法，通过优势加权回归将策略优化转化为监督学习，避开了策略梯度的不稳定性。"
---

# AWR: 优势加权回归

**Advantage-Weighted Regression (AWR)** 提供了一种不同于 PPO 的 RL 路径，它完全基于回归分析。

## 核心思想
AWR 不直接计算策略梯度，而是通过对优势函数进行指数加权来拟合策略。它试图在数据集中寻找那些表现优于平均水平（即优势为正）的动作，并增加它们的出现概率。

## 主要技术路线
| 步骤 | 关键公式 / 技术 | 说明 |
|------|---------------|------|
| **价值估计** | (s) \leftarrow$ [Bellman 方程](../formalizations/bellman-equation.md) TD-Error | 训练一个基准值网络来评估状态好坏 |
| **优势计算** | (s,a) = R - V(s)$ | 计算采集到的动作比平均水平好多少 |
| **策略更新** | Weighted Cross-Entropy | 权重为 $\exp(A/\beta)$ 的监督学习更新 |

## 优势
- **极度简单**：代码实现容易，不需要复杂的信赖域计算。
- **离策兼容**：可以无缝地处理来自人类演示或历史缓冲区的数据。

## 关联页面
- [[protomotions]] — 提供大规模并行训练支持。
- [[reinforcement-learning]] — RL 基础。
- [[policy-optimization]] — 算法对比。

## 参考来源
- [sources/papers/awr.md](../../sources/papers/awr.md)
