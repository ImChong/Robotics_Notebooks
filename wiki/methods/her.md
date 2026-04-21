---
type: method
tags: [rl, sparse-reward, manipulation, goal-conditioned]
status: complete
updated: 2026-04-21
related:
  - ./reinforcement-learning.md
  - ../concepts/reward-design.md
sources:
  - ../../sources/papers/policy_optimization.md
summary: "事后经验回放（HER）通过将失败的轨迹重新标记为指向实际到达位置的成功轨迹，有效解决了操作任务中稀疏奖励难以训练的问题。"
---

# Hindsight Experience Replay (HER)

**HER** 是一种处理“稀疏奖励（Sparse Reward）”任务的绝佳技巧。在抓取或装配任务中，如果机器人只有在完美完成任务时才得到 1 分奖励，它很难通过随机探索学到任何东西。

## 主要技术路线

HER 的核心逻辑是“**错有错着**”：
1. 机器人试图把杯子放到 A 点，结果却放在了 B 点（失败）。
2. 在回放缓冲区（Replay Buffer）中，我们将这条轨迹复制一份。
3. **重新标记 (Relabeling)**：将目标从 A 改为 B，此时这条轨迹变成了一条完美的、通往 B 的“专家数据”。

## 带来的价值

- **加速收敛**：使智能体无论在什么状态下都能得到正向反馈。
- **目标条件化**：强制策略学习如何达到空间中的任意点，而不仅仅是特定的目标。

## 关联页面
- [Reinforcement Learning](./reinforcement-learning.md)
- [Reward Design 概念](../concepts/reward-design.md)

## 参考来源
- Andrychowicz, M., et al. (2017). *Hindsight Experience Replay*.
