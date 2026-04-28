---
type: method
tags: [imitation-learning, tracking, rl, xbpeng]
status: complete
updated: 2026-04-28
related:
  - ../entities/protomotions.md
  - ./amp-reward.md
  - ../entities/mimickit.md
sources:
  - ../../sources/papers/deepmimic.md
summary: "DeepMimic 是物理角色动画的基石工作，通过精确的轨迹跟踪奖励实现复杂的运动模仿。"
---

# DeepMimic: 示例引导的技能学习

**DeepMimic** 是第一个能够让物理仿真智能体高保重模仿后空翻、武术等高难度动作的深度 RL 算法。

## 核心：显式跟踪 (Explicit Tracking)
不同于后来的 AMP 靠判别器“悟”，DeepMimic 靠“盯”。它要求机器人的每一个关节在每一时刻都要尽可能贴合参考轨迹。

## 主要技术路线
| 模块 | 方案 | 作用 |
|------|-----|------|
| **奖励函数** | [奖励函数设计](../concepts/reward-design.md) Multi-term Reward | 综合位置、速度、末端位姿和质心偏差 |
| **初始化** | RSI (Reference State Initialization) | 在轨迹的任意点开始训练，增加样本多样性 |
| **早期终止** | Early Exit | 如果跌倒或偏离过大则重置，提高训练效率 |

## 关联页面
- [[protomotions]] — 提供大规模并行训练支持。
- [[amp-reward]] — 后续的“无奖励设计”版本。
- [[mimickit]] — 现代化的实现框架。

## 参考来源
- [sources/papers/deepmimic.md](../../sources/papers/deepmimic.md)
