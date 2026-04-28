---
type: method
tags: [imitation-learning, tracking, rl, xbpeng]
status: complete
updated: 2026-04-28
related:
  - ./amp-reward.md
  - ../entities/mimickit.md
sources:
  - ../../sources/papers/deepmimic.md
summary: "DeepMimic 是基于物理的特征技能学习的基石工作，通过 RL 实现对参考运动轨迹的精确跟踪。"
---

# DeepMimic: 示例引导的技能学习

**DeepMimic** 证明了复杂的物理技能（如后空翻、踢腿、舞蹈）可以通过强化学习对单一或多个参考剪辑（Reference Clips）进行模仿来获得。

## 核心机制：跟踪奖励 (Tracking Reward)

DeepMimic 使用一个显式的复合奖励函数来引导策略：
280870r_t = w_p r_p + w_v r_v + w_e r_e + w_c r_c280870
- **Pose ($)**: 关节角度的 MSE。
- **Velocity ($)**: 关节速度的 MSE。
- **End-Effector ($)**: 末端执行器位置的 MSE。
- **Center of Mass ($)**: 质心位置偏差。

## 局限性
- **阶段依赖**：需要显式的相位变量 (Phase Variable) 来同步参考动作。
- **奖励工程**：需要为不同动作精细调整各项权重。

## 参考来源
- [sources/papers/deepmimic.md](../../sources/papers/deepmimic.md)
