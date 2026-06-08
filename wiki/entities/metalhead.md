---
type: entity
tags: [repo, quadruped, amp, unitree-a1, legged-gym]
status: complete
updated: 2026-06-08
summary: "inspirai/MetalHead 在 Unitree A1 上用 AMP 实现自然行走、跳跃与跌倒恢复，基于 AMP_for_hardware + Isaac Gym + legged_gym。"
related:
  - ./amp-for-hardware.md
  - ./legged-gym.md
  - ../methods/amp-reward.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/metalhead.md
---

# MetalHead

**MetalHead**（<https://github.com/inspirai/MetalHead>）在 **Unitree A1** 四足上实现 **walk / run / jump / recovery** 等自然运动，核心算法为 **AMP**，代码基于 [AMP_for_hardware](./amp-for-hardware.md) 与 [legged_gym](./legged-gym.md)，仿真后端为 **Isaac Gym**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AMP | Adversarial Motion Prior | 从参考运动库学习风格约束 |
| A1 | Unitree A1 Quadruped | 宇树四足科研平台 |
| RL | Reinforcement Learning | PPO + 风格判别器 |
| Sim2Real | Simulation to Real | 论文/项目强调真机可迁移调参 |

## 为什么重要

- **四足 AMP 工程样板**：展示如何把动物/参考运动风格迁移到 A1 并处理高动态技能（跳跃、恢复）的超参与奖励权衡。
- 与纯几何重定向不同，重点在 **仿真内可跟踪 + 判别器风格匹配**。

## 关联页面

- [AMP_for_hardware](./amp-for-hardware.md)
- [legged_gym](./legged-gym.md)
- [AMP 奖励设计](../methods/amp-reward.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [MetalHead 仓库归档](../../sources/repos/metalhead.md)

## 推荐继续阅读

- GitHub：<https://github.com/inspirai/MetalHead>
