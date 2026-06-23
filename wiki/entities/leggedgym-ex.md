---

type: entity
tags: [repo, quadruped, legged-gym, amp, deepmimic, multi-simulator, eth, booster]
status: complete
updated: 2026-06-08
summary: "lupinjia/LeggedGym-Ex 扩展 legged_gym 至多仿真器并集成 AMP、DeepMimic 等模仿任务，覆盖 Go2、K1、TRON1 等机型。"
related:
  - ./legged-gym.md
  - ../methods/amp-reward.md
  - ../methods/deepmimic.md
  - ./isaac-lab.md
sources:
  - ../../sources/repos/leggedgym_ex.md
---

# LeggedGym-Ex

**LeggedGym-Ex**（<https://github.com/lupinjia/LeggedGym-Ex>）在 [legged_gym](./legged-gym.md) 范式上扩展 **多仿真后端**（Isaac Gym、Isaac Sim、Genesis）与 **多种模仿学习算法**（AMP、DeepMimic、DreamWaQ 等），面向 **Go2、Booster K1、TRON1** 等机型提供统一训练入口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AMP | Adversarial Motion Prior | 风格先验 + 任务奖励 |
| RL | Reinforcement Learning | legged_gym 风格 PPO 训练 |
| legged_gym | Legged Gym | ETH RSL 足式 RL 框架 |
| Sim2Real | Simulation to Real | 域随机化与 actuator 建模 |

## 为什么重要

- **四足模仿学习合集**：把「参考运动 → 策略」多种算法挂在同一 legged_gym 配置体系下，便于对照 AMP vs DeepMimic。
- **仿真器迁移试验床**：同一 env 配置可切换 IsaacGym / IsaacSim / Genesis，适合验证重定向参考在不同物理后端下的可跟踪性。

## 关联页面

- [legged_gym](./legged-gym.md)
- [AMP_for_hardware](./amp-for-hardware.md)
- [DeepMimic](../methods/deepmimic.md)
- [Isaac Lab](./isaac-lab.md)

## 参考来源

- [LeggedGym-Ex 仓库归档](../../sources/repos/leggedgym_ex.md)

## 推荐继续阅读

- GitHub：<https://github.com/lupinjia/LeggedGym-Ex>
