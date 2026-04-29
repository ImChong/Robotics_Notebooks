---
type: entity
tags: [repo, amp, imitation-learning, mjlab, rsl-rl, unitree, humanoid, locomotion, recovery]
status: complete
updated: 2026-04-29
related:
  - ../methods/amp-reward.md
  - ./unitree-g1.md
  - ./legged-gym.md
  - ../methods/imitation-learning.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/amp_mjlab.md
summary: "AMP_mjlab 是基于 mjlab + rsl_rl 的 Unitree G1 统一 AMP 策略实现，用单一 actor-critic + 判别器同时覆盖 locomotion 与 fall-recovery，消除模式切换断裂。"
---

# AMP_mjlab (G1 统一 AMP 策略)

**AMP_mjlab** 是一个针对 **Unitree G1** 人形机器人的强化学习训练框架，建立在 **mjlab**（MuJoCo 并行仿真）和 **rsl_rl**（RSL PPO 训练库）之上，核心贡献在于用一个统一策略同时学习正常行走（locomotion）与跌倒恢复（fall-recovery）。

## 为什么重要？

传统做法需要维护独立的 "locomotion 策略" 和 "recovery 策略"，并在运行时检测跌倒再触发切换，模式切换时易产生动作撕裂（behavioral discontinuity）。AMP_mjlab 的统一策略消除了这个切换逻辑，同时 AMP 判别器保证了动作的自然风格。

## 核心架构

```
AMP_mjlab 统一训练
├── Actor-Critic 网络（单一策略）
│   ├── 速度跟踪奖励（locomotion 目标）
│   └── AMP 风格奖励（自然度）
├── AMP Discriminator
│   ├── 参考数据：WalkandRun clip
│   └── 参考数据：Recovery clip（跌倒恢复）
└── Delayed Termination
    └── 部分 env 在 reset 前给 recovery 窗口
```

**关键机制：Delayed Termination**——不立即 reset 跌倒的 env，而是给策略一个时间窗口尝试自主恢复，迫使策略学习爬起行为。

## 训练特征

- **规模**：4096 并行环境
- **收敛特征**：约 2 万步时 recovery 行为突然涌现，loss 指标跳变属正常
- **任务**：`Unitree-G1-AMP-Rough` / `Unitree-G1-AMP-Flat`
- **部署**：ONNX export，训练与推理 pipeline 一致

## 与 AMP 方法的关系

AMP_mjlab 是 [AMP & HumanX](../methods/amp-reward.md) 方法的一个具体实现，区别在于：

| 维度 | AMP 原论文 | AMP_mjlab |
|------|-----------|-----------|
| 硬件目标 | 角色控制（通用） | Unitree G1（人形） |
| 仿真框架 | IsaacGym / PhysX | mjlab（MuJoCo） |
| 任务范围 | 单一风格 | Locomotion + Recovery 统一 |
| 训练库 | 各异 | rsl_rl (PPO) |

## 关联页面

- [AMP & HumanX 方法](../methods/amp-reward.md) — AMP 方法本体
- [Unitree G1](./unitree-g1.md) — 目标硬件
- [legged_gym](./legged-gym.md) — 同为 rsl_rl + 并行仿真，基于 IsaacGym
- [Imitation Learning](../methods/imitation-learning.md) — AMP 属于模仿学习范式
- [Locomotion](../tasks/locomotion.md) — 任务方向

## 参考来源

- [sources/repos/amp_mjlab.md](../../sources/repos/amp_mjlab.md)
- [ccrpRepo/AMP_mjlab GitHub Repo](https://github.com/ccrpRepo/AMP_mjlab)
