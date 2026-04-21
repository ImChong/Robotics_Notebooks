---
type: entity
tags: [robot, quadruped, hardware, platform, eth]
status: complete
updated: 2026-04-20
related:
  - ./humanoid-robot.md
  - ../tasks/locomotion.md
  - ../entities/legged-gym.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/locomotion_rl.md
summary: "ANYmal 是由 ETH Zurich 和 ANYbotics 开发的高性能四足机器人平台，是机器人强化学习（RMA、ANYmal C）研究的标志性载体。"
---

# ANYmal 四足机器人

**ANYmal** 是由苏黎世联邦理工学院（ETH Zurich）的机器人系统实验室（RSL）研发，并由衍生公司 ANYbotics 商业化的高性能四足机器人。它在学术界（特别是足式机器人强化学习领域）和工业巡检领域具有极高的影响力。

## 核心特征

1. **高度集成与鲁棒性**：ANYmal 设计用于极端的工业环境（如矿井、海上油气平台）。其外壳具备高等级的防水防尘能力（IP67）。
2. **力控执行器**：ANYmal 使用专门开发的 **串联弹性执行器 (SEA)**。相比于刚性驱动，SEA 提供了天然的碰撞缓冲和更精确的力矩反馈，非常适合处理不确定地形的接触。
3. **顶尖的算法载体**：它是许多里程碑式论文的实验平台，包括：
   - **RMA (Rapid Motor Adaptation)**：实现了极强的越野与地形自适应能力。
   - **Learning Agile Locomotion**：展示了四足机器人如何通过 RL 学会类似动物的翻越动作。
   - **ANYmal C/D 系列**：集成了先进的感知（LiDAR, 深度相机）和全自主导航能力。

## 与 Unitree 的区别

| 维度 | ANYmal | Unitree (A1/Go1/B2) |
|------|--------|---------------------|
| **定位** | 高端巡检 / 顶尖科研 | 极高性价比 / 普及科研 |
| **驱动方式** | 串联弹性执行器 (SEA) | 准直接驱动 (QDD) |
| **重量/负载** | 较重，负载能力强 | 较轻，运动速度快 |
| **软件生态** | 开源 raisimGym / OCS2 | 开源 Unitree SDK / legged_gym |

## 对机器人研究的贡献

ANYmal 系列是“**数据驱动足式控制**”路线的先驱。ETH RSL 团队通过 ANYmal 证明了在仿真（Raisim/Isaac Gym）中大规模训练的策略，通过合理的域随机化和状态估计，可以无缝迁移到极端复杂的真实户外地形。

## 关联页面
- [人形机器人](./humanoid-robot.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Legged Gym](../entities/legged-gym.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源
- Hutter, M., et al. (2016). *ANYmal - a highly mobile and dynamic quadrupedal robot*.
- Lee, J., et al. (2020). *Learning quadrupedal locomotion over challenging terrain*.
- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md)
