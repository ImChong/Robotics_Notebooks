---
type: entity
tags: [humanoid, reinforcement-learning, motion-imitation, sim2real, character-animation, sfu, nvidia, berkeley]
status: complete
updated: 2026-05-21
related:
  - ./paper-deeprl-locomotion-action-space-sca2017.md
  - ./mimickit.md
  - ./protomotions.md
  - ../methods/deepmimic.md
  - ../methods/amp-reward.md
  - ../methods/ase.md
  - ../concepts/sim2real.md
  - ./tairan-he.md
  - ./zhengyi-luo.md
sources:
  - ../../sources/sites/xue-bin-peng.md
  - ../../sources/papers/deeprl_locomotion_action_space_sca2017.md
summary: "彭学斌（Xue Bin Peng）为 SFU 助理教授兼 NVIDIA 研究科学家，博士师从 Levine / Abbeel；以 DeepMimic、AMP、ASE、动力学随机化 Sim2Real 等工作定义了物理角色与腿式机器人 RL 运动控制的一条主干研究线，并通过 MimicKit 统一开源实现。"
---

# Xue Bin Peng（彭学斌）

## 一句话定义

**Xue Bin Peng** 是 **物理仿真角色与腿式机器人强化学习运动控制** 领域的核心研究者之一：将 **示例引导 RL（DeepMimic）**、**对抗式运动先验（AMP）** 与 **可复用技能潜空间（ASE）** 等方法体系化，并推动 **动力学随机化 Sim2Real** 在 locomotion 上的早期标杆实践；工程上通过 [MimicKit](https://github.com/xbpeng/MimicKit) 与 [ProtoMotions](./protomotions.md) 等框架把论文算法收敛到可维护代码栈。

## 为什么重要

- **仓库内多条主线的「原作者索引」**：[DeepMimic](../methods/deepmimic.md)、[AMP](../methods/amp-reward.md)、[ASE](../methods/ase.md)、[Sim2Real](../concepts/sim2real.md) 与 [MimicKit](./mimickit.md) 的叙事均直接依赖其论文与主页给出的 **项目页 / PDF**。
- **从图形学到机器人控制的桥**：同一套 RL + 物理仿真方法论既覆盖 **角色动画**（SIGGRAPH 系）也覆盖 **Cassie 双足、四足与 sim-to-real**（RSS / ICRA / IJRR），便于理解「动画社区与机器人社区共享的 motion imitation 语言」。

## 核心研究脉络（归纳）

1. **显式模仿与跟踪**：DeepMimic 一类工作强调 **参考运动 + 物理约束下的策略学习**，是后续大量 humanoid tracking 工作的参照系。
2. **对抗式先验与表征**：AMP / ASE / ADD 等沿 **判别器或潜变量** 注入运动统计，降低手工奖励 shaping 成本，并与分层控制结合。
3. **Sim2Real 与系统论文**：ICRA 2018 的动力学随机化论文常被作为 **域随机化 loco 迁移** 的入门引用；后续工作延续到多项目页（以主页列表为准）。
4. **动作空间与 locomotion 前导**：[DeepRL 动作空间对比（SCA 2017）](./paper-deeprl-locomotion-action-space-sca2017.md) 在平面角色上系统比较 **扭矩 / PD 目标角** 等接口，为后续 Cassie 与 sim2real 中的 **「低维目标 + 内环」** 选型提供早期实证。
5. **开源整合**：MimicKit 将多种算法置于统一训练循环，降低复现与对照实验门槛。

## 常见误区或局限

- **主页年份表 ≠ 引用格式**：会议全名、页码与 arXiv 版本应以 **最终 PDF** 或 DBLP 为准。
- **角色 vs 机器人**：部分论文以 **物理角色** 为实验体，结论迁移到硬件人形时需结合接触模型、执行器与观测差异单独评估。

## 关联页面

- [MimicKit](./mimickit.md)
- [ProtoMotions](./protomotions.md)
- [DeepMimic](../methods/deepmimic.md)
- [Sim2Real](../concepts/sim2real.md)
- [Tairan He（何泰然）](./tairan-he.md)
- [Zhengyi Luo（罗正宜）](./zhengyi-luo.md)
- [DeepRL 动作空间对比（SCA 2017）](./paper-deeprl-locomotion-action-space-sca2017.md)
- [Character Animation vs Robotics](../concepts/character-animation-vs-robotics.md) — 其图形学起源方法被搬到真实人形 RL 的张力讨论

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| ADD | Adversarial Differential Discriminator | 差分判别、减少碎片化 reward 的 AMP 演进 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |

## 参考来源

- [Xue Bin Peng 个人主页原始资料](../../sources/sites/xue-bin-peng.md)
- [DeepRL 动作空间 SCA 2017 原始资料](../../sources/papers/deeprl_locomotion_action_space_sca2017.md)

## 推荐继续阅读

- [MimicKit（GitHub）](https://github.com/xbpeng/MimicKit)
- [DeepMimic 项目页](https://xbpeng.github.io/projects/DeepMimic/index.html)
