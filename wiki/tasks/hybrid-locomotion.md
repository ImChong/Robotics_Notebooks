---
type: task
tags: [locomotion, hybrid, wheel-legged, transformable, whole-body]
status: in-progress
summary: "Hybrid Locomotion 关注结合不同运动模式（如轮腿结合、双足/四足切换）的机器人系统及其控制挑战。"
updated: 2026-07-24
sources:
  - ../../sources/papers/x2n_transformable.md
  - ../../sources/papers/mujica_arxiv_2605_13058.md
  - ../../sources/papers/aware_arxiv_2604_23761.md
related:
  - ../concepts/wheel-legged-quadruped.md
  - ../entities/paper-mujica-wheel-legged-multi-skill.md
  - ../entities/paper-aware-wheeled-legged-reflexive-evasion.md
---

# Hybrid Locomotion (混合运动)

**混合运动（Hybrid Locomotion）**：机器人系统能通过不同的机械模式进行运动，最常见的是轮腿混合（Wheel-legged）以及形态可变（Transformable）的设计。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 核心挑战

1. **模态切换（Mode Transition / Transformation）：** 如何在不同结构形态或运动模式（例如，双足行走模式切换到带轮滑行模式）之间实现平稳和高效的转换。
2. **全身协调（Whole-body Coordination）：** 在特定的形态下（特别是包含上肢和双腿的完整平台），如何实现下肢移动与上肢操作的协调。
3. **混合控制框架（Hybrid Control Framework）：** 传统控制中需要分别针对不同模态设计控制器并处理切换逻辑，现代趋势逐渐使用强化学习（RL）来提供统一的策略。

## 技术路线

- **基于强化学习的统一控制（RL-based Unified Control）：** 采用 RL 学习统一的全身控制（Whole-body control）策略，通过端到端的方式处理变形态、轮式移动、步行以及操作等多任务并发的需求。

## 代表系统

### MUJICA（Unitree Go2-W 轮足四足）

- [MUJICA](../entities/paper-mujica-wheel-legged-multi-skill.md)（arXiv:2605.13058）在 **轮足四足** 上实现 **全向滚动、高台攀爬、摔倒恢复** 三类异构技能的 **单策略联合学习**，并用 **技能指示变量 + 高层选择器** 做自主模态切换（纯本体，无外部感知）。
- 强调 **轮–腿协同** 与 **DC 电机速度–扭矩包络** 约束，真机零样本完成 **1 m 高台** 与楼梯/坡道/高台连续任务链。

### AWARE（Deep Robotics M20 高动态反射避障）

- [AWARE](../entities/paper-aware-wheeled-legged-reflexive-evasion.md)（arXiv:2604.23761）面向 **快速动态障碍** 的反射式规避：高层用 RAR 威胁特征输出速度指令并 **硬切换** 低层双专家（导航全向 / 高动态敏捷）。
- 相对 MUJICA 的盲走多技能，AWARE 强调 **外源威胁下的滚动/踏步逃逸** 与导航↔反射连续切换；真机场景含抛箱、棍戳、脚踢。

### X2-N (Transformable Wheel-legged Humanoid)
- 结合了类人（humanoid）形态和轮腿（wheel-legged）形态的高自由度可变形机器人。
- 采用基于 RL 的控制框架统一处理 locomotion（运动）、transformation（变形）和 manipulation（操作）。
- 展示了出色的动态滑行、爬楼梯和包裹配送等 Loco-manipulation 能力。

## 关联页面

- [轮足四足机器人（四轮足）](../concepts/wheel-legged-quadruped.md)
- [Locomotion](./locomotion.md)
- [Loco-Manipulation](./loco-manipulation.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [MUJICA（轮足多技能统一控制）](../entities/paper-mujica-wheel-legged-multi-skill.md)
- [AWARE（轮足高动态反射式避障）](../entities/paper-aware-wheeled-legged-reflexive-evasion.md)

## 参考来源

- [X2-N: A Transformable Wheel-legged Humanoid Robot with Dual-mode Locomotion and Manipulation](../../sources/papers/x2n_transformable.md)
- [MUJICA（arXiv:2605.13058）](../../sources/papers/mujica_arxiv_2605_13058.md)
- [AWARE（arXiv:2604.23761）](../../sources/papers/aware_arxiv_2604_23761.md)
