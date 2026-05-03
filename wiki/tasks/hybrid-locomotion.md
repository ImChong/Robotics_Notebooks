---
type: task
tags: [locomotion, hybrid, wheel-legged, transformable, whole-body]
status: in-progress
summary: "Hybrid Locomotion 关注结合不同运动模式（如轮腿结合、双足/四足切换）的机器人系统及其控制挑战。"
updated: 2026-05-03
sources:
  - ../../sources/papers/x2n_transformable.md
---

# Hybrid Locomotion (混合运动)

**混合运动（Hybrid Locomotion）**：机器人系统能通过不同的机械模式进行运动，最常见的是轮腿混合（Wheel-legged）以及形态可变（Transformable）的设计。

## 核心挑战

1. **模态切换（Mode Transition / Transformation）：** 如何在不同结构形态或运动模式（例如，双足行走模式切换到带轮滑行模式）之间实现平稳和高效的转换。
2. **全身协调（Whole-body Coordination）：** 在特定的形态下（特别是包含上肢和双腿的完整平台），如何实现下肢移动与上肢操作的协调。
3. **混合控制框架（Hybrid Control Framework）：** 传统控制中需要分别针对不同模态设计控制器并处理切换逻辑，现代趋势逐渐使用强化学习（RL）来提供统一的策略。

## 技术路线

- **基于强化学习的统一控制（RL-based Unified Control）：** 采用 RL 学习统一的全身控制（Whole-body control）策略，通过端到端的方式处理变形态、轮式移动、步行以及操作等多任务并发的需求。

## 代表系统

### X2-N (Transformable Wheel-legged Humanoid)
- 结合了类人（humanoid）形态和轮腿（wheel-legged）形态的高自由度可变形机器人。
- 采用基于 RL 的控制框架统一处理 locomotion（运动）、transformation（变形）和 manipulation（操作）。
- 展示了出色的动态滑行、爬楼梯和包裹配送等 Loco-manipulation 能力。

## 关联页面

- [Locomotion](./locomotion.md)
- [Loco-Manipulation](./loco-manipulation.md)
- [Whole-Body Control](../concepts/whole-body-control.md)

## 参考来源

- [X2-N: A Transformable Wheel-legged Humanoid Robot with Dual-mode Locomotion and Manipulation](../../sources/papers/x2n_transformable.md)
