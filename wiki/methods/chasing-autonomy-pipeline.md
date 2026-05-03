---
type: method
title: Chasing Autonomy Pipeline
tags: [robot-learning, humanoid, locomotion, reinforcement-learning, sim2real, motion-retargeting]
summary: "一种结合硬约束动态重定向与控制引导强化学习的流水线，利用单一人类演示实现高性能的人形机器人跑步与避障。"
---

# Chasing Autonomy Pipeline

**Chasing Autonomy Pipeline** 是由加州理工学院和 Unitree 团队（Olkin et al., 2026）提出的一套使人形机器人能够高性能奔跑的系统框架。它有效衔接了人体动作重定向（Motion Retargeting）与强化学习（RL），利用单一的人类演示动作生成高质量的参考库，并在真实世界上实现了快速、具有环境避障能力的跑步。

## 核心理念

该框架的核心洞见是：通过**带有硬约束的动态优化**处理人体动作数据，比直接进行运动学重定向能提供更好的 RL 训练参考（Reference Library）；在强化学习阶段，**目标条件（Goal-conditioned）和控制引导（Control-guided）的奖励结构**对于在真实环境中实现稳定的高动态运动（如跑步）至关重要。

## 主要技术路线

该系统包含三个核心阶段：

```text
单一人类运动演示 (Single Human Demonstration)
          ↓
阶段1：带硬约束的动态优化重定向 (Dynamic Retargeting with Hard Constraints)
       - 生成考虑机器人动力学和物理约束的改进周期性参考库
          ↓
阶段2：控制引导的强化学习 (Control Guided RL)
       - 目标条件与控制引导奖励
       - 追踪动态优化的参考数据
          ↓
阶段3：全感知与规划的自主栈集成 (Full Perception and Planning Autonomy Stack)
       - 室外环境避障
       - 真机部署 (Unitree G1)
```

## 关键技术

### 1. 带硬约束的动态重定向 (Dynamic Retargeting)
传统的人体动作重定向往往只考虑运动学映射，导致生成的动作对机器人而言可能不符合物理规律或动力学约束。该方法使用带有硬约束的优化例程，不仅完成运动学映射，还确保生成的周期性参考库符合机器人的硬件限制和动力学可行性。

### 2. 控制引导与目标条件奖励 (Control Guided Reward)
研究表明，不仅参考动作的质量重要，奖励函数的结构也直接影响速度追踪的效果。通过在 RL 训练中引入目标条件和控制层面的引导，策略不仅能更好地模仿参考动作，还能更稳健地响应上层速度指令。

### 3. 可控性与自主感知集成 (Controllability & Autonomy)
该控制策略不仅用于回放单一动作，还能无缝接入上层的感知与规划栈中。在硬件部署中，机器人能够根据 LiDAR 或视觉感知生成的实时目标，动态调整运行速度和方向，实现室外环境中的自主避障奔跑。

## 性能表现

- **高速度**：在 Unitree G1 硬件上实现了高达 3.3 m/s 的奔跑速度。
- **高耐力与鲁棒性**：成功在现实室外环境中穿越数百米。
- **完全自主的避障奔跑**：在全感知与规划栈的支持下，机器人能够实时避障并保持高速移动。

## 关联页面

- [Motion Retargeting (运动重定向)](../concepts/motion-retargeting.md)
- [Humanoid Locomotion (人形移动)](../tasks/humanoid-locomotion.md)
- [Reinforcement Learning (强化学习)](./reinforcement-learning.md)
- [Sim2Real (仿真到现实迁移)](../concepts/sim2real.md)
- [Reward Design (奖励设计)](../concepts/reward-design.md)
- [Unitree G1 (机器人实体)](../entities/unitree-g1.md)

## 参考来源
- [Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running](../../sources/papers/chasing_autonomy.md)
