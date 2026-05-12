---
type: entity
tags: [cassie, reinforcement-learning, biped, sim2real, pd-control]
status: stable
summary: "高保真 Cassie 模型上的 DRL 反馈控制：跟踪参考运动、扰动与延迟鲁棒性，为「PD 目标空间」提供早期清晰 MDP 表述。"
updated: 2026-05-12
related:
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../entities/paper-cassie-iterative-locomotion-sim2real.md
  - ../entities/paper-cassie-biped-versatile-locomotion-rl.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
---

# Feedback Control For Cassie With Deep Reinforcement Learning

**一句话定义**：在 **贴近硬件的 Cassie 仿真** 中，把 **反馈跟踪参考步态** 表述为 MDP，用深度 RL 学得 **关节级目标 + 底层跟踪（PD 语义）** 的策略，并系统测试 **延迟、盲不规则地形、骨盆扰动** 等鲁棒性。

## 为什么重要

- 较早把 **「为何用位置/速度目标 + PD，而不是一上来就扭矩」** 放在 **可学习、可部署** 的框架里讲清楚：低维目标空间 **缩小探索、利用已知内环**。
- 与后续 [迭代式 Cassie sim2real](./paper-cassie-iterative-locomotion-sim2real.md)、[双历史多技能](./paper-cassie-biped-versatile-locomotion-rl.md) 形成 **同一平台的方法演进链**。

## 核心机制（提炼）

- **参考运动条件化**：不同速度可由时间伸缩参考轨迹得到，策略学习 **反馈补偿**。
- **鲁棒性测试**：传感延迟、地形盲走、外力推拽等，作为 **同一控制器族** 的压力案例。

```mermaid
flowchart LR
  ref["参考步态<br/>时间伸缩"]
  pol["DRL 策略"]
  tgt["关节目标空间"]
  pd["PD 跟踪"]
  ref --> pol
  pol --> tgt --> pd
  cassie["Cassie 硬件或高保真模型"]
  pd --> cassie
```

## 与 Kp / Kd 设置的关系

- 读此文时把 **PD 当作已建模内环**：调 `Kp/Kd` 等价于改变 **策略看到的有效 plant**；过大增益会掩盖策略缺陷，过小则让策略扛全部扰动。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- Xie et al., *Feedback Control For Cassie With Deep Reinforcement Learning*, [arXiv:1803.05580](https://arxiv.org/abs/1803.05580)

## 关联页面

- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)
- [Cassie 迭代式 sim2real](./paper-cassie-iterative-locomotion-sim2real.md)
- [Cassie 双足多技能 RL](./paper-cassie-biped-versatile-locomotion-rl.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/1803.05580.pdf)
