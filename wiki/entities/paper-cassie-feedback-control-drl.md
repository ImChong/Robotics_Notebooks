---

type: entity
tags: [cassie, reinforcement-learning, biped, sim2real, pd-control, oregon-state, ubc]
status: stable
summary: "高保真 Cassie 模型上的 DRL 反馈控制：跟踪参考运动、扰动与延迟鲁棒性，为「PD 目标空间」提供早期清晰 MDP 表述。"
updated: 2026-05-22
arxiv: "1803.05580"
related:
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../entities/paper-cassie-iterative-locomotion-sim2real.md
  - ../entities/paper-cassie-biped-versatile-locomotion-rl.md
  - ../entities/paper-deeprl-locomotion-action-space-sca2017.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
  - ../../sources/papers/deeprl_locomotion_action_space_sca2017.md
---

# Feedback Control For Cassie With Deep Reinforcement Learning

**一句话定义**：在 **贴近硬件的 Cassie 仿真** 中，把 **反馈跟踪参考步态** 表述为 MDP，用深度 RL 学得 **关节级目标 + 底层跟踪（PD 语义）** 的策略，并系统测试 **延迟、盲不规则地形、骨盆扰动** 等鲁棒性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| MDP | Markov Decision Process | 状态–动作–奖励–转移的标准序贯决策建模框架 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Kp | Proportional Gain | PD 控制的位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | PD 控制的速度误差增益，抑制振荡 |

## 为什么重要

- 较早把 **「为何用位置/速度目标 + PD，而不是一上来就扭矩」** 放在 **可学习、可部署** 的框架里讲清楚：低维目标空间 **缩小探索、利用已知内环**。
- 与图形学侧前史 [DeepRL 动作空间对比 SCA 2017](./paper-deeprl-locomotion-action-space-sca2017.md)（平面角色上四种动作语义对照）构成 **从角色动画到机器人** 的同主题前后参照。
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

## 实验与评测

- 量化指标、消融与 sim2real / 实机结果见 **原文 PDF** 与 [参考来源](#参考来源)；本页正文侧重方法结构与知识库交叉引用。

## 与其他工作对比

- 正文已给出与相邻路线 / baseline 的 **定性对照**；定量表格与 ablation 见原文（[参考来源](#参考来源)）。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- Xie et al., *Feedback Control For Cassie With Deep Reinforcement Learning*, [arXiv:1803.05580](https://arxiv.org/abs/1803.05580)

## 关联页面

- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)
- [DeepRL 动作空间对比 SCA 2017](./paper-deeprl-locomotion-action-space-sca2017.md)
- [Cassie 迭代式 sim2real](./paper-cassie-iterative-locomotion-sim2real.md)
- [Cassie 双足多技能 RL](./paper-cassie-biped-versatile-locomotion-rl.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/1803.05580.pdf)
