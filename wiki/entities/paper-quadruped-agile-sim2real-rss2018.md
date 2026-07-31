---

type: entity
tags: [quadruped, sim2real, reinforcement-learning, legged, mit]
status: stable
summary: "RSS 2018：随机化动力学与感知，在仿真中学敏捷四足运动并迁移真机；建立早期 sim2real 扭矩/敏捷控制参照系。"
updated: 2026-07-28
venue: "RSS 2018"
related:
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../entities/paper-quadruped-torque-control-rl.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
---

# Sim-to-Real: Learning Agile Locomotion For Quadruped Robots（RSS 2018）

**一句话定义**：通过 **域随机化** 覆盖模型与传感不确定性，在仿真中训练 **高频敏捷四足运动策略**，并 **零样本或低开销** 迁移到实物平台，是后续大量 **sim2real 腿足工作** 的常用引用基线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Kp | Proportional Gain | PD 控制的位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | PD 控制的速度误差增益，抑制振荡 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |

## 为什么重要

- 帮助读者建立 **历史直觉**：为何工业与论文里长期并存 **「位置目标 + PD」** 与 **「力矩/扭矩直驱」** —— 取决于 **当时硬件、控制栈与安全文化**。
- 与 [Learning Torque Control…](./paper-quadruped-torque-control-rl.md) 对照：从 **随机化+敏捷** 到 **端到端扭矩 RL**，问题意识一脉相承。

## 核心机制（提炼）

- **随机化**：质量、摩擦、驱动、传感器偏差等，迫使策略 **不依赖单一名义模型**。
- **敏捷行为**：强调高动态步态与快速足端运动（具体以原文实验为准）。

```mermaid
flowchart LR
  rand["域随机化采样"]
  sim["仿真中 RL"]
  pol["敏捷策略"]
  rand --> sim
  sim --> pol
  real["真机四足"]
  pol -->|"sim2real"| real
```

## 与 Kp / Kd 设置的关系

- 若你的实现仍用 **PD 内环**：可把此文当作 **「随机化清单」** 的历史参照，再映射到当前栈（Isaac / MuJoCo）的 **等效参数名**。

## 实验与评测

- 量化指标、消融与 sim2real / 实机结果见 **原文 PDF** 与 [参考来源](#参考来源)；本页正文侧重方法结构与知识库交叉引用。

## 结论

**这篇的价值不在某个具体数字，而在确立了一条工程主线：用域随机化覆盖模型与传感的不确定性，让敏捷四足策略在仿真中训完就能低开销落到真机——后续腿足 sim2real 工作大多以它为引用基线。**

- 真正起作用的机制是 **随机化清单本身**：质量、摩擦、驱动、传感器偏差一并打散，迫使策略不依赖单一名义模型，而不是靠更精确的建模去追真机。
- 它同时解释了一个历史现象：**「位置目标 + PD」与「力矩直驱」为何长期并存**——取决于当时的硬件、控制栈与安全文化，而非哪种接口天然更优。
- 实用边界：若你的实现仍带 PD 内环，本文更适合当作 **随机化项的历史参照表**，需要映射到 Isaac / MuJoCo 等当前仿真栈的等效参数名后才可直接使用。
- 与 [四足扭矩控制 RL](./paper-quadruped-torque-control-rl.md) 构成一条演进线：从「随机化 + 敏捷」到「端到端扭矩 RL」，问题意识一脉相承，接口层不断下沉。
- 本页正文侧重方法结构与知识库交叉引用，**量化指标与消融以原文 PDF 为准**，不应据本页做性能比较。

## 与其他工作对比

- 正文已给出与相邻路线 / baseline 的 **定性对照**；定量表格与 ablation 见原文（[参考来源](#参考来源)）。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- Hwangbo et al., *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*, RSS 2018 proceedings [PDF p10](https://www.roboticsproceedings.org/rss14/p10.pdf)

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [Sim2Real 闭环误差分层工程](../queries/sim2real-closed-loop-engineering.md) — SysID 后再 DR 的早期敏捷迁移参照
- [Locomotion](../tasks/locomotion.md)
- [四足扭矩控制 RL](./paper-quadruped-torque-control-rl.md)
- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)

## 推荐继续阅读

- [RSS 2018 PDF（p10）](https://www.roboticsproceedings.org/rss14/p10.pdf)
