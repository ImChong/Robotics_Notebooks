---
type: entity
tags: [quadruped, sim2real, reinforcement-learning, legged]
status: stable
summary: "RSS 2018：随机化动力学与感知，在仿真中学敏捷四足运动并迁移真机；建立早期 sim2real 扭矩/敏捷控制参照系。"
updated: 2026-05-12
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

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- Hwangbo et al., *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*, RSS 2018 proceedings [PDF p10](https://www.roboticsproceedings.org/rss14/p10.pdf)

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [Locomotion](../tasks/locomotion.md)
- [四足扭矩控制 RL](./paper-quadruped-torque-control-rl.md)
- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)

## 推荐继续阅读

- [RSS 2018 PDF（p10）](https://www.roboticsproceedings.org/rss14/p10.pdf)
