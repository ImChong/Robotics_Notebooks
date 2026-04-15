---
type: entity
tags: [humanoid, hardware, platform, actuator]
related:
  - ../queries/hardware-comparison.md
  - ../tasks/locomotion.md
  - ../tasks/loco-manipulation.md
  - ../concepts/whole-body-control.md
  - ../comparisons/wbc-vs-rl.md
sources:
  - ../../sources/papers/humanoid_hardware.md
---

# 人形机器人（Humanoid Robot）

## 一句话定义

人形机器人是具有双足步行能力和类人形态（躯干 + 双臂 + 双腿）的机器人平台，兼顾移动能力与操作能力，是当前具身智能研究的核心载体。

## 为什么重要

人形机器人平台是连接算法研究与真实世界部署的关键桥梁：

> "人形平台的选择直接决定了你能运行哪些控制算法、能完成哪些任务、以及 sim2real 的难度。"

## 主流平台速览

| 平台 | 组织 | DOF | 执行器类型 | 当前状态 |
|------|------|-----|-----------|---------|
| Atlas | Boston Dynamics | 28 | 液压 | 研究平台 |
| Unitree H1 | Unitree | 20 | 刚性电机 | 商业可购 |
| Unitree G1 | Unitree | 23 | 刚性电机 | 商业可购 |
| Agility Digit | Agility Robotics | 20 | SEA | 商业部署 |
| Fourier GR-1 | Fourier Intelligence | 44 | 力控电机 | 商业可购 |

详细对比见：[主流人形机器人硬件对比](../queries/hardware-comparison.md)

## 执行器类型决定控制策略

- **液压**（Atlas）：高功率密度 → 适合 WBC + 高动态运动，sim2real 最难
- **刚性电机**（H1/G1）：轻量低成本 → 适合 RL 研究，力控精度有限
- **SEA**（Digit）：内置柔顺性 → 适合人机协作和接触丰富任务
- **QDD**（MIT Cheetah 风格）：高带宽力矩透明 → 适合精密接触控制

## 关联页面
- [主流人形机器人硬件对比](../queries/hardware-comparison.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [全身运动控制](../concepts/whole-body-control.md)
- [WBC vs RL 对比](../comparisons/wbc-vs-rl.md)
- [Unitree](./unitree.md)

## 参考来源
- [humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
