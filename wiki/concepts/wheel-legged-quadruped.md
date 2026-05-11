---
type: concept
tags: [wheel-legged, quadruped, locomotion, hybrid, unitree]
status: complete
updated: 2026-05-11
related:
  - ../tasks/hybrid-locomotion.md
  - ../tasks/locomotion.md
  - ../entities/unitree.md
  - ../entities/robot-lab.md
  - ../entities/legged-gym.md
sources:
  - ../../sources/repos/robot_lab.md
summary: "轮足四足机器人在四条腿末端集成驱动轮，平地偏滚动效率，崎岖地形仍依赖足式步态；典型如 Unitree Go2W / B2W，仿真框架 robot_lab 将其与纯四足、人形并列注册。"
---

# 轮足四足机器人（四轮足 / Wheel-Legged Quadruped）

## 一句话定义

轮足四足机器人在四条腿末端集成驱动轮，平地偏滚动效率与能效，崎岖地形仍依赖足式步态；典型量产如 Unitree Go2W / B2W，仿真资产可按 [robot_lab](../entities/robot-lab.md) 机型表接入。

## 为什么重要

- **能效与续航**：相比同等尺度下持续高频踏步的纯足式，轮式成分可降低平坦工况能耗。
- **室内外过渡**：仓储、园区道路等结构化路面追求吞吐量；切入非结构化地带仍希望保留腿式越障能力。
- **研究与仿真资产**：与纯四足共用大量 URDF / RL locomotion 管线（速度指令跟踪、域随机化），在开源框架里常与「四足」「人形」并列作为独立机型注册。

## 形态要点

1. **混合运动学**：同一套腿上既要满足 **轮速–地面滚动约束**，又要满足 **抬腿–摆动–落地** 的步态周期；控制器需在滚动、踏步或二者混合之间调度。
2. **接触与滑移**：轮胎–地面摩擦、侧滑与制动不同于足端橡胶垫；状态估计需处理轮速计与 IMU 融合的可观测性问题。
3. **仿真建模**：除关节与连杆外，往往需明确 **轮半径、摩擦锥、滚动阻力**，否则 Sim2Real 在「滑行」模态上容易失真。

## 代表型号（与本库仿真入口）

下列型号出现在 IsaacLab 生态扩展框架 **robot_lab** 的机型枚举中（用于任务与资产划分），**不等同于性能排名或采购建议**：

| 类别（robot_lab） | 典型型号 |
|-------------------|-----------|
| 轮足（Wheeled） | Unitree **Go2W** / **B2W**、Deeprobotics **M20**、DDTRobot **Tita** |

仿真与任务扩展入口：[robot_lab](../entities/robot-lab.md)。

## 与相邻概念的关系

- **纯四足**：无驱动轮的足式平台，更偏崎岖地形与腿式 RL 基准；量产与研究机型分布见各硬件实体页。
- **[Hybrid Locomotion](../tasks/hybrid-locomotion.md)**：更广义的「多模态运动」（含人形变形、轮–腿切换等）；本页只覆盖 **四足底盘 + 足端轮** 这一子类。
- **人形轮足（如 X2-N）**：同为轮足混合，但是 **双足 + 上肢** 的可变形构型，任务侧重 loco-manipulation，参见 Hybrid Locomotion 中的代表系统描述。

## 关联页面

- [Hybrid Locomotion](../tasks/hybrid-locomotion.md)
- [Locomotion](../tasks/locomotion.md)
- [Unitree](../entities/unitree.md)
- [robot_lab](../entities/robot-lab.md)
- [legged_gym](../entities/legged-gym.md)

## 参考来源

- [robot_lab](../../sources/repos/robot_lab.md)
